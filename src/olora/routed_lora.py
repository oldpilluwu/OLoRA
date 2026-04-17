"""
routed_lora.py -- The core multi-adapter LoRA implementation.

KEY IDEA:
  Standard LoRA adds one set of low-rank matrices (A, B) to each target layer.
  This module adds MULTIPLE sets -- one (A, B) pair per adapter job. During a
  forward pass, each sample in the batch is routed to the correct adapter's
  (A, B) matrices based on an `adapter_ids` tensor.

  This enables "fused" training: samples from different adapter jobs can be
  mixed into ONE batch, processed in ONE forward pass and ONE backward pass,
  and each adapter's LoRA parameters only receive gradients from their own
  samples.

Architecture overview:

  RoutedCausalLM (top-level wrapper)
    |
    +-- base_model (frozen GPT-2, no gradients)
          |
          +-- transformer.h[0].attn.c_attn  -->  replaced with RoutedLoRAConv1D
          +-- transformer.h[0].attn.c_proj  -->  replaced with RoutedLoRAConv1D
          +-- transformer.h[0].mlp.c_fc     -->  replaced with RoutedLoRAConv1D
          +-- ...  (same for every transformer block)

  Each RoutedLoRAConv1D holds:
    - base_layer:  the original frozen Conv1D weights (shared, no gradients)
    - lora_a[i]:   the "down-projection" matrix for adapter i  [in_features, rank]
    - lora_b[i]:   the "up-projection" matrix for adapter i    [rank, out_features]

  Forward pass for one layer:
    output = base_layer(x) + route_to_correct_adapter_delta(x)

ROUTING MECHANISM:
  Since HuggingFace's model.forward() doesn't natively accept an adapter_ids
  argument, we use a module-level global variable (_ACTIVE_ADAPTER_IDS) as a
  "routing context". Before calling base_model.forward(), RoutedCausalLM sets
  this global. Each RoutedLoRAConv1D reads it during its own forward() to know
  which adapter to apply to each sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import Conv1D


# ---------------------------------------------------------------------------
# Module-level routing context (global variable)
# ---------------------------------------------------------------------------
# This tensor tells each RoutedLoRAConv1D layer which adapter to use for each
# sample in the current batch. Shape: [batch_size], values are adapter indices.
# It's set by RoutedCausalLM.forward() before calling the base model, and
# read by every RoutedLoRAConv1D.forward() during the pass.
_ACTIVE_ADAPTER_IDS: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class RoutedLoraConfig:
    """
    Hyperparameters for the LoRA adapters.

    rank    -- Dimension of the low-rank bottleneck (smaller = fewer params, less expressive).
               E.g. rank=8 means the A matrix projects from in_features down to 8 dims.
    alpha   -- Scaling factor. The LoRA delta is multiplied by (alpha / rank).
               Higher alpha = stronger adapter effect relative to the base model.
    dropout -- Dropout probability applied to the input before the LoRA projection
               during training. Helps regularize the adapter.
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.00


def _set_active_adapter_ids(adapter_ids: torch.Tensor | None) -> torch.Tensor | None:
    """Set the global routing context. Returns the previous value (for restoring later)."""
    global _ACTIVE_ADAPTER_IDS
    previous = _ACTIVE_ADAPTER_IDS
    _ACTIVE_ADAPTER_IDS = adapter_ids
    return previous


def _get_active_adapter_ids() -> torch.Tensor:
    """Read the global routing context. Raises if it hasn't been set."""
    if _ACTIVE_ADAPTER_IDS is None:
        raise RuntimeError("adapter_ids routing context is missing for the routed LoRA forward pass.")
    return _ACTIVE_ADAPTER_IDS


class RoutedLoRAConv1D(nn.Module):
    """
    Replaces a single Conv1D layer in GPT-2 with a routed multi-adapter LoRA layer.

    For each adapter i, this module holds two small matrices:
      lora_a[i]: shape [in_features, rank]   -- "down-projection" (compresses input)
      lora_b[i]: shape [rank, out_features]  -- "up-projection"   (expands back)

    The forward pass computes:
      output = base_layer(x) + sum_over_adapters( route(x, adapter_i) )

    where route(x, adapter_i) only applies to samples in the batch that belong
    to adapter i (determined by the adapter_ids routing context).
    """
    def __init__(
        self,
        base_layer: Conv1D,
        adapter_names: Iterable[str],
        config: RoutedLoraConfig,
    ) -> None:
        super().__init__()

        # Keep the original layer but freeze it (no gradients for base weights).
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)

        adapter_names = tuple(adapter_names)
        self.adapter_names = ()  # Filled via add_adapter() so init and online insertion share one path.
        self.rank = config.rank
        self.scaling = config.alpha / config.rank   # LoRA output is multiplied by this
        self.dropout_p = config.dropout

        # Conv1D in GPT-2 stores weights as [in_features, out_features]
        # (transposed compared to nn.Linear).
        in_features, out_features = self.base_layer.weight.shape
        self.in_features = in_features
        self.out_features = out_features

        # Create one (A, B) pair per adapter.
        # lora_a[i] is the down-projection for adapter i.
        # lora_b[i] is the up-projection for adapter i (initialized to zeros so
        # the adapter starts as a no-op: A @ zeros = zeros).
        self.lora_a = nn.ParameterList()
        self.lora_b = nn.ParameterList()
        for adapter_name in adapter_names:
            self.add_adapter(adapter_name)

    def _new_adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Create one LoRA (A, B) pair on the same device/dtype as the base layer."""
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        lora_a = nn.Parameter(torch.empty(self.in_features, self.rank, device=device, dtype=dtype))
        lora_b = nn.Parameter(torch.zeros(self.rank, self.out_features, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        return lora_a, lora_b

    def add_adapter(self, adapter_name: str) -> int:
        """
        Append a brand-new adapter to this routed layer without rebuilding it.

        Returns the integer adapter index assigned to the new adapter.
        """
        if adapter_name in self.adapter_names:
            raise ValueError(f"Adapter '{adapter_name}' already exists in this routed layer.")

        lora_a, lora_b = self._new_adapter_parameters()
        self.lora_a.append(lora_a)
        self.lora_b.append(lora_b)
        self.adapter_names = (*self.adapter_names, adapter_name)
        return len(self.adapter_names) - 1

    def adapter_parameters(self, adapter_index: int) -> list[nn.Parameter]:
        """Return the trainable parameters for one specific adapter."""
        return [self.lora_a[adapter_index], self.lora_b[adapter_index]]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-sample adapter routing.

        Args:
          hidden_states: shape [batch_size, seq_len, in_features]
                         (or [batch_size, in_features] for non-sequence inputs)

        Returns:
          output: same shape as base_layer(hidden_states), with the appropriate
                  LoRA delta added per sample.

        How it works:
          1. Run the frozen base layer on ALL samples -> base_output
          2. Read adapter_ids from the global routing context
          3. For each unique adapter index in the batch:
             a. Mask out just the samples belonging to this adapter
             b. Compute their LoRA delta:  delta = (x @ A) @ B * scaling
             c. Write the delta into the corresponding positions
          4. Return base_output + delta
        """
        # Step 1: Full base layer forward (frozen, shared across all adapters).
        base_output = self.base_layer(hidden_states)

        # Step 2: Read which adapter each sample in the batch belongs to.
        # adapter_ids shape: [batch_size], e.g. [0, 0, 1, 1] for 2 adapters with 2 samples each.
        adapter_ids = _get_active_adapter_ids()

        # Sanity checks.
        if adapter_ids.dim() != 1:
            raise ValueError(f"adapter_ids must be rank-1, got shape {tuple(adapter_ids.shape)}.")
        if adapter_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "adapter_ids batch dimension must match the model batch dimension: "
                f"{adapter_ids.shape[0]} != {hidden_states.shape[0]}"
            )

        # Step 3: Compute the LoRA delta for each adapter present in this batch.
        # Start with zeros; we'll fill in each adapter's contribution.
        delta = torch.zeros_like(base_output)
        for adapter_index_tensor in adapter_ids.unique(sorted=True):
            adapter_index = int(adapter_index_tensor.item())
            if adapter_index < 0 or adapter_index >= len(self.adapter_names):
                raise IndexError(f"Adapter index {adapter_index} is out of range for routed LoRA modules.")

            # Boolean mask: which samples in the batch belong to this adapter.
            # e.g. if adapter_ids = [0, 0, 1, 1] and adapter_index = 1,
            # then sample_mask = [False, False, True, True].
            sample_mask = adapter_ids == adapter_index

            # Extract just this adapter's samples.
            adapter_hidden = hidden_states[sample_mask]

            # Apply dropout during training for regularization.
            if self.training and self.dropout_p > 0:
                adapter_hidden = F.dropout(adapter_hidden, p=self.dropout_p, training=True)

            # LoRA computation: delta = (x @ A) @ B * scaling
            # x @ A: [samples, seq_len, in_features] @ [in_features, rank] -> [samples, seq_len, rank]
            low_rank = torch.matmul(adapter_hidden, self.lora_a[adapter_index])
            # low_rank @ B: [samples, seq_len, rank] @ [rank, out_features] -> [samples, seq_len, out_features]
            adapter_delta = torch.matmul(low_rank, self.lora_b[adapter_index]) * self.scaling

            # Write this adapter's delta back into the correct batch positions.
            delta[sample_mask] = adapter_delta

        # Step 4: Add the routed LoRA delta to the base output.
        return base_output + delta


class RoutedCausalLM(nn.Module):
    """
    Top-level wrapper that turns a HuggingFace causal LM into a multi-adapter
    routed LoRA model.

    What it does during __init__:
      1. Takes the base GPT-2 model
      2. Finds all Conv1D layers whose names match target_modules (e.g. "c_attn", "c_proj", "c_fc")
      3. Replaces each one with a RoutedLoRAConv1D (which keeps the frozen base + adds LoRA params)
      4. Freezes ALL base model parameters
      5. Unfreezes ONLY the LoRA A and B matrices

    What it does during forward:
      1. Sets the global adapter_ids routing context
      2. Calls base_model.forward() (which internally hits the replaced RoutedLoRAConv1D layers)
      3. Restores the previous routing context

    Attributes:
      base_model      -- The original HuggingFace model (with layers replaced in-place)
      adapter_names   -- Tuple of adapter names, e.g. ("ag_news", "emotion")
      adapter_index   -- Dict mapping adapter name -> integer index, e.g. {"ag_news": 0, "emotion": 1}
      routed_modules  -- List of all RoutedLoRAConv1D layers that were injected
    """
    def __init__(
        self,
        base_model: nn.Module,
        adapter_names: list[str],
        target_modules: Iterable[str],
        config: RoutedLoraConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config   # Expose the HF config for compatibility
        self.adapter_names = tuple(adapter_names)
        # Maps adapter name to its integer index (used in adapter_ids tensors).
        self.adapter_index = {name: index for index, name in enumerate(self.adapter_names)}
        # Which layer names to replace (e.g. {"c_attn", "c_proj", "c_fc"} for GPT-2).
        self.target_modules = set(target_modules)
        # Keeps track of all replaced layers so we can easily collect their parameters.
        self.routed_modules: list[RoutedLoRAConv1D] = []

        self._replace_target_modules(config)
        self._freeze_non_lora_parameters()

    def _replace_target_modules(self, config: RoutedLoraConfig) -> None:
        """
        Walk through the base model and replace matching Conv1D layers with
        RoutedLoRAConv1D layers.

        For GPT-2, this replaces layers like:
          transformer.h.0.attn.c_attn  (query/key/value projection)
          transformer.h.0.attn.c_proj  (attention output projection)
          transformer.h.0.mlp.c_fc     (MLP first layer)
        """
        for module_name, module in list(self.base_model.named_modules()):
            if not module_name:
                continue

            # Get the leaf name (e.g. "c_attn" from "transformer.h.0.attn.c_attn").
            child_name = module_name.rsplit(".", maxsplit=1)[-1]

            # Only replace if: (a) the name is in our target list, and
            # (b) it's actually a Conv1D layer (not some other module with the same name).
            if child_name not in self.target_modules or not isinstance(module, Conv1D):
                continue

            # Find the parent module so we can swap the child.
            parent = self._resolve_parent_module(module_name)
            # Create the routed replacement (wraps the original frozen layer + adds LoRA params).
            routed = RoutedLoRAConv1D(module, self.adapter_names, config)
            # Swap it in: parent.c_attn = routed
            setattr(parent, child_name, routed)
            self.routed_modules.append(routed)

    def _resolve_parent_module(self, module_name: str) -> nn.Module:
        """
        Given a dotted module name like "transformer.h.0.attn.c_attn",
        return the parent module (transformer.h.0.attn).
        """
        parts = module_name.split(".")
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    def _freeze_non_lora_parameters(self) -> None:
        """
        Freeze everything, then selectively unfreeze just the LoRA matrices.

        After this:
          - All base model weights:  requires_grad = False (frozen)
          - All lora_a[i], lora_b[i]:  requires_grad = True (trainable)
        """
        # First, freeze absolutely everything in the base model.
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        # Then unfreeze just the LoRA A and B parameters.
        for routed in self.routed_modules:
            for parameter in routed.parameters():
                parameter.requires_grad = False
            for lora_parameter in list(routed.lora_a) + list(routed.lora_b):
                lora_parameter.requires_grad = True

    def adapter_parameters(self, adapter_name: str) -> list[nn.Parameter]:
        """
        Collect all trainable parameters for one adapter across all layers.

        E.g. adapter_parameters("ag_news") returns the lora_a[0] and lora_b[0]
        from every RoutedLoRAConv1D layer. This is what we pass to the adapter's
        optimizer so it only updates this adapter's weights.
        """
        adapter_index = self.adapter_index[adapter_name]
        parameters: list[nn.Parameter] = []
        for routed in self.routed_modules:
            parameters.extend(routed.adapter_parameters(adapter_index))
        return parameters

    def add_adapter(self, adapter_name: str) -> int:
        """
        Add a new adapter across all routed layers without restarting the model.

        This is the core primitive Phase C needs for online job insertion:
        a job can arrive after training has already started, receive freshly
        initialized LoRA weights, and join subsequent fused steps immediately.
        """
        if adapter_name in self.adapter_index:
            raise ValueError(f"Adapter '{adapter_name}' already exists.")

        adapter_index = len(self.adapter_names)
        for routed in self.routed_modules:
            routed.add_adapter(adapter_name)

        self.adapter_names = (*self.adapter_names, adapter_name)
        self.adapter_index[adapter_name] = adapter_index

        for parameter in self.adapter_parameters(adapter_name):
            parameter.requires_grad = True
        return adapter_index

    def forward(self, *args, adapter_ids: torch.Tensor | None = None, **kwargs):
        """
        Forward pass with adapter routing.

        Args:
          adapter_ids: [batch_size] tensor of integer adapter indices.
                       E.g. [0, 0, 1, 1] means samples 0-1 use adapter 0 ("ag_news")
                       and samples 2-3 use adapter 1 ("emotion").
          *args, **kwargs: Passed through to the HuggingFace model's forward().
                           Typically includes input_ids, attention_mask, labels.

        Returns:
          The HuggingFace model output (CausalLMOutput with .loss and .logits).
        """
        if adapter_ids is None:
            raise ValueError("RoutedCausalLM.forward requires adapter_ids for all training and evaluation calls.")

        # Set the global routing context so RoutedLoRAConv1D layers know which
        # adapter to use for each sample. Save the previous value to restore after.
        previous = _set_active_adapter_ids(adapter_ids)
        try:
            # This calls base_model.forward(), which internally passes through
            # all the replaced RoutedLoRAConv1D layers. Each one reads adapter_ids
            # from the global context and routes accordingly.
            return self.base_model(*args, **kwargs)
        finally:
            # Always restore the previous routing context (important for nested calls).
            _set_active_adapter_ids(previous)
