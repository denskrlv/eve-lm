"""The interfaces of the swappable components (registered in ``eve.registry``).

Every component meets the others in a ``Batch``: a dict of tensors made by a data module's
``collate`` and read by the model, the losses and the metrics. The keys (B = batch size):

- ``enc_ids``, ``enc_mask`` (B, n): encoder token ids and their padding mask (True = real);
- ``dec_ids``, ``dec_mask``, ``dec_labels`` (B, n_dec): decoder inputs, mask and next-token
  labels (``IGNORE`` where nothing is predicted);
- ``tool_pos``, ``tool_mask`` (B, max_tools): where each ``<tool>`` marker is, padded tools
  False; ``intent`` (B,) and ``slots`` (B, n_slots): the question labels (task data only).

A component documents which keys it makes or needs; the span-corruption data has no tool or
question keys, so only the span loss and metrics fit it.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import torch
from torch import nn

from eve.train.config import TrainConfig

Batch = dict[str, torch.Tensor]
LossFn = Callable[[Any, Batch], dict[str, torch.Tensor]]
"""A loss: (model, batch) -> named scalar losses. The trainer backpropagates their sum, so each
term has weight 1. Registered as kind "loss"; a config's params are bound as keywords."""
MetricFn = LossFn
"""Val metrics: (model, batch) -> named scalars, averaged over the val batches and logged with a
``val_`` prefix, never trained on. Registered as kind "metrics"."""


class Items(Protocol):
    """A map-style dataset: a length and an item per index. It may also have
    ``set_epoch(epoch)``, which the trainer calls before each batch."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> Any: ...


class DataModule(Protocol):
    """Kind "data": the train and val sets and how to batch them."""

    train: Items
    val: Items

    def collate(self, items: Sequence[Any]) -> Batch:
        """Items of ``train`` or ``val`` -> one batch (see the module docstring for its keys)."""
        ...

    def batch_tokens(self, batch: Batch) -> int:
        """The real (unpadded) tokens of a batch, for the tok/s the trainer logs."""
        ...


class ConfiguredModel(Protocol):
    """Kind "model": an ``nn.Module`` class that a checkpoint can rebuild from its config.

    Its instances have ``cfg`` with ``to_dict()``; ``from_config`` turns that dict back into
    the same architecture, so a checkpoint needs no code path but the registered name.
    """

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> nn.Module:
        """A new model (random weights) of the architecture ``config`` describes."""
        ...


class OptimizerFactory(Protocol):
    """Kind "optimizer": builds the optimizer of a run from its ``TrainConfig``."""

    def __call__(
        self, model: nn.Module, cfg: TrainConfig, device: torch.device
    ) -> torch.optim.Optimizer: ...


class Schedule(Protocol):
    """Kind "schedule": the learning rate of optimizer step ``step`` (0-based)."""

    def __call__(self, step: int, cfg: TrainConfig) -> float: ...
