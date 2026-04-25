from __future__ import annotations

__all__ = [
]

from functools import partial

from columnflow.util import MockModule, maybe_import, DotDict, classproperty
from columnflow.types import Callable
from collections.abc import Container

from columnflow.columnar_util import Route, EMPTY_FLOAT, EMPTY_INT

torch = maybe_import("torch")
torchdata = maybe_import("torchdata")
np = maybe_import("numpy")
ak = maybe_import("awkward")
import law


model_clss: DotDict[str, torch.nn.Module] = DotDict()

