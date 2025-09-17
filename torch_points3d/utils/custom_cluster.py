"""Utility helpers to load optional custom clustering plugins.

This module makes it possible to plug external clustering algorithms into the
existing PointGroup models without modifying the model internals.  A plugin can
be defined either as a standard Python module that exposes a callable or as a
Python file on disk.  The callable is expected to return a list of point index
collections describing the discovered clusters.  Optionally, the callable can
also return the cluster types that should be associated with each cluster.

The entry-point of a plugin is resolved from a string such as
``"my.module:cluster_embeddings"`` or a dictionary configuration with the
following fields:

```
{
    "target": "path.to.module:callable",  # or "path/to/file.py:callable"
    "kwargs": {"eps": 0.5},               # Optional keyword arguments that
                                            # will always be provided to the
                                            # callable.
    "default_type": 0                      # Default cluster type id used when
                                            # the callable does not specify
                                            # cluster types explicitly.
}
```

The callable receives tensors that live on the same device as the model, so it
can decide to run entirely on the GPU or to move data back to the CPU.  The
return value is flexible: either a sequence of clusters or a ``(clusters,
cluster_types)`` tuple.  Clusters can be expressed as any iterable of indices
(``torch.Tensor``, ``numpy.ndarray`` or a Python sequence).  The helper takes
care of converting them to ``torch.Tensor`` instances placed on the correct
device so that the rest of the code base can remain unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

try:  # OmegaConf is an optional dependency at import time.
    from omegaconf import DictConfig, OmegaConf

    _OMEGACONF_AVAILABLE = True
except Exception:  # pragma: no cover - OmegaConf might not be installed yet.
    DictConfig = None
    OmegaConf = None
    _OMEGACONF_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def _maybe_resolve_env(value: Optional[str]) -> Optional[str]:
    """Resolve values of the form ``"env:MY_VAR"`` using environment vars."""

    if not value:
        return value

    if value.startswith("env:"):
        env_name = value.split(":", 1)[1]
        resolved = os.environ.get(env_name)
        if not resolved:
            raise ValueError(
                f"Requested to resolve environment variable '{env_name}' for "
                "the clustering plugin, but it is not defined."
            )
        return resolved
    return value


def _load_module_from_path(path: str, module_name: Optional[str] = None) -> ModuleType:
    """Load a Python module from an explicit file path."""

    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot import clustering plugin from '{path}' (file not found)")

    if not module_name:
        module_name = f"tp3d_cluster_plugin_{module_path.stem}"

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create a module spec for '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_callable(target: str, fallback_function: str = "cluster_embeddings") -> Callable[..., Any]:
    """Load a callable either from a module path or a python file."""

    target = _maybe_resolve_env(target)
    if not target:
        raise ValueError("No target specified for the clustering plugin")

    module_ref, _, func_name = target.partition(":")
    if not func_name:
        func_name = fallback_function

    if module_ref.endswith(".py") and Path(module_ref).exists():
        module = _load_module_from_path(module_ref)
    else:
        module = importlib.import_module(module_ref)

    if not hasattr(module, func_name):
        raise AttributeError(
            f"The clustering plugin '{module.__name__}' does not expose a callable named '{func_name}'"
        )

    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(
            f"The attribute '{func_name}' inside '{module.__name__}' is not callable and cannot be used as a plugin"
        )

    return func


def _normalize_spec(spec: Any) -> Tuple[Optional[str], MutableMapping[str, Any]]:
    """Normalise the configuration representation into a dictionary."""

    if spec is None:
        return None, {}

    if _OMEGACONF_AVAILABLE and isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)

    if isinstance(spec, Mapping):
        spec_dict = dict(spec)
        target = spec_dict.pop("target", None) or spec_dict.pop("module", None) or spec_dict.pop("path", None)
        target = _maybe_resolve_env(target)
        return target, spec_dict

    if isinstance(spec, str):
        target = _maybe_resolve_env(spec)
        return target, {}

    raise TypeError(
        "Unsupported configuration type for clustering plugin. Expected a string, a mapping or OmegaConf DictConfig."
    )


def _ensure_iterable(value: Any) -> Sequence[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _to_long_tensor(values: Sequence[Any], device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.long)

    return torch.as_tensor(list(values), device=device, dtype=torch.long)


def _to_uint8_tensor(values: Sequence[Any], device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.uint8)

    return torch.as_tensor(list(values), device=device, dtype=torch.uint8)


def _normalise_cluster_output(
    result: Any, device: torch.device, default_type: int
) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
    """Normalise outputs from a clustering plugin into tensors."""

    cluster_data: Any
    cluster_types: Any

    if isinstance(result, tuple) and len(result) == 2:
        cluster_data, cluster_types = result
    else:
        cluster_data, cluster_types = result, None

    clusters: list[torch.Tensor] = []
    for cluster in _ensure_iterable(cluster_data):
        if cluster is None:
            continue
        tensor_cluster = _to_long_tensor(cluster, device)
        if tensor_cluster.numel() == 0:
            continue
        clusters.append(tensor_cluster)

    if not clusters:
        empty_type = torch.empty(0, dtype=torch.uint8, device=device)
        return clusters, empty_type

    if cluster_types is None:
        cluster_types_tensor = torch.full((len(clusters),), int(default_type), dtype=torch.uint8, device=device)
        return clusters, cluster_types_tensor

    cluster_types_values = _ensure_iterable(cluster_types)
    if len(cluster_types_values) == 1 and len(clusters) > 1:
        cluster_types_tensor = torch.full((len(clusters),), int(cluster_types_values[0]), dtype=torch.uint8, device=device)
        return clusters, cluster_types_tensor

    if len(cluster_types_values) not in (0, len(clusters)):
        raise ValueError(
            "The clustering plugin returned a number of cluster types that does not match the number of clusters."
        )

    cluster_types_tensor = _to_uint8_tensor(cluster_types_values, device)
    return clusters, cluster_types_tensor


@dataclass
class CustomClusterAdapter:
    """Callable wrapper around a user provided clustering function."""

    func: Callable[..., Any]
    static_kwargs: Mapping[str, Any]
    default_type: int

    def __call__(self, **kwargs: Any) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
        call_kwargs = dict(self.static_kwargs)
        call_kwargs.update(kwargs)

        # Users may rely on this metadata dictionary to adapt their behaviour
        # without altering the public function signature.  If the user already
        # provided a metadata key through ``static_kwargs`` we respect it.
        metadata = call_kwargs.setdefault("metadata", {}) or {}
        if not isinstance(metadata, MutableMapping):  # pragma: no cover - defensive programming
            metadata = {"value": metadata}
            call_kwargs["metadata"] = metadata

        result = self.func(**call_kwargs)
        device = kwargs.get("global_indices")
        if isinstance(device, torch.Tensor):
            device = device.device
        else:
            device = torch.device("cpu")
        return _normalise_cluster_output(result, device, self.default_type)


def resolve_custom_clusterer(spec: Any) -> Optional[CustomClusterAdapter]:
    """Resolve a clustering plugin from a configuration specification.

    Parameters
    ----------
    spec:
        Either ``None`` (no plugin), a string describing the target module, or a
        mapping/DictConfig following the structure documented at the top of this
        file.
    """

    target, extra = _normalize_spec(spec)
    if not target:
        return None

    default_type = int(extra.pop("default_type", 0))
    static_kwargs = extra.pop("kwargs", {}) or {}

    if extra:
        LOGGER.debug("Ignoring unrecognised keys in custom clustering spec: %s", sorted(extra.keys()))

    func = _load_callable(target)
    LOGGER.info("Loaded custom clustering plugin '%s' with default type %s", target, default_type)
    return CustomClusterAdapter(func=func, static_kwargs=static_kwargs, default_type=default_type)


__all__ = ["CustomClusterAdapter", "resolve_custom_clusterer"]

