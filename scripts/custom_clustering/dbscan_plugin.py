"""Sample clustering plugin leveraging DBSCAN on the 5D embedding."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def cluster_embeddings(
    embeddings: torch.Tensor,
    batch_ids: torch.Tensor,
    global_indices: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    group_by_semantic: bool = True,
    eps: float = 0.45,
    min_samples: int = 15,
    bandwidth: Optional[float] = None,
    metadata: Optional[dict] = None,
    **_: dict,
) -> Tuple[Sequence[torch.Tensor], Sequence[int]]:
    """Cluster points based on their learned embedding using DBSCAN.

    Parameters
    ----------
    embeddings:
        Embedding vectors for the points to cluster.
    batch_ids:
        Batch index for every point (multiple spheres can be processed at once).
    global_indices:
        Original indices of the points with respect to the full point cloud.
    predicted_labels:
        Semantic label predicted by the network for each point.  The implementation
        can optionally cluster points per semantic class.
    group_by_semantic:
        Whether to run DBSCAN independently for each semantic class.  This mirrors
        the behaviour of the reference implementation which only clusters "thing"
        classes together.
    eps / min_samples:
        Standard DBSCAN parameters.  ``eps`` can be tuned using the paper results
        as guidance.  ``bandwidth`` is accepted for API compatibility but not used
        here.
    metadata:
        Optional dictionary populated by the training loop.  The function does not
        use it directly but it is part of the public interface for future
        extensions.
    """

    if metadata is None:
        metadata = {}

    # Convert tensors to NumPy to interact with scikit-learn.
    embeddings_np = _to_numpy(embeddings)
    batch_np = _to_numpy(batch_ids).astype(np.int64, copy=False)
    global_indices_np = _to_numpy(global_indices).astype(np.int64, copy=False)
    labels_np = _to_numpy(predicted_labels).astype(np.int64, copy=False)

    batches = np.unique(batch_np)
    if batches.size == 0:
        return [], []

    semantic_groups: Iterable[Tuple[Optional[int], np.ndarray]]
    if group_by_semantic:
        semantic_groups = (
            (label, labels_np == label)
            for label in np.unique(labels_np)
        )
    else:
        semantic_groups = ((None, np.ones_like(labels_np, dtype=bool)),)

    clusters: List[torch.Tensor] = []
    for _, semantic_mask in semantic_groups:
        if not np.any(semantic_mask):
            continue
        for batch in batches:
            batch_mask = (batch_np == batch) & semantic_mask
            if np.count_nonzero(batch_mask) < min_samples:
                continue

            local_embeddings = embeddings_np[batch_mask]
            local_indices = global_indices_np[batch_mask]

            algorithm = DBSCAN(eps=eps, min_samples=min_samples)
            local_labels = algorithm.fit_predict(local_embeddings)

            for cluster_id in np.unique(local_labels):
                if cluster_id < 0:
                    continue
                member_mask = local_labels == cluster_id
                if not np.any(member_mask):
                    continue
                clusters.append(torch.from_numpy(local_indices[member_mask]).long())

    cluster_types = [0 for _ in clusters]
    return clusters, cluster_types


__all__ = ["cluster_embeddings"]

