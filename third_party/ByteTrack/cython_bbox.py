"""NumPy adapter for cython_bbox.bbox_overlaps, with upstream inclusive pixels.

The +1 coordinate convention is intentional: changing it would change the
official ByteTrack matching costs for tiny boxes. This adapter replaces only
the unavailable compiled overlap extension, not assignment or tracking logic.
Formula verified against samson-wang/cython_bbox src/cython_bbox.pyx at
4874834e6de5c3a07c68d027edcaa62012d76ef0.
"""
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    queries = np.asarray(query_boxes, dtype=np.float64)
    if boxes.ndim != 2 or queries.ndim != 2 or boxes.shape[1] != 4 or queries.shape[1] != 4:
        raise ValueError("bbox_overlaps expects Nx4 and Kx4 arrays")
    left_top = np.maximum(boxes[:, None, :2], queries[None, :, :2])
    right_bottom = np.minimum(boxes[:, None, 2:], queries[None, :, 2:])
    size = np.maximum(0.0, right_bottom - left_top + 1.0)
    overlap = size[..., 0] * size[..., 1]
    area_a = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    area_b = (queries[:, 2] - queries[:, 0] + 1.0) * (queries[:, 3] - queries[:, 1] + 1.0)
    union = area_a[:, None] + area_b[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=(overlap > 0) & (union > 0))
