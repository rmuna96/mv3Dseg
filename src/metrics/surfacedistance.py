# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt

from monai.metrics.utils import (
    do_metric_reduction,
    get_mask_edges,
    ignore_background,
)
from monai.utils import MetricReduction, convert_data_type

from .metrics import CumulativeIterationMetric


class SurfaceDistanceMetric(CumulativeIterationMetric):
    """
    Compute Surface Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both symmetric and asymmetric surface distance calculation.
    Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        symmetric: whether to calculate the symmetric average surface distance between
            `seg_pred` and `seg_gt`. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        include_background: bool = False,
        symmetric: bool = False,
        distance_metric: str = "euclidean",
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.symmetric = symmetric
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.
            kwargs: additional parameters, e.g. ``spacing`` should be passed to correctly compute the metric.
                ``spacing``: spacing of pixel (or voxel). This parameter is relevant only
                if ``distance_metric`` is set to ``"euclidean"``.
                If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
                the length of the sequence must be equal to the image dimensions.
                This spacing will be used for all images in the batch.
                If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
                If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
                else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
                for all images in batch. Defaults to ``None``.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        if y_pred.dim() < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        # compute (BxC) for each channel for each batch
        return compute_average_surface_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            symmetric=self.symmetric,
            distance_metric=self.distance_metric,
            spacing=kwargs.get("spacing"),
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_average_surface_distance`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_average_surface_distance(
    y_pred: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    include_background: bool = False,
    symmetric: bool = False,
    distance_metric: str = "euclidean",
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None = None,
) -> torch.Tensor:
    """
    This function is used to compute the Average Surface Distance from `y_pred` to `y`
    under the default setting.
    In addition, if sets ``symmetric = True``, the average symmetric surface distance between
    these two inputs will be returned.
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        symmetric: whether to calculate the symmetric average surface distance between
            `seg_pred` and `seg_gt`. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
            the length of the sequence must be equal to the image dimensions. This spacing will be used for all images in the batch.
            If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
            If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
            else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
            for all images in batch. Defaults to ``None``.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    asd = np.empty((batch_size, n_class))

    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt) = get_mask_edges(y_pred[b, c], y[b, c])
        if not np.any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan/inf distance.")
        if not np.any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan/inf distance.")
        surface_distance = get_surface_distance(
            edges_pred, edges_gt, distance_metric=distance_metric, spacing=spacing_list[b]
        )
        if symmetric:
            surface_distance_2 = get_surface_distance(
                edges_gt, edges_pred, distance_metric=distance_metric, spacing=spacing_list[b]
            )
            surface_distance = np.concatenate([surface_distance, surface_distance_2])
        asd[b, c] = np.nan if surface_distance.shape == (0,) else surface_distance.mean()

    return convert_data_type(asd, output_type=torch.Tensor, device=y_pred.device, dtype=torch.float)[0]


def prepare_spacing(
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None,
    batch_size: int,
    img_dim: int,
) -> Sequence[None | int | float | np.ndarray | Sequence[int | float]]:
    """
    This function is used to prepare the `spacing` parameter to include batch dimension for the computation of
    surface distance, hausdorff distance or surface dice.

    An example with batch_size = 4 and img_dim = 3:
    input spacing = None -> output spacing = [None, None, None, None]
    input spacing = 0.8 -> output spacing = [0.8, 0.8, 0.8, 0.8]
    input spacing = [0.8, 0.5, 0.9] -> output spacing = [[0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9]]
    input spacing = [0.8, 0.7, 1.2, 0.8] -> output spacing = [0.8, 0.7, 1.2, 0.8] (same as input)

    An example with batch_size = 3 and img_dim = 3:
    input spacing = [0.8, 0.5, 0.9] ->
    output spacing = [[0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9]]

    Args:
        spacing: can be a float, a sequence of length `img_dim`, or a sequence with length `batch_size`
        that includes floats or sequences of length `img_dim`.

    Raises:
        ValueError: when `spacing` is a sequence of sequence, where the outer sequence length does not
        equal `batch_size` or inner sequence length does not equal `img_dim`.

    Returns:
        spacing: a sequence with length `batch_size` that includes integers, floats or sequences of length `img_dim`.
    """
    if spacing is None or isinstance(spacing, (int, float)):
        return list([spacing] * batch_size)
    if isinstance(spacing, (Sequence, np.ndarray)):
        if any(not isinstance(s, type(spacing[0])) for s in list(spacing)):
            raise ValueError(f"if `spacing` is a sequence, its elements should be of same type, got {spacing}.")
        if isinstance(spacing[0], (Sequence, np.ndarray)):
            if len(spacing) != batch_size:
                raise ValueError(
                    "if `spacing` is a sequence of sequences, "
                    f"the outer sequence should have same length as batch size ({batch_size}), got {spacing}."
                )
            if any(len(s) != img_dim for s in list(spacing)):
                raise ValueError(
                    "each element of `spacing` list should either have same length as"
                    f"image dim ({img_dim}), got {spacing}."
                )
            if not all(isinstance(i, (int, float)) for s in list(spacing) for i in list(s)):
                raise ValueError(
                    f"if `spacing` is a sequence of sequences or 2D np.ndarray, "
                    f"the elements should be integers or floats, got {spacing}."
                )
            return list(spacing)
        if isinstance(spacing[0], (int, float)):
            if len(spacing) != img_dim:
                raise ValueError(
                    f"if `spacing` is a sequence of numbers, "
                    f"it should have same length as image dim ({img_dim}), got {spacing}."
                )
            return [spacing for _ in range(batch_size)]  # type: ignore
        raise ValueError(f"`spacing` is a sequence of elements with unsupported type: {type(spacing[0])}")
    raise ValueError(
        f"`spacing` should either be a number, a sequence of numbers or a sequence of sequences, got {spacing}."
    )

def get_surface_distance(
    seg_pred: np.ndarray,
    seg_gt: np.ndarray,
    distance_metric: str = "euclidean",
    spacing: int | float | np.ndarray | Sequence[int | float] | None = None,
) -> np.ndarray:
    """
    This function is used to compute the surface distances from `seg_pred` to `seg_gt`.

    Args:
        seg_pred: the edge of the predictions.
        seg_gt: the edge of the ground truth.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.

            - ``"euclidean"``, uses Exact Euclidean distance transform.
            - ``"chessboard"``, uses `chessboard` metric in chamfer type of transform.
            - ``"taxicab"``, uses `taxicab` metric in chamfer type of transform.
        spacing: spacing of pixel (or voxel) along each axis. If a sequence, must be of
            length equal to the image dimensions; if a single number, this is used for all axes.
            If ``None``, spacing of unity is used. Defaults to ``None``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            Several input options are allowed: (1) If a single number, isotropic spacing with that value is used.
            (2) If a sequence of numbers, the length of the sequence must be equal to the image dimensions.
            (3) If ``None``, spacing of unity is used. Defaults to ``None``.

    Note:
        If seg_pred or seg_gt is all 0, may result in nan/inf distance.

    """

    if not np.any(seg_gt):
        dis = np.inf * np.ones_like(seg_gt)
    else:
        if not np.any(seg_pred):
            dis = np.inf * np.ones_like(seg_gt)
            return np.asarray(dis[seg_gt])
        if distance_metric == "euclidean":
            dis = distance_transform_edt(~seg_gt, sampling=spacing)
        elif distance_metric in {"chessboard", "taxicab"}:
            dis = distance_transform_cdt(~seg_gt, metric=distance_metric)
        else:
            raise ValueError(f"distance_metric {distance_metric} is not implemented.")

    return np.asarray(dis[seg_pred])

