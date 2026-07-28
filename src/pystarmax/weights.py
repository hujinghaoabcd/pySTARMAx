# SPDX-FileCopyrightText: 2026 Jinghao Hu
# SPDX-License-Identifier: MIT

"""Spatial-weight construction and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from pystarmax._validation import FloatArray, as_float_matrix, validate_nonnegative_int


def row_standardize(matrix: Any, *, zero_policy: str = "keep") -> FloatArray:
    """Row-standardize a square spatial-weight matrix.

    Parameters
    ----------
    matrix:
        Square matrix to standardize.
    zero_policy:
        ``"keep"`` retains all-zero rows; ``"raise"`` rejects them.
    """
    values = as_float_matrix(matrix, name="matrix")
    if values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    row_sums = values.sum(axis=1)
    zero_rows = np.isclose(row_sums, 0.0)
    if zero_policy not in {"keep", "raise"}:
        raise ValueError("zero_policy must be 'keep' or 'raise'")
    if zero_policy == "raise" and np.any(zero_rows):
        raise ValueError("matrix contains rows with zero total weight")
    output = np.zeros_like(values, dtype=float)
    nonzero = ~zero_rows
    output[nonzero] = values[nonzero] / row_sums[nonzero, None]
    return output


def lattice_weights(rows: int, columns: int, *, rook: bool = True) -> FloatArray:
    """Create a binary rectangular-lattice adjacency matrix."""
    rows = validate_nonnegative_int(rows, name="rows")
    columns = validate_nonnegative_int(columns, name="columns")
    if rows == 0 or columns == 0:
        raise ValueError("rows and columns must be positive")
    n = rows * columns
    adjacency = np.zeros((n, n), dtype=float)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if not rook:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for row in range(rows):
        for column in range(columns):
            source = row * columns + column
            for row_offset, column_offset in offsets:
                other_row = row + row_offset
                other_column = column + column_offset
                if 0 <= other_row < rows and 0 <= other_column < columns:
                    target = other_row * columns + other_column
                    adjacency[source, target] = 1.0
    return adjacency


def distance_weights(
    coordinates: Any,
    *,
    threshold: float | None = None,
    power: float = 1.0,
    include_diagonal: bool = False,
    standardize: bool = True,
) -> FloatArray:
    """Construct inverse-distance weights from two-dimensional coordinates."""
    points = as_float_matrix(coordinates, name="coordinates")
    if points.shape[1] < 1:
        raise ValueError("coordinates must contain at least one dimension")
    if power <= 0:
        raise ValueError("power must be positive")
    if threshold is not None and threshold <= 0:
        raise ValueError("threshold must be positive")
    distances = cdist(points, points)
    with np.errstate(divide="ignore"):
        weights = np.where(distances > 0, distances ** (-power), 0.0)
    if threshold is not None:
        weights[distances > threshold] = 0.0
    if include_diagonal:
        np.fill_diagonal(weights, 1.0)
    else:
        np.fill_diagonal(weights, 0.0)
    return row_standardize(weights) if standardize else weights


@dataclass(frozen=True, slots=True)
class SpatialWeights:
    """Immutable ordered collection of spatial-lag matrices.

    Matrix zero is conventionally the identity matrix. All matrices use the
    same location order.
    """

    matrices: tuple[FloatArray, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.matrices:
            raise ValueError("at least one spatial-weight matrix is required")
        size = self.matrices[0].shape[0]
        checked: list[FloatArray] = []
        for index, matrix in enumerate(self.matrices):
            values = as_float_matrix(matrix, name=f"matrices[{index}]")
            if values.shape != (size, size):
                raise ValueError(
                    "all spatial-weight matrices must share one square shape"
                )
            frozen = values.copy()
            frozen.setflags(write=False)
            checked.append(frozen)
        if len(self.names) != len(checked):
            raise ValueError("names must match the number of matrices")
        object.__setattr__(self, "matrices", tuple(checked))
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))

    @property
    def n_locations(self) -> int:
        """Number of locations represented by every matrix."""
        return int(self.matrices[0].shape[0])

    @property
    def max_order(self) -> int:
        """Largest represented spatial-lag index."""
        return len(self.matrices) - 1

    def __len__(self) -> int:
        return len(self.matrices)

    def __iter__(self) -> Iterator[FloatArray]:
        return iter(self.matrices)

    def __getitem__(self, item: int) -> FloatArray:
        return self.matrices[item]

    @classmethod
    def from_matrices(
        cls,
        matrices: Sequence[Any],
        *,
        names: Sequence[str] | None = None,
        prepend_identity: bool = False,
    ) -> "SpatialWeights":
        """Create a validated collection from user-supplied matrices."""
        values = [as_float_matrix(matrix, name="matrix") for matrix in matrices]
        if not values:
            raise ValueError("matrices must not be empty")
        if prepend_identity:
            values.insert(0, np.eye(values[0].shape[0], dtype=float))
        if names is None:
            resolved_names = tuple(f"W{index}" for index in range(len(values)))
        else:
            resolved_names = tuple(names)
            if prepend_identity:
                resolved_names = ("W0",) + resolved_names
        return cls(tuple(values), resolved_names)

    @classmethod
    def from_adjacency(
        cls,
        adjacency: Any,
        *,
        max_order: int = 1,
        standardize: bool = True,
    ) -> "SpatialWeights":
        """Build identity and mutually exclusive higher-order contiguity weights."""
        max_order = validate_nonnegative_int(max_order, name="max_order")
        base = as_float_matrix(adjacency, name="adjacency")
        if base.shape[0] != base.shape[1]:
            raise ValueError("adjacency must be square")
        binary = base != 0
        np.fill_diagonal(binary, False)
        n = binary.shape[0]
        matrices: list[FloatArray] = [np.eye(n, dtype=float)]
        reached = np.eye(n, dtype=bool)
        frontier = np.eye(n, dtype=bool)
        for _order in range(1, max_order + 1):
            candidates = (frontier.astype(int) @ binary.astype(int)) > 0
            candidates &= ~reached
            np.fill_diagonal(candidates, False)
            matrix = candidates.astype(float)
            matrices.append(row_standardize(matrix) if standardize else matrix)
            reached |= candidates
            frontier = candidates
        names = tuple(f"W{index}" for index in range(len(matrices)))
        return cls(tuple(matrices), names)


def coerce_weights(weights: Any, *, n_locations: int) -> SpatialWeights:
    """Coerce a common weight representation to :class:`SpatialWeights`."""
    if isinstance(weights, SpatialWeights):
        resolved = weights
    elif isinstance(weights, np.ndarray):
        resolved = SpatialWeights.from_matrices([weights], prepend_identity=True)
    elif isinstance(weights, Iterable):
        resolved = SpatialWeights.from_matrices(list(weights))
    else:
        raise TypeError(
            "weights must be SpatialWeights, a matrix, or a matrix sequence"
        )
    if resolved.n_locations != n_locations:
        raise ValueError(
            "spatial weights contain "
            f"{resolved.n_locations} locations but data contain {n_locations}"
        )
    return resolved
