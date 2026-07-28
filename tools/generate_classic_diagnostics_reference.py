"""Generate exact rational fixtures for classical STARMA diagnostics.

This script deliberately does not import :mod:`pystarmax`. Covariances and
nested Yule-Walker systems are evaluated with :class:`fractions.Fraction`, and
linear systems are solved by a small exact Gauss-Jordan routine. The generated
JSON therefore exercises an implementation path independent of NumPy/SciPy.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

RationalMatrix: TypeAlias = list[list[Fraction]]
RationalVector: TypeAlias = list[Fraction]

DATA = [
    [1, 2, 0],
    [2, 1, 3],
    [0, 4, 2],
    [3, 2, 5],
    [4, 1, 1],
    [2, 5, 3],
    [5, 3, 4],
    [6, 2, 5],
]

WEIGHTS: list[RationalMatrix] = [
    [
        [Fraction(1), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(1)],
    ],
    [
        [Fraction(0), Fraction(1), Fraction(0)],
        [Fraction(1, 2), Fraction(0), Fraction(1, 2)],
        [Fraction(0), Fraction(1), Fraction(0)],
    ],
]


def matvec(matrix: RationalMatrix, vector: RationalVector) -> RationalVector:
    return [
        sum((value * vector[column] for column, value in enumerate(row)), Fraction())
        for row in matrix
    ]


def dot(left: RationalVector, right: RationalVector) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction())


def covariance(
    data: RationalMatrix,
    past_weight: RationalMatrix,
    future_weight: RationalMatrix,
    temporal_lag: int,
) -> Fraction:
    n_locations = len(data[0])
    n_pairs = len(data) - temporal_lag
    total = Fraction()
    for time_index in range(n_pairs):
        past = matvec(past_weight, data[time_index])
        future = matvec(future_weight, data[time_index + temporal_lag])
        total += dot(past, future)
    return total / (n_pairs * n_locations)


def transpose(matrix: RationalMatrix) -> RationalMatrix:
    return [list(column) for column in zip(*matrix, strict=True)]


def solve_exact(matrix: RationalMatrix, vector: RationalVector) -> RationalVector:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ValueError("reference Yule-Walker system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                augmented[row][entry] - factor * augmented[column][entry]
                for entry in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def main() -> None:
    n_times = len(DATA)
    n_locations = len(DATA[0])
    mean = Fraction(sum(sum(row) for row in DATA), n_times * n_locations)
    centered: RationalMatrix = [
        [Fraction(value) - mean for value in row] for row in DATA
    ]
    max_tlag = 2
    spatial_lags = len(WEIGHTS)

    covariance_blocks: list[RationalMatrix] = []
    for temporal_lag in range(max_tlag + 1):
        covariance_blocks.append(
            [
                [
                    covariance(
                        centered,
                        WEIGHTS[past_lag],
                        WEIGHTS[future_lag],
                        temporal_lag,
                    )
                    for future_lag in range(spatial_lags)
                ]
                for past_lag in range(spatial_lags)
            ]
        )

    gamma00 = covariance_blocks[0][0][0]
    stacf: list[list[float]] = []
    for temporal_lag in range(max_tlag + 1):
        row: list[float] = []
        for spatial_lag in range(spatial_lags):
            numerator = covariance_blocks[temporal_lag][spatial_lag][0]
            denominator_squared = (
                covariance_blocks[0][spatial_lag][spatial_lag] * gamma00
            )
            row.append(float(numerator) / float(denominator_squared) ** 0.5)
        stacf.append(row)

    yw_size = max_tlag * spatial_lags
    yw_matrix: RationalMatrix = [
        [Fraction() for _ in range(yw_size)] for _ in range(yw_size)
    ]
    for row_lag in range(max_tlag):
        for column_lag in range(max_tlag):
            if row_lag == column_lag:
                block = covariance_blocks[0]
            elif row_lag > column_lag:
                block = covariance_blocks[row_lag - column_lag]
            else:
                block = transpose(covariance_blocks[column_lag - row_lag])
            for row_spatial in range(spatial_lags):
                for column_spatial in range(spatial_lags):
                    yw_matrix[row_lag * spatial_lags + row_spatial][
                        column_lag * spatial_lags + column_spatial
                    ] = block[row_spatial][column_spatial]

    yw_vector: RationalVector = []
    for temporal_lag in range(1, max_tlag + 1):
        for spatial_lag in range(spatial_lags):
            yw_vector.append(
                covariance(
                    centered,
                    WEIGHTS[spatial_lag],
                    WEIGHTS[0],
                    temporal_lag,
                )
            )

    stpacf_exact: RationalVector = []
    for index in range(yw_size):
        stop = index + 1
        solution = solve_exact(
            [row[:stop] for row in yw_matrix[:stop]],
            yw_vector[:stop],
        )
        stpacf_exact.append(solution[-1])

    payload = {
        "schema_version": 1,
        "description": (
            "Exact rational reference for the Pfeifer-Deutsch covariance "
            "orientation and nested Yule-Walker STPACF"
        ),
        "provenance": {
            "generator": "tools/generate_classic_diagnostics_reference.py",
            "runtime_dependency": "Python standard library only",
            "package_imported": False,
            "arithmetic": "fractions.Fraction plus exact Gauss-Jordan elimination",
            "weight_note": "W1 is deliberately non-symmetric after row standardization",
        },
        "data": DATA,
        "weight_names": ["W0", "W1"],
        "weights": [
            [[fraction_text(value) for value in row] for row in matrix]
            for matrix in WEIGHTS
        ],
        "global_mean": fraction_text(mean),
        "max_tlag": max_tlag,
        "covariance_blocks_exact": [
            [[fraction_text(value) for value in row] for row in block]
            for block in covariance_blocks
        ],
        "stacf": stacf,
        "yule_walker_matrix_exact": [
            [fraction_text(value) for value in row] for row in yw_matrix
        ],
        "yule_walker_vector_exact": [fraction_text(value) for value in yw_vector],
        "stpacf_exact": [fraction_text(value) for value in stpacf_exact],
        "stpacf": [
            [
                float(stpacf_exact[row * spatial_lags + column])
                for column in range(spatial_lags)
            ]
            for row in range(max_tlag)
        ],
    }
    output = Path(__file__).parents[1] / "tests/reference/classic_diagnostics_v1.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
