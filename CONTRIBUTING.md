# Contributing

1. Create a focused branch from `main`.
2. Install development dependencies with `python -m pip install -e ".[dev,docs]"`.
3. Add tests for every numerical or public-API change.
4. Run `pytest`, `ruff`, `black --check`, `isort --check-only`, and `mypy`.
5. Document modelling assumptions, sign conventions, and numerical tolerances.

Numerical contributions should include either an analytically checkable case,
a deterministic simulation, or an independently generated reference fixture.
