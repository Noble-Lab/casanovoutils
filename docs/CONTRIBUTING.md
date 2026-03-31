# Contributing

Thank you for your interest in contributing to casanovoutils! We welcome bug
reports, feature requests, documentation improvements, and code contributions.

## Reporting issues

Please open an issue on
[GitHub](https://github.com/Noble-Lab/casanovoutils/issues) and include:

- A clear description of the problem or request.
- Steps to reproduce the issue (for bugs).
- The versions of casanovoutils and Python you are using.

## Setting up a development environment

Clone the repository and install the package with development dependencies
using uv:

```bash
git clone https://github.com/Noble-Lab/casanovoutils.git
cd casanovoutils
uv sync --group dev
```

## Running the tests

```bash
uv run pytest tests/
```

## Code style

casanovoutils uses [black](https://black.readthedocs.io/) for formatting and
[isort](https://pycqa.github.io/isort/) for import ordering. Before
submitting a pull request, please run:

```bash
uv run isort src/ tests/
uv run black src/ tests/
```

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and add tests where appropriate.
3. Ensure all tests pass and the code is formatted.
4. Open a pull request against `main` with a clear description of the changes.

## Building the documentation

Install the documentation dependencies and build locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

Or with uv:

```bash
uv sync --extra docs
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.
