# OODKit Development

This guide is for contributors working from a cloned checkout. For normal
package use, install from PyPI with `pip install oodkit`.

## Local install

Use an editable install when developing locally:

```bash
pip install -e .
```

That installs `oodkit` from the current checkout, so changes under `src/oodkit`
are picked up without reinstalling.

Install development and notebook tools with:

```bash
pip install -e ".[dev]"
```

## Docker workflow

Docker is the recommended development environment for this repo because host
Python may not be installed or may not match the project environment.

```bash
docker compose build
docker compose run --rm dev pytest
docker compose run --rm dev python -m pytest
docker compose run --rm dev python -c "import oodkit; print(oodkit.__file__)"
docker compose run --rm dev python -m pip list
```

## Tests

Tests mirror `src/oodkit/` under `tests/pkg/`.

```bash
docker compose run --rm dev pytest
```

For a local virtual environment:

```bash
pip install -e ".[dev]"
pytest
```

## Notebooks

The `.ipynb` notebooks in `notebooks/` are the source of truth and the
human-facing examples for GitHub. Edit the `.ipynb` files directly unless a
`.py` export workflow is explicitly needed.

Keep dataset paths configurable through environment variables such as
`OODKIT_DATASETS`, and avoid committing outputs with tracebacks, local machine
paths, tokens, or one-off environment noise.

Useful entry points:

- `notebooks/imagenet_ood_showcase.ipynb`
- `notebooks/coco_ood_showcase.ipynb`
- `notebooks/README.md`

## Packaging checks

Before publishing, verify install, import, tests, and package artifacts inside
Docker:

```bash
docker compose run --rm dev python -m pip install -e .
docker compose run --rm dev python -c "import oodkit; print(oodkit.__file__)"
docker compose run --rm dev pytest
docker compose run --rm dev sh -lc "python -m pip install build twine && python -m build && python -m twine check dist/*"
```

Build artifacts are written to `dist/` and should not be committed.
