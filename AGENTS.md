# Repository Guidelines

## Project Structure & Module Organization
This repository contains a small Python simulation and a paper draft for a graduate state-estimation project.

- `src/` holds the simulation code. `temp_diffeq.py` defines the temperature dynamics, and `random_rooms.py` runs a sample simulation and writes figures.
- `paper/` contains `main.tex` and generated SVG figures such as `ac_simulation.svg`.
- `README.md` gives the project summary.
- There is no `tests/` directory yet. Add new tests under `tests/` if you introduce reusable logic.

## Build, Test, and Development Commands
- `uv sync` installs the project dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python src/random_rooms.py` runs the current simulation and regenerates the figures in `paper/`.
- `uv run python -m compileall src` performs a quick syntax check for the Python modules.
- `pdflatex paper/main.tex` can be used to compile the paper if a LaTeX toolchain is installed locally.

Run commands from the repository root unless a tool requires otherwise.

## Coding Style & Naming Conventions
Use Python 3.12+ and follow standard PEP 8 conventions:

- 4-space indentation, no tabs.
- `snake_case` for functions, variables, and module names.
- Prefer clear parameter names such as `num_rooms`, `cool_air_temp`, and `outside_temperature`.
- Keep numerical model code separated from plotting or file-output code when extending the project.

There is no formatter or linter configured yet. Keep style consistent with the existing files and avoid large, mixed-purpose scripts.

## Testing Guidelines
There is no automated test suite yet. For now:

- Run `uv run python -m compileall src` before submitting changes.
- Run `uv run python src/random_rooms.py` and confirm the figures in `paper/` regenerate without errors.
- If you add estimation or disturbance-modeling utilities, add targeted tests in `tests/` using `pytest` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent commits use short, descriptive messages such as `Initializing UV` and `Adding the code from unpushed part of the project`. Follow that pattern:

- Use concise, imperative commit subjects.
- Keep each commit focused on one change set.
- In pull requests, include a short summary, note any modeling assumptions changed, and attach regenerated plots or paper updates when relevant.
