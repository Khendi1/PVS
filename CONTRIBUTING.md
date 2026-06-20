# Contributing to Python Video Synthesizer

Thanks for your interest in contributing! This guide walks through setting up a
development environment, the project conventions, and how to get your changes
merged. Contributions of all kinds are welcome — new animations, new effects,
bug fixes, documentation, and tooling improvements.

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/Khendi1/PVS.git
cd video_synth
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows (PowerShell/cmd)
```

Python **3.11+** is required.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the web UI

The React control surface is served at `/ui/` and must be built before that
endpoint will work:

```bash
cd web && npm install && npm run build && cd ..
```

## Running the App

Use the root launcher `run.py` — it configures `sys.path` itself, so **no
`PYTHONPATH` is needed** and the same command works on Windows, macOS, and
Linux. All CLI args are forwarded to `video_synth.__main__`.

```bash
# Desktop GUI (default)
python run.py

# Headless API server (web UI only, no Qt window)
python run.py --headless --api --api-host 0.0.0.0

# Disable virtual camera (required in Docker / CI)
python run.py --headless --api --no-virtualcam
```

> **src/ layout note:** the package lives under `src/video_synth/` and uses bare
> intra-package imports. The launcher handles this for you. If you invoke the
> package directly with `python -m video_synth`, set
> `PYTHONPATH=src:src/video_synth` (bash) or `PYTHONPATH=src;src\video_synth`
> (Windows cmd) first. Prefer `python run.py`.

## Running Tests

The project uses `pytest`:

```bash
pytest tests/ -v
```

Please add or update tests when you change behavior, and make sure the full
suite passes before opening a pull request.

## Linting and Formatting

Code is linted and formatted with [Ruff](https://github.com/astral-sh/ruff).
A [pre-commit](https://pre-commit.com/) configuration is provided in
`.pre-commit-config.yaml` to run these checks automatically on every commit.

Install the hooks once after cloning:

```bash
pip install pre-commit
pre-commit install
```

You can run the hooks against the whole tree at any time:

```bash
pre-commit run --all-files
```

Or run Ruff directly:

```bash
ruff check .       # lint
ruff format .      # format
```

## Adding Animations and Effects

The project ships with helper conventions documented in `CLAUDE.md`:

- **Animations** go in `src/video_synth/animations/`, extend `Animation` from
  `animations.base`, register params via `params.new(...)`, implement
  `get_frame(self, frame)`, and are registered in `animations/enums.py` and
  `effects_manager.py`. See `animations/metaballs.py` for the canonical pattern.
- **Effects** go in `src/video_synth/effects/`, extend `EffectBase` from
  `effects.base`, follow the `do_xxx()` method pattern, and are registered in
  `effects_manager.py`. See `effects/color.py` for the canonical pattern.

## Submitting Changes

### Branching

Create a feature branch off `main`:

```bash
git checkout main
git pull
git checkout -b my-feature
```

Do not commit directly to `main`.

### Commit Messages

Write clear, descriptive commit messages. A concise summary line (imperative
mood, ~50 chars) followed by a body explaining *what* and *why* is ideal.
Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`,
`test:`, `chore:`) are encouraged for readability and changelog generation.

### Opening a Pull Request

1. Push your branch to your fork (or to the repository if you have access).
2. Open a pull request against `main`.
3. Describe the change, the motivation, and any testing you performed.
4. Ensure CI (tests, build, docs) passes — see `.github/workflows/`.
5. Keep PRs focused; smaller, single-purpose PRs are easier to review.

## What to Work On

The development roadmap is the single source of truth for project direction and
open work. See [docs/roadmap.md](docs/roadmap.md) for phased priorities and the
backlog. Items marked with a backlog/in-progress status are good starting
points. When you complete a roadmap item, update its entry in that file as part
of your PR.

## Documentation

Docs live in `docs/` and are built with MkDocs. Preview locally:

```bash
pip install -r requirements_docs.txt
mkdocs serve
```

Thanks for contributing!
