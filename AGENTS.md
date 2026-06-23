# AGENTS.md

Onboarding for contributors and AI coding agents working on **FFXIV-Character-Card-Studio**.

## Read first

- [docs/knowledge-layer.md](docs/knowledge-layer.md) — design and direction (perception vs. recognition, the knowledge DB, the spec). **Start here.**
- [docs/architecture.md](docs/architecture.md) — responsibility boundaries and the full pipeline vision.
- [README.md](README.md) — current milestone and quick start.

## Where the project is

- **Done (current milestone):** screenshots → VLM features → character prompt.
- **WIP / not the focus:** SDXL / IP-Adapter image generation, RAG, card layout.
- **Next concrete step:** discriminating-trait VLM extraction (see knowledge-layer.md §10).

## Core principle — do not break

Separate **perception** from **recognition**:

- The VLM reports raw visual traits; it must never be expected to "know FFXIV."
- The maintainer-edited knowledge DB (`content_packs/ffxiv/`) turns traits into canonical FFXIV entities.
- New FFXIV content (a new race, an NPC change) = a **data update**, never a model retrain.

Lore is the floor (prevent broken anatomy such as a Miqo'te with human ears); **personalization is the product**.

## Conventions

- Python 3.12. Lint with ruff (`pyproject.toml`): `ruff check .` must pass; imports are isort-ordered.
- `pytest -q` must pass. Add tests for new logic and keep them dependency-light — no GPU or model downloads in unit tests.
- Evidence-first: never invent character facts the screenshot or user did not provide.
- Content packs are the source of truth, not model weights. Follow [content_packs/ffxiv/README.md](content_packs/ffxiv/README.md) when editing data; version NPC appearances by new ID instead of overwriting.
- Docs and comments in English. Add Japanese + Traditional Chinese **only** for FFXIV in-game nouns (e.g. Au Ra / アウラ / 敖龍族).
- Record meaningful work in [docs/progress-log.md](docs/progress-log.md) using the Done / Changed / Tech / Impact format.

## Never commit

Private screenshots (`character/`, `private_inputs/`), model weights and LoRAs (`models/`), generated output (`outputs/`), and local env (`.env`). See [.gitignore](.gitignore).

## Run

- Triage screenshots: `./triage.ps1` (or `scripts/triage_screenshots.py`).
- Screenshots → prompt: `./run.ps1` with `$FeaturesOnly = $true` (or `scripts/run_baseline_experiment.py --features-only`).

The GPU stack (torch + transformers + diffusers) installs separately — see [docs/gpu_and_models.md](docs/gpu_and_models.md).
