# Contributor Notes

Read first:

- [README.md](README.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/knowledge-layer.md](docs/knowledge-layer.md)

## Rules

- Keep the public repo framework-only.
- Do not commit screenshots, model weights, LoRAs, generated images, datasets, or filled FFXIV knowledge files.
- Treat `knowledge/ffxiv/*.example.yaml` as public templates.
- Treat `knowledge/ffxiv/*.yaml`, `reference/`, `lore/`, and `*.npz` as private maintained data.
- The vision model reports visible traits; `src/knowledge/` maps those traits to FFXIV entities.
- Do not invent jobs, weapons, pets, props, or race traits without screenshot evidence or user confirmation.

## Checks

```powershell
ruff check .
pytest -q
```
