# FFXIV Knowledge Data

This folder contains public templates plus private local data.

Tracked:

- `manifest.yaml`
- `*.example.yaml`
- `locales/`
- this README

Ignored:

- `anatomy_rules.yaml`
- `entities.yaml`
- `race_signatures.yaml`
- `special_features.yaml`
- `*.npz`
- `reference/`
- `lore/`

Use the examples as formats. Keep filled records local.

## Update Check

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_content_pack.py -q -p no:cacheprovider
```
