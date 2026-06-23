# FFXIV Content Pack

This directory is the editable source of truth for FFXIV-specific knowledge. Model weights are not the database.

## Files

- `manifest.yaml`: data version, supported locales, and active data files.
- `entities.yaml`: verified stable IDs and multilingual aliases for races, clans, jobs, weapons, equipment, and NPCs.
- `anatomy_rules.yaml`: required, conditional, and forbidden visual traits plus generation tokens.
- `locales/`: interface labels only.

## Update Rules

1. Use a stable lowercase ID that does not change when a translation changes.
2. Record Japanese, Simplified Chinese, Traditional Chinese, and English names only after verification.
3. Add `game_version`, `source`, `reviewed_at`, and short notes to records that can change between patches.
4. Keep NPC appearances versioned instead of overwriting old appearances.
5. Run the YAML and unit tests before changing `data_version` in `manifest.yaml`.

## Entity Template

Add records under `entities:` in `entities.yaml`:

```yaml
- canonical_id: race.example
  entity_type: race
  names:
    ja-JP: ""
    zh-CN: ""
    zh-TW: ""
    en-US: ""
  aliases: {}
  visual_prompt: ""
  negative_prompt: []
  game_version: ""
  source: ""
  reviewed_at: YYYY-MM-DD
  notes: ""
```

Race anatomy belongs in `anatomy_rules.yaml`; names and aliases belong in `entities.yaml`. For an NPC appearance
change, add a new versioned ID instead of rewriting the old record.

Run after every update:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_content_pack.py -q -p no:cacheprovider
```

Character-level evidence has priority over this pack. Explicit user edits come first, then confirmed screenshots, then
canonical defaults, and finally model guesses. `advisory` mode is the default so modded hair and deliberate
non-canonical choices survive while accidental anatomy errors can still be flagged.
