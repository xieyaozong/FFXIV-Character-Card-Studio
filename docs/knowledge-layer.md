# Knowledge Layer

The model reads the screenshot. The knowledge files keep FFXIV-specific facts consistent.

## Principle

Screenshot and user input drive the character. FFXIV data only prevents broken anatomy or wrong entity names.

Examples:

- Au Ra should keep horns, scales, and a scaled tail.
- Miqo'te should not gain human ears.
- Viera ears should not become generic elf ears.
- Modded hair, outfit colors, poses, and personality remain user-owned choices.

## Private Data

The public repo ships templates only. Real maintained data stays local under `knowledge/ffxiv/`:

```text
anatomy_rules.yaml
entities.yaml
race_signatures.yaml
special_features.yaml
reference/
lore/
race_index.npz
```

These files are ignored by git because they are personal maintained data, like model weights.

## Files

- `entities.yaml`: stable IDs, multilingual names, aliases, gear/NPC records.
- `anatomy_rules.yaml`: required traits, forbidden traits, generation tokens, local LoRA references.
- `race_signatures.yaml`: traits used to recognize a race from model observations.
- `special_features.yaml`: optional reference folders for parts models often draw badly.

## Flow

```text
screenshot
  -> visual traits
  -> race / gear matching
  -> required and forbidden traits
  -> generation spec
```

The vision model does not need to know FFXIV lore. It only reports what it sees. The maintained files decide what those
traits mean.

## Update Rules

- Use stable lowercase IDs.
- Keep Japanese, Chinese, and English names only after checking them.
- Record source, game version, and review date when a record can change.
- Add a new NPC appearance ID when a patch changes the look.
- Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_content_pack.py -q -p no:cacheprovider
```
