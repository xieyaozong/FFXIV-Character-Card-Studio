# Architecture

```text
Screenshots
  -> validation, crop selection, background mask
  -> VLM observations.json
  -> FFXIV anatomy rules and multilingual entity lookup
  -> one-time user confirmation
  -> character_bible.json
  -> generation_plan.json
  -> reference conditioning + anatomy LoRA + style LoRA + ControlNet
  -> individual panel images
  -> VLM validation.json
  -> deterministic final_card.png
```

## Responsibility Boundaries

| Component | Owns | Does not own |
| --- | --- | --- |
| VLM observer | visible hair, horns, scales, tail, outfit, accessories, and weapon evidence | final prompts, lore, or image generation |
| FFXIV knowledge layer | canonical race rules, job and weapon compatibility, multilingual aliases | visual detection |
| Character Bible | user-confirmed identity, anatomy, outfit, and optional entities | rendering style |
| Generation planner | prompt fragments, LoRA roles, reference strength, pose controls | inventing missing character facts |
| Diffusion pipeline | one full body, portrait, expression, pose, or prop image per run | card text and final page layout |
| VLM validator | missing traits, forbidden anatomy, outfit drift, and weapon mismatch | silently editing the Character Bible |
| Layout renderer | exact text, panels, palette, icons, and final card composition | drawing the character |

Hard anatomy constraints use YAML lookup, not vector search. RAG remains optional for fuzzy equipment names and longer
descriptions. A VLM or generator is fine-tuned only after a repeatable benchmark shows that rules and adapters are not
enough.

## Evidence Priority

The knowledge pack is advisory by default. Resolution order is:

1. explicit user edits, including modded hair or non-canonical styling;
2. screenshot evidence accepted by the user;
3. FFXIV race, clan, NPC, job, and equipment defaults;
4. model guesses.

Each character can use `strict`, `advisory`, or `freeform` compatibility. `advisory` preserves visible or user-entered
exceptions while still preventing accidental human ears, missing tails, wrong scales, and similar generation errors.

## Maintainable FFXIV Data

`knowledge/ffxiv/` is the editable source of truth. YAML files hold stable IDs, multilingual aliases, anatomy rules,
prompt tokens, provenance, game version, and review dates. New races, clans, NPC appearances, and patch changes are data
updates rather than model retraining. Vector search may index this content for lookup, but it never replaces the YAML
records.
