# Knowledge Layer — Design

> **Audience:** contributors to this project, including AI coding agents. This document is written to be
> self-contained: read it before touching the VLM extraction, the content packs, or the prompt/spec compiler.
> It explains *why* the design is shaped this way, not only *what* to build.

## 1. What this project produces

A **personalized FFXIV character profile card** — a multi-panel reference sheet (hero portrait, full-body
turnaround, expression sheet, chibi poses, color palette, personality notes, optional pet). Reference examples (made
with a cloud image model, kept private) define the *output*, not the *implementation*.

The card sections map almost field-for-field onto the existing [`CharacterProfile`](../src/domain/models.py) schema,
which is the central artifact of the whole system.

## 2. Guiding principle: lore guards, the user drives

**This is the key of the whole architecture.** The FFXIV worldview is a *guardrail*, not a creative driver: its only
job is to keep a generated character from breaking lore (wrong/lost tails, phantom human ears on a Miqo'te, missing
Au Ra scales). The *features worth generating* come from the **user's needs** — their outfit, hairstyle (even a
modded one), pose, personality, the things they want emphasized. The architecture's value is the reconciliation:

> **Maximize fidelity to the user's request while keeping the output inside FFXIV's race / anatomy bounds.**

Lore never contributes "flavor"; it only prevents errors. Personalization is the product.

### The north star (zero-prompt for the end user)

An end user drops in a screenshot and gets a correct card **without typing prompts to explain anything**. The system
recognizes *what is in the screenshot* on its own. When FFXIV ships a new race or changes an NPC, the **maintainer
updates the knowledge base** and recognition keeps working — no model retraining, no end-user explanation.

| User | Input | Must never have to |
| --- | --- | --- |
| **End user** | a screenshot (+ optional personality flavor) | type prompts to explain race/anatomy/lore |
| **Maintainer** (project owner) | curates the knowledge base | retrain a model to add a new race |

### How the clamp works (three mechanisms)

1. **Two independent channels into the generator.** A *content channel* (user-driven: outfit, hair incl. mods, pose,
   personality, emphasized features) drives what is drawn; a *guardrail channel* (lore-driven: the recognized race's
   required traits as forced positives, forbidden traits as negatives + checks) clamps what must hold. They do not
   contaminate each other — user creativity is not diluted by lore, and lore is not mistaken for style.
2. **Race anatomy is a hard invariant.** Styling (hair incl. mods, outfit, color, accessories, pose) is fully
   user-driven and never clamped; the recognized race's defining anatomy is **non-negotiable** — the architecture
   never produces a lore-violating composition (a human-eared Miqo'te, a short Hrothgar), whatever the input. This
   does not fight the user: the game itself cannot make those, so a real character is already lore-valid, and the
   invariant only blocks *generator errors*. There is no "user overrides the guardrail" path for anatomy.
3. **Output validation + repair loop.** Prompt constraints do not *guarantee* obedience (a model can still draw the
   wrong ears despite a negative). So the generated image is checked back against the guardrails with the VLM
   (required traits present? forbidden traits absent?) and **repaired** when violated (redraw the face, inpaint,
   strengthen negatives, re-roll). This loop — not the prompt alone — is what actually keeps the output in-lore.

The anatomy invariant is enforced at three points: the **UI** never exposes a raw high-privilege prompt — end users
give constrained styling / personality / preset choices, so there is no way to *request* a violation (this also fits
the zero-prompt goal); the **compiler** always emits the forbidden-trait negatives and required positives, with no
override branch; the **validation loop** repairs any violation the model produces anyway.

## 3. Core principle: separate *perception* from *recognition*

This is the single most important decision in the design.

| Layer | Job | Needs FFXIV knowledge? |
| --- | --- | --- |
| **VLM = perception** | Report raw, discriminating anatomical traits it can see (ear type, horns, scales, tail type, stature, weapon shape). | **No.** It only has to see accurately. |
| **Knowledge DB = recognition** | Match observed traits against each race's visual signature → canonical race / clan (+ optional job / weapon). | **Yes**, and it is maintainer-editable data, not model weights. |

Why split them:

- **New content = data update, not fine-tuning.** A new race in a new patch becomes one new signature record, and
  the system recognizes it immediately.
- **Deterministic and explainable.** Race identity is decided by rule-matching against data, so a wrong result is a
  data fix, not a black-box retrain.
- **The VLM stays swappable.** Any capable VLM that reports traits accurately works; it never has to "know FFXIV."

> Consequence for prompt work: the next improvement to VLM extraction is **not** prettier generation prompts. It is
> redesigning extraction to elicit the *discriminating traits the recognizer needs*. Past failure: the VLM described
> Au Ra horns as `"black cap with white horn-like protrusions"` — perception that recognition cannot use.

## 4. The FFXIV race problem (why this layer is necessary)

Base image and vision models do not understand FFXIV's playable races, so they hallucinate generic-anime anatomy:
Miqo'te drawn with human ears, Au Ra scales dropped, tails redrawn as the wrong kind, Lalafell sized like normal
adults, Elezen rendered as blonde western-fantasy elves. The error-prone races are exactly the non-human ones.

The eight playable races and the **traits that discriminate them**:

| Race (EN / 日本語 / 繁中) | Decisive signature |
| --- | --- |
| Hyur / ヒューラン / 人族 | Baseline human; no special markers (the fallback). |
| Elezen / エレゼン / 精靈族 | Tall + long pointed ears; **no** horns/scales/feline traits; not western-elf coloring. |
| Lalafell / ララフェル / 拉拉菲爾族 | Very short, child-like stature (decisive). |
| Miqo'te / ミコッテ / 貓魅族 | Feline ears on top of head + furred tail + slit pupils. |
| Roegadyn / ルガディン / 魯加族 | Very large, muscular build; human ears. |
| Au Ra / アウラ / 敖龍族 | Horns + facial/body scales + scaled tail + limbal-ring eyes. |
| Hrothgar / ロスガル / 硌獅族 | Leonine/feline face (muzzle) + large build. |
| Viera / ヴィエラ / 維埃拉族 | Very long rabbit-like ears + tall. |

Clans (e.g. Au Ra → Raen / Xaela, Miqo'te → Seekers of the Sun / Keepers of the Moon) refine skin/scale coloring and
some traits; they are a second recognition step after race.

## 5. Data model (what the maintainer edits)

The maintainer should almost never hand-write structured YAML. The natural inputs are an **image** or a **sentence**
(a new race's look, an NPC, a weapon, a piece of lore); the system AI-drafts the structured record and the maintainer
**reviews and confirms** it — the same perception → confirm pattern the end user gets, turned inward for authoring.

### 5.1 Three data types, three representations

| Data | Natural maintainer input | Stored / used as | Source of truth |
| --- | --- | --- | --- |
| **Hard rules** (race anatomy, entity IDs/aliases, weapon→job) | a reference image + a sentence | AI-drafted record → maintainer confirms → YAML | the structured record (the recognizer needs determinism) |
| **Lore / worldview / story** | prose paragraphs | prose docs → RAG index | the prose itself (not derived from YAML) |
| **Visual appearance** (race / NPC / gear) | an image | reference-image store → (a) draft a signature, (b) direct IP-Adapter reference, (c) visual-similarity match | the image |

So "YAML is the single source of truth" holds only for **hard rules**. Lore lives as prose (RAG indexes it; the prose
is primary, not a derived view of YAML). Images are source material for the other two.

### 5.2 Structured records (the files)

- `manifest.yaml` — data version, locales, active files, default compatibility. *(tracked / public)*
- `*.example.yaml` — format templates to copy and fill. *(tracked / public)*
- `anatomy_rules.yaml` — required / conditional / forbidden traits and generation tokens. *(git-ignored / private)*
- `entities.yaml` — canonical IDs + multilingual aliases (loaded by
  [`EntityCatalog`](../src/catalog/entity_catalog.py)). *(git-ignored / private)*
- `locales/` — interface labels only. *(tracked / public)*

**Proposed addition — race *signatures* for recognition.** Today `anatomy_rules.yaml` encodes what a race must look
like for *generation*; recognition additionally needs the *discriminating perception traits*. Proposed shape (exact
file organization is a maintainer decision — new `race_signatures.yaml`, or a `signature:` block inside each anatomy
profile):

```yaml
races:
  au_ra:
    names: { ja-JP: アウラ, zh-TW: 敖龍族, en-US: Au Ra }
    clans: [raen, xaela]
    signature:            # discriminating traits the recognizer matches against VLM perception
      horns: required     # decisive: horns ⇒ Au Ra
      scales: required    # decisive: face/neck/body scales
      tail: scaled
      ears: human_small
      stature: average
    eliminates: [feline_ears, rabbit_ears, leonine_face]   # traits that rule this race out
    # generation/anatomy rules continue to live in anatomy_rules.yaml, keyed by race[_clan_gender]
```

Maintenance rules already documented in [`content_packs/ffxiv/README.md`](../content_packs/ffxiv/README.md): stable
lowercase IDs, verified multilingual names, `game_version` / `source` / `reviewed_at` per record, **NPC appearances
versioned by new ID instead of overwriting**. Run `tests/test_content_pack.py` before bumping `data_version`.

### 5.3 Maintenance workflow (data authoring)

A **data-authoring mode** in the app, separate from the end-user flow:

1. The maintainer uploads a reference image and/or pastes a description.
2. The VLM drafts a structured record (race signature, anatomy rules, entity names) — reusing the project's own
   Qwen3-VL; a text step parses prose into fields.
3. The maintainer reviews/edits the draft in a form and **confirms**.
4. The confirmed record is written to the content pack with provenance (source, date, `game_version`).
5. Lore prose is pasted into `lore/` and indexed for retrieval; no structuring required.

The maintainer's job is *natural input + one confirmation*, not authoring YAML by hand.

### 5.4 Public framework vs. private data (decided & implemented)

The curated knowledge is the maintainer's IP, kept private like model weights. **Published = the framework only:**
code, schema, format docs, and minimal `*.example.yaml` templates. The filled-in data (`anatomy_rules.yaml`,
`entities.yaml`, `lore/`, reference images, any RAG index) is git-ignored. Others fork the framework and build their
own database. Implemented 2026-06-23: templates shipped, real data untracked and scrubbed from history.

## 6. Recognition flow

```text
screenshot
  │  (preprocessing: triage + subject crop + background removal — already implemented)
  ▼
VLM perception  →  structured traits with confidence
  │              ear_type, horns, scales, tail_type, stature, face_type, weapon_shape, + universal visuals
  ▼
race recognizer →  score each race signature vs observed traits; apply eliminations
  │              decisive trait seen ⇒ assign; defining trait occluded/ambiguous ⇒ low confidence
  ▼
clan refinement (optional) + job/weapon recognition (optional, from weapon shape / job icon)
  ▼
anatomy fill   →  inject required / forbidden traits + generation tokens for the recognized race
  ▼
+ user personality input (personality / likes / dislikes / quote / mood — not derivable from a screenshot)
  ▼
CharacterProfile spec  →  master prompt / panel specs
  ▼
render (swappable — see §10)
```

**Recognizer logic** is deterministic rule-matching, not vector search:

- Some traits are **decisive** (horns+scales ⇒ Au Ra; feline ears+tail ⇒ Miqo'te; very long rabbit ears ⇒ Viera;
  leonine face ⇒ Hrothgar; very short stature ⇒ Lalafell).
- Some traits are **eliminating** (clearly visible round human ears ⇒ not Miqo'te/Viera/Hrothgar).
- Score = weighted matches − eliminations, weighting decisive traits highest. Best score over a threshold wins;
  otherwise the result is "uncertain" and the UX asks for one confirmation (§7).

This maps onto the existing schema: VLM output is [`VLMFeatureResponse`](../src/domain/models.py); recognized facts
land in [`AnatomyProfile`](../src/domain/models.py) (`race_id`, `clan_id`, required/forbidden traits) and the
optional [`OptionalEntity`](../src/domain/models.py) job/weapon slots.

## 7. Confidence & confirmation UX

The zero-prompt ideal degrades gracefully — **never into prompt writing.**

| Recognition confidence | Behavior |
| --- | --- |
| High (decisive trait clearly seen) | Auto-assign race; no interruption. |
| Medium | Assign, mark `detected`; user can correct with one tap. |
| Low / ambiguous (defining trait occluded — horns under a hat, back-facing shot) | Ask the user to pick the race from a short list, pre-ranked by partial evidence. One selection, no text. |

Backed by existing schema fields: [`EvidenceStatus`](../src/domain/models.py) (`DETECTED` / `UNCERTAIN` /
`CONFIRMED_NONE` / `USER_ADDED`), per-feature `confidence`, and one-time confirmation. These exist but are **not yet
wired** to a recognizer.

**Evidence priority is per-layer.** The order `user_override > confirmed_screenshot > canonical_default >
model_guess` (from [`anatomy_rules.yaml`](../content_packs/ffxiv/anatomy_rules.yaml) and
[`docs/architecture.md`](architecture.md)) governs **race selection and styling / ambiguous facts only**. It never
lets the user violate an established race's defining anatomy — that is a hard invariant (§2).

**Mods / overrides.** Hair, outfit, and color are the *universal styling layer* and are never gated by race rules, so
a MOD hairstyle always survives. `CompatibilityMode` no longer decides *whether* anatomy is enforced (it always is);
at most it tunes conditional / minor traits or stylization intensity — see §13.

## 8. Mapping: card section → schema field → source

| Card section | `CharacterProfile` field | Source |
| --- | --- | --- |
| Race / clan | `anatomy.race_id`, `anatomy.clan_id` | **Recognition (DB)** |
| Job | `job` (OptionalEntity) | Recognition (job icon / weapon) or user |
| Weapon | `weapon` (OptionalEntity) | Recognition (weapon shape) or user |
| Personality / likes / dislikes / quote / mood | `personality`, `likes`, `dislikes`, `quote` | **User input** (not in screenshots) |
| Hair / outfit / colors | `identity_features`, `outfits`, `palette` | VLM perception + user |
| Expressions / poses / view / product | `PanelRequest` + `configs/presets/` | User selects |
| Color palette | `palette` | `extract_palette` (implemented) |

## 9. Spec compilation (mechanism 1 in detail)

The compiler turns the recognized facts plus the user's `CharacterProfile` into a generation spec, keeping the two
channels (§2) separate.

**Inputs:** the user's `CharacterProfile` (content) + the recognized race's anatomy rules
(`required` / `forbidden` / `generation_tokens`) + the product / style preset.

**Output — a `GenerationSpec`** (extends today's [`PromptPlan`](../src/domain/models.py)): `positive_prompt`,
`negative_prompt`, `constraints: {required, forbidden}` (the checklist the validation loop reuses — §11),
and renderer params (IP-Adapter reference, panel / style).

**Two channels, assembled as separate blocks (never woven together):**

```text
positive =  [style / quality]   preset
          + [lore · required]   ← guardrail   race required + generation_tokens.positive ("horns, facial scales, scaled tail")
          + [user core]         ← content     confirmed visual features, outfit, hair (incl. mods), accessories
          + [flavor]                          lower-priority extras

negative =  [lore · forbidden]  ← guardrail   race forbidden ("human ears, cat ears, elf ears")
          + [standard negatives]              text, watermark, extra limbs …
```

The lore blocks are *appended as distinct segments*, not merged into the user's description, so the channels stay
independently traceable. There is **no override branch** for the forbidden block — race anatomy is a hard invariant (§2).

**CLIP-budget priority changes.** `fit_prompt_to_clip` keeps leading fragments and drops the tail, so the keep-order
must be **style → lore-required (the floor, must survive) → user core → [budget line] → flavor (dropped first)**.
Lore-required tokens are short and high-value (dropping one forces a repair), so they sit *ahead* of user content;
only flavor is sacrificed under pressure. In practice prompts are usually under budget, so this rarely bites — and the
validation loop backstops it regardless.

**In code:** upgrade [`src/prompting/compiler.py`](../src/prompting/compiler.py) `compile_prompt` to take the
recognized anatomy rules as a second input and emit the `GenerationSpec`. The current `build_prompt` in
[`scripts/run_baseline_experiment.py`](../scripts/run_baseline_experiment.py) is the content-channel-only precursor; it
folds into the content block.

## 10. The renderer is swappable; the spec is the deliverable

The durable output of this system is the **profile spec** (a populated `CharacterProfile` plus a compiled master
prompt). It can drive either:

- **Path A — one-shot cloud composer** (e.g. a strong image model that lays out the whole multi-panel sheet with
  text in a single pass). Already proven by the maintainer's private reference cards. Fastest route to the target quality.
- **Path B — fully local** panel generation (needs a character LoRA + IP-Adapter for cross-panel consistency) plus a
  deterministic layout engine that renders text/palette/notes in code (Pillow/HTML), per
  [`docs/architecture.md`](architecture.md). More control, no per-image cost, much larger build.

The knowledge DB + spec is where this project's unique, maintainer-owned value lives. The renderer is a commodity
that keeps improving; do not couple the spec to one renderer.

## 11. Validation & repair loop (mechanism 3 in detail)

This is the actual guarantee that the output stays in-lore — perception applied to the *generated image*, checked
against the spec's constraint checklist.

**The check (reuses everything):** run the same VLM used for input perception on the result, extract its traits, and
compare against `GenerationSpec.constraints` (`{required, forbidden}`, emitted by §9 — no separate config).

**Outcomes and repair** (escalating, region-targeted, bounded):

| Case | Action |
| --- | --- |
| Pass (required present, forbidden absent) | done |
| Missing required (e.g. scales dropped) | inpaint the region and re-emphasize that positive |
| Forbidden present (e.g. human ears on a Miqo'te) | redraw / inpaint the **face region**, strengthen the negative, re-roll that area |
| Still failing after N tries | best-of-N, or surface to the user |

Most violations are facial (ears, horns, face scales), so the main repair is a **face-region inpaint** — the same
face-detailer machinery (SAM crop → inpaint) used for quality. Only the offending region is touched; the user's other
content is preserved.

**Strictness (ties to the `CompatibilityMode` open decision, §13):** defining race anatomy (the human-eared Miqo'te
case) is *always* repaired — it is the hard invariant (§2). Conditional / minor traits may be tolerated per product; a
character card does not over-iterate on minutiae.

**Per render path:** Path B (local panels) validates and repairs each panel naturally. Path A (one-shot cloud) must
validate on cropped panels / faces after the fact, and repair is more limited (often a re-prompt rather than a precise
inpaint) — a known weakness of Path A.

**Maps to scaffolding:** architecture.md's **VLM validator** (missing traits / forbidden anatomy / outfit drift) is
this loop; a `ValidationResult` model can carry the comparison.

## 12. Current state vs to-build

**Implemented today:**

- Preprocessing: triage, subject crop, background removal ([`src/preprocessing/`](../src/preprocessing/)).
- VLM perception (freeform feature JSON) — [`src/vlm/`](../src/vlm/).
- Visual-only prompt building (`build_prompt` in
  [`scripts/run_baseline_experiment.py`](../scripts/run_baseline_experiment.py)); **does not use anatomy rules yet.**
- Schemas, enums, compatibility modes ([`src/domain/models.py`](../src/domain/models.py)).
- Entity catalog + content-pack scaffolding; `anatomy_rules.yaml` has one race.
- Experimental SDXL / IP-Adapter generation (WIP, not part of the milestone).

**To build (in rough order):**

1. **Discriminating-trait VLM extraction** — rewrite [`FEATURE_EXTRACTION_PROMPT`](../src/vlm/prompts.py) to elicit
   the §6 traits. *(This is the planned next concrete step.)*
2. **Race signatures** for all eight races + the recognizer that matches traits → race/clan.
3. **Wire anatomy rules into the spec** (required/forbidden traits + generation tokens) — the missing link in both
   `build_prompt` and [`src/prompting/compiler.py`](../src/prompting/compiler.py).
4. **Personality input layer** (UI form populating `personality` / `likes` / `dislikes` / `quote`).
5. **Confidence + one-tap confirmation UX**.
6. **Spec → renderer** adapter(s) for Path A and/or Path B.
7. **Data-authoring mode** (image/prose → VLM draft → confirm → write record) so the knowledge base is maintainable
   without hand-writing YAML (§5.3).

## 13. Open decisions

- File organization of race signatures (new file vs. a block inside anatomy profiles).
- Scope of job/weapon recognition for the first version (weapon-shape signatures, or defer to user selection).
- RAG timing and shape: its role is settled (it indexes lore prose — lore's own source of truth — and resolves fuzzy
  gear/NPC names to structured IDs), but *when* to build it and how to chunk/enrich is open. Hard race/anatomy never
  uses it.
- How much the data-authoring mode automates in v1 (full AI draft vs. assisted fields) before maintainer confirmation.
- Whether `CompatibilityMode` (`strict` / `advisory` / `freeform`) survives now that race anatomy is a hard invariant
  and styling is always free — it may only govern conditional / minor traits or stylization intensity, or be dropped.
- Render path priority: A, B, or both in parallel.
