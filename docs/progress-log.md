# Progress Log

A short, dated record of what changed each working session so the history is easy to recover later.
Newest entries first. Each entry follows the same shape:

```text
## YYYY-MM-DD — <title>

**Done:**    what was accomplished
**Changed:** files / modules touched
**Tech:**    technologies and techniques used
**Impact:**  the effect, and why it matters
```

---

## 2026-06-23 — Preprocessing, prompt fixes, and first repo cleanup

### Screenshot triage + subject crop

**Done:** Added a triage step that scores each screenshot and crops the character before analysis.
**Changed:** new `src/preprocessing/triage.py`; `alpha_bbox` / `pad_bbox` in `src/preprocessing/background.py`; new `scripts/triage_screenshots.py`; default background backend `blue_screen` → `rembg`; crop wired into the pipeline.
**Tech:** rembg (U²-Net) subject mask → bounding box; brightness / coverage / Laplacian-variance gates for usability scoring.
**Impact:** 4K in-game screenshots shrank the character to ~80 px before the VLM saw it; cropping to the subject first let the VLM read small markers (tail, headphones, horns) that earlier runs missed. Two dark interior shots are now auto-rejected.

### Full-body framing fix

**Done:** Fixed generated outputs that were cropped down to the torso.
**Changed:** `ImageOps.fit` → `ImageOps.pad` for the SDXL init image in `scripts/run_baseline_experiment.py`.
**Tech:** letterbox-pad to the target aspect instead of center-crop.
**Impact:** a tightly cropped subject is tall and narrow; `fit` was cutting off head and feet. Padding keeps the whole body.

### Prompt: CLIP 77-token truncation

**Done:** Stopped CLIP from silently dropping the end of the prompt.
**Changed:** rewrote `build_prompt` (dedupe, trim verbose values, prefer outfit construction over a redundant color summary); added `fit_prompt_to_clip` as a tokenizer-aware safety net.
**Tech:** CLIP tokenizer token counting; greedy front-to-back fragment keeping.
**Impact:** prompt dropped 83 → 69 tokens with all key features kept (`yellow headphones`, horns, tail no longer truncated).

### Style-LoRA generation (experimental)

**Done:** First styled generation with the Illustrious base and a style LoRA.
**Changed:** ran `run_baseline_experiment.py` with Illustrious XL + keyframe (style) + Au Ra (anatomy) LoRAs.
**Tech:** Diffusers SDXL img2img, multiple PEFT LoRA adapters.
**Impact:** cleaner anime linework and full-body framing; confirmed low img2img strength behaves like tracing, so identity vs. art is a real trade-off to solve with conditioning.

### IP-Adapter identity injection (WIP)

**Done:** Added IP-Adapter support so a high-strength redraw can keep the character's face.
**Changed:** `--ip-adapter-*` options in `run_baseline_experiment.py`; `run.ps1` exposes `$IpAdapterScale`.
**Tech:** IP-Adapter plus-face for SDXL; required the ViT-H image encoder (1280-dim), not the bigG one shipped under `sdxl_models` — that mismatch was the first failure (loaded the ViT-H encoder explicitly and registered it on the pipeline).
**Impact:** validated end-to-end — IP-Adapter 0.6 with strength 0.75 produced a full redraw (clean anime line art, not tracing) that kept the cap, Au Ra horns, sunglasses, headphones, outfit, and tail. Confirms the identity-vs-art trade-off is solvable: high-strength redraw + reference identity injection.

### Repo cleanup for first GitHub release

**Done:** Tidied the project to the screenshots → prompt milestone for publishing.
**Changed:** ruff import sorting repo-wide (added `I` rule); fixed lint (B008, E501); `.gitignore` now excludes `character/` and `.claude/settings.local.json`; removed an empty training stub; trimmed verbose comments; marked the generation half WIP; rewrote the README (English technical core, FFXIV sections in EN/JP/zh-TW).
**Tech:** ruff, markdown, git.
**Impact:** the public repo presents one finished stage (VLM → prompt) without leaking private screenshots, weights, or outputs, and without committing to a larger restructure yet.

### Knowledge-layer design + collaborator onboarding

**Done:** Wrote down the product vision and the recognition architecture, and added an onboarding entry point for contributors and AI agents.
**Changed:** new `docs/knowledge-layer.md` (perception vs. recognition, race signatures, recognition flow, confidence/confirmation UX, CharacterProfile mapping, current-state vs. to-build); new `AGENTS.md`; README refined (English prose, FFXIV nouns kept trilingual, no About section, roadmap reframed around DB-driven recognition); `.gitignore` now excludes `example/`.
**Tech:** —
**Impact:** the design (zero-prompt goal, perception/recognition split, swappable renderer) is now legible enough for other contributors to pick up; private reference images stay out of git.

### Private knowledge data + maintenance model

**Done:** Made the curated FFXIV knowledge data private (the maintainer's IP, like model weights) and folded the data-maintenance model into the design doc.
**Changed:** `content_packs/ffxiv/` real data (`anatomy_rules.yaml`, `entities.yaml`) untracked + git-ignored and shipped as `*.example.yaml` templates; `test_content_pack.py` validates against templates; both files scrubbed from all git history (git-filter-repo + force-push); `docs/knowledge-layer.md` §5 now covers natural-input → AI-draft → confirm authoring, three data types (hard rules / lore+RAG / images), and the public-framework / private-data split.
**Tech:** git-filter-repo, force-push, ruff, pytest.
**Impact:** the public repo is framework-only; the maintainer keeps their data private and grows it via image/prose + one confirmation instead of hand-writing YAML.

### Knowledge-layer design detailed (guardrail principle + 3 clamp mechanisms)

**Done:** Fleshed out the reconciliation design in `docs/knowledge-layer.md`.
**Changed:** added the guardrail-vs-driver guiding principle (§2); race anatomy as a hard invariant with three enforcement points (UI / compiler / validator); per-layer evidence priority; §9 spec compilation (two channels assembled as separate blocks, lore-floor CLIP-budget priority, `constraints` output); §11 validation & repair loop (VLM checks the output against the constraint checklist, region-targeted face inpaint repair).
**Tech:** —
**Impact:** all three clamp mechanisms (channels / invariant / validation) now have written detail; the design is ready to implement from the §12 to-build list, starting with discriminating-trait VLM extraction.

### Implementation: discriminating-trait VLM extraction (first brick)

**Done:** Implemented the perception layer's discriminating-trait extraction (knowledge-layer §12, item 1).
**Changed:** added `RaceTraits` (ear_type / horns / scales / tail_type / stature / face_type, default `occluded`) + a `traits` field on `VLMFeatureResponse` (`models.py`); rewrote `FEATURE_EXTRACTION_PROMPT` to emit the traits block with allowed values plus careful-inspection / occluded-bias guidance (`prompts.py`); added parse tests (`test_models.py`).
**Tech:** Qwen3-VL, pydantic.
**Impact:** validated on a real Au Ra screenshot — the decisive trait (`horns=present`) is detected and outfit/weapon stay rich. The VLM mis-typed the tail (`feline_furred` vs `scaled`) and missed subtle facial scales; these are deliberately left for the recognizer to correct from race context, confirming the perception/recognition split. A first prompt version that under-detected horns showed why the forced-choice format needs explicit "look carefully / prefer occluded" guidance.
