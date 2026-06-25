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

## 2026-06-25 — Training-free fidelity: ControlNet structural conditioning

**Done:** Course-corrected the fidelity plan to the product premise (one screenshot in, **zero
training by the user**) and implemented the training-free fix. Researched SOTA low-prompt
character generation and assessed OCR/Vision-RAG. Reframed `docs/character-lora.md` into two tiers
(maintainer trains race/gear-level assets once; end users supply one screenshot) and demoted the
per-character LoRA to an optional maintainer/showcase tool. Then wired ControlNet so the redraw
follows the screenshot's *real* geometry (horns, hair, pose) instead of inventing it.
**Changed:** new `src/preprocessing/control_images.py` (`canny`/`depth`/`none` control maps); the
runner now builds a control map from the screenshot, loads a `ControlNetModel`, and uses
`StableDiffusionXLControlNetImg2ImgPipeline`, threading the control image + conditioning scale
through base + hi-res + face-detail (head crop of the control map); new args
`--controlnet-model/--controlnet-scale/--control-preprocessor/--depth-model/--control-image`;
`run.ps1` exposes the ControlNet block; `prepare_character_dataset.py` + `prepare_dataset.ps1`
kept as the maintainer-tier dataset tool (earlier brick); `.gitignore` excludes `datasets/`.
**Tech:** diffusers SDXL ControlNet img2img; OpenCV Canny (no extra model) for horn/hair edges;
optional transformers depth estimator for the already-downloaded depth-sdxl ControlNet.
**Impact:** directly attacks the "horns/hair drawn wrong, scales/sunglasses fudged" complaint
without asking users for more images — geometry is read from their one screenshot. Canny works
with no download once a matching canny/lineart/union SDXL ControlNet is fetched; depth reuses
`models/controlnet/depth-sdxl` plus a depth estimator. 24 tests pass. Next: download a
canny/lineart SDXL ControlNet and tune `$ControlNetScale`; then stronger single-image identity
(PuLID/InstantID) and DB-loaded race/gear references. OCR (PaddleOCR for the Adventurer Plate /
gear names) stays queued as a parallel recognition increment.

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

### Implementation: race recognizer (second brick)

**Done:** Implemented the recognition layer — traits → race + correction (knowledge-layer §12, item 2).
**Changed:** new `src/catalog/race_recognizer.py` (`RaceSignature`, `load_race_signatures`, `recognize_race` weighted decisive/eliminate matcher, `apply_canonical_traits` correction); `race_signatures.example.yaml` template with the real file git-ignored; tests in `test_race_recognizer.py`.
**Tech:** pydantic, deterministic scoring.
**Impact:** closes the perception → recognition loop. On the Au Ra case the recognizer picks the race from the decisive `horns=present` even though the VLM mis-typed the tail (furred) and missed scales, then locks those race-defining traits back to canonical (scaled tail, facial scales) — proving the VLM need not be perfect; the knowledge layer corrects from race context.

### Implementation: guardrail spec compilation (third brick, §9)

**Done:** Wired recognition → lore guardrails into the prompt (knowledge-layer §9).
**Changed:** new `src/prompting/spec.py` (`GenerationSpec`, `compile_generation_spec` — two channels as separate blocks, lore-required tokens ahead of content, forbidden → negative, `constraints` checklist); refactored `build_prompt` into `content_terms` + a thin `build_prompt`; wired guardrails into the runner (`--guardrails` / `--race-signatures` / `--anatomy-rules`, emits `constraints.json`); fixed content extraction to also read headwear/glasses from outfit; tests in `test_generation_spec.py`.
**Tech:** race recognizer, Illustrious + Au Ra/keyframe LoRA, IP-Adapter.
**Impact:** an end-to-end run recognized `au_ra` from the VLM traits and injected the Au Ra LoRA trigger tokens + forbidden negatives, emitting `constraints.json`. The result is clean and faithful and hugely better than the initial SDXL-base outputs — but the guardrail tokens did **not** visibly strengthen horns/tail this run (IP-Adapter + the cap in the init image suppress them), and the extra tokens pushed content past the CLIP limit (boots/headphones dropped). This confirms prompt guardrails are necessary but not sufficient, and motivates mechanism 3 (output validation + repair), which `constraints.json` now feeds.

### Implementation: face-detailer / repair pass (mechanism 3)

**Done:** Added the face-detailer repair — crop the head, regenerate at high resolution, blend it back.
**Changed:** `detail_face` in `run_baseline_experiment.py` (head crop → upscale → img2img with IP-Adapter + the guardrail prompt → gaussian-feathered paste-back); `--face-detail` / `--face-detail-strength` args; saves `result_predetail.png` for comparison.
**Tech:** SDXL img2img on the cropped head, IP-Adapter identity, feathered compositing.
**Impact:** the long-standing "nightmare face" problem is largely fixed — a head-zoom side-by-side shows a messy, asymmetric face (warped sunglasses, broken features) becoming a clean, symmetric one with the Au Ra horns/ears visible. Same machinery serves both goals: the head finally has enough pixels to render the face *and* re-asserts race anatomy in the head region. Remaining art-quality work: whole-image hi-res / upscale and possibly a refiner.

### Implementation: whole-image hi-res pass

**Done:** Added a hi-res pass — upscale the whole image ~1.5x and lightly re-render, before the face-detailer.
**Changed:** `hires_fix` in `run_baseline_experiment.py`; `--hires` / `--hires-scale` / `--hires-strength` args; pipeline order is now gen → hires → face-detail; saves `result_base.png`; `run.ps1` exposes the new hi-res / face-detail tunables.
**Tech:** SDXL img2img refine at 1152×1536, IP-Adapter identity.
**Impact:** the final card is 1152×1536 with markedly higher artistic completion — sharper linework, more finished outfit/boots, and a clean symmetric face with visible Au Ra horns / fin-ears and hinted neck scales. Combined with the face-detailer this resolves the "low completion" complaint for this character; the output now reads as a real anime illustration rather than a feature dump.
