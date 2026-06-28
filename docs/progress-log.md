# Progress Log

## 2026-06-28 - Repository cleanup

Cleaned the project into a framework-only repository:

- moved FFXIV data templates to `knowledge/ffxiv`
- moved matching logic to `src/knowledge`
- removed the unused API/UI/Docker shell
- trimmed dependencies
- shortened public docs
- kept model weights, generated outputs, screenshots, and curated FFXIV data local-only

## 2026-06-28 - Current shape

Implemented:

- screenshot triage and crop
- background removal
- local vision-model feature extraction (+ head-zoom traits pass)
- FFXIV race and gear matching hooks (VLM traits + image-embedding ensemble; force-race override)
- prompt/spec compilation with race guardrails
- local generation path: ControlNet structure, IP-Adapter identity, hi-res + anime ESRGAN upscale, face detail
- expression-sheet generation
- multi-panel card layout renderer (Path B)

Still local/private:

- model weights
- LoRAs
- race reference images
- curated FFXIV YAML data
- generated examples
