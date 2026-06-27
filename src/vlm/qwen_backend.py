from __future__ import annotations

from json import JSONDecodeError, loads

from PIL import Image

from src.domain.models import EvidenceRef, RaceTraits, VLMFeatureResponse
from src.vlm.prompts import FEATURE_EXTRACTION_PROMPT, TRAITS_EXTRACTION_PROMPT


def prepare_vlm_images(images: list[Image.Image], max_side: int = 1024) -> list[Image.Image]:
    prepared: list[Image.Image] = []
    for image in images:
        resized = image.convert("RGB")
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        prepared.append(resized)
    return prepared


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM response does not contain a JSON object.")
    try:
        return loads(text[start : end + 1])
    except JSONDecodeError as exc:
        raise ValueError("VLM response contains invalid JSON.") from exc


class QwenVLMBackend:
    def __init__(self, model_id: str, max_new_tokens: int = 1200) -> None:
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt and a GPU build of Torch first.") from exc

        if "Qwen2.5-VL" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
        else:
            from transformers import Qwen3VLForConditionalGeneration as QwenVLModel

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = QwenVLModel.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def _generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def extract_traits(self, image: Image.Image) -> RaceTraits:
        """Second pass on a zoomed head crop, where small/pale horns and faint scales are legible."""
        prepared = prepare_vlm_images([image])
        response = self._generate(prepared, TRAITS_EXTRACTION_PROMPT, max_new_tokens=200)
        return RaceTraits.model_validate(extract_json(response))

    def analyze(self, images: list[Image.Image]) -> VLMFeatureResponse:
        images = prepare_vlm_images(images)
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": FEATURE_EXTRACTION_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated, strict=True)]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        result = VLMFeatureResponse.model_validate(extract_json(response))
        result.identity = [
            item for item in result.identity if "not visible" not in item.value.lower().replace("_", " ")
        ]
        result.outfit = [
            item for item in result.outfit if "not visible" not in item.value.lower().replace("_", " ")
        ]
        for item in result.identity + result.outfit:
            if not item.evidence:
                item.evidence = [EvidenceRef(source_image="image_1")]
        context_source = "image_2" if len(images) > 1 else "image_1"
        for item in result.job.candidates + result.weapon.candidates:
            if not item.evidence:
                item.evidence = [EvidenceRef(source_image=context_source)]
        has_visible_evidence = bool(result.identity or result.outfit or result.job.include or result.weapon.include)
        if not has_visible_evidence:
            raise ValueError("VLM returned no visible character evidence.")
        return result
