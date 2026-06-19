from __future__ import annotations

from json import JSONDecodeError, loads
from PIL import Image
from src.domain.models import VLMFeatureResponse
from src.vlm.prompts import FEATURE_EXTRACTION_PROMPT


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

    def analyze(self, images: list[Image.Image]) -> VLMFeatureResponse:
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
        return VLMFeatureResponse.model_validate(extract_json(response))
