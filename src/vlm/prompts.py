from __future__ import annotations

FEATURE_EXTRACTION_PROMPT = """
Analyze only visible evidence in the supplied character screenshots. Images are numbered image_1, image_2, and so on.
Separate stable body features from clothing. Do not infer hidden features or game lore. A job or weapon may be
uncertain, but must be detected when visible. Return only one JSON object using this exact shape:
{
  "identity": [
    {
      "key": "hair_color",
      "value": "black",
      "confidence": 0.95
    }
  ],
  "outfit": [],
  "job": {
    "include": false,
    "status": "not_visible"
  },
  "weapon": {
    "include": true,
    "status": "detected",
    "canonical_id": null,
    "candidates": [
      {
        "key": "weapon_type",
        "value": "large edged weapon",
        "confidence": 0.8
      }
    ]
  }
}
Allowed status values are detected, uncertain, not_visible, confirmed_none, and user_added. Include visible hair, eyes,
skin, horns, ears, tail, glasses, headwear, clothing colors, clothing construction, accessories, and weapon appearance.
Omit identity or outfit entries when their value is not visible. Do not infer a job from weapon shape alone. Set job
include to false unless a job name, icon, or other direct evidence is visible. Use concise English visual descriptions.
Omit evidence, confirmed, canonical_id, empty candidates, and uncertain to keep the response short. Image 1 is the clean
character reference; image 2 provides weapon or job context. Do not use Markdown.
""".strip()
