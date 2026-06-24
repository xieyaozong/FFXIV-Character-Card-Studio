from __future__ import annotations

FEATURE_EXTRACTION_PROMPT = """
Analyze only visible evidence in the supplied character screenshots. Images are numbered image_1, image_2, and so on.
Report what you SEE. Do not infer game lore and do not name the race.
Return only one JSON object using this exact shape:
{
  "traits": {
    "ear_type": "human",
    "horns": "absent",
    "scales": "absent",
    "tail_type": "none",
    "stature": "average",
    "face_type": "human"
  },
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
    "candidates": [
      {
        "key": "weapon_type",
        "value": "large edged weapon",
        "confidence": 0.8
      }
    ]
  }
}
The "traits" object is required. For each field choose exactly one allowed value:
- ear_type: human, long_pointed, feline, rabbit_long, leonine, occluded
- horns: present, absent, occluded
- scales: face, body, absent, occluded
- tail_type: scaled, feline_furred, none, occluded
- stature: child_short, average, large_tall, occluded
- face_type: human, feline_muzzle, occluded
Look carefully before choosing each trait: small horns near the hairline or beside a hat, subtle scales on the cheeks
or neck, and a thin or short tail are easy to miss. Inspect the head, sides of the face, neck, and lower body. Prefer
"occluded" over "absent" or "none" when a hat, hair, pose, or framing hides the region, or when you are unsure; use
"absent" or "none" only when the region is clearly visible and the feature is truly not there. Judge from image_1.

Allowed status values are detected, uncertain, not_visible, confirmed_none, and user_added. In identity and outfit list
every visible feature: hair, eyes, skin, any horns, ears, or tail, glasses, headwear, accessories, clothing colors, and
clothing construction. Do not return a minimal list. When you note horns, scales, or a tail in identity, the matching
traits field must agree. Omit an identity or outfit entry only when its value is not visible. Do not infer a job from
weapon shape alone; set job include to false unless a job name, icon, or other direct evidence is visible. Use concise
English visual descriptions. Image 1 is the clean character reference; image 2 provides weapon or job context. Do not
use Markdown.
""".strip()
