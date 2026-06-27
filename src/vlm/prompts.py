from __future__ import annotations

TRAITS_EXTRACTION_PROMPT = """
This is a CLOSE-UP crop of one character's head and shoulders. Report ONLY the visible anatomical
traits, as a single JSON object with exactly these fields:
{ "ear_type": "human", "horns": "absent", "scales": "absent", "face_type": "human",
  "tail_type": "occluded", "stature": "occluded" }
Allowed values:
- ear_type: human, long_pointed, feline, rabbit_long, leonine, scaled_fin, occluded
- horns: present, absent, occluded
- scales: face, body, absent, occluded
- face_type: human, feline_muzzle, occluded
This is a zoomed-in view, so small or subtle details are now clearly visible. Look very carefully:
horns may be small, pale, or rise near the hairline; ears may be pointed, feline, rabbit-like, or
scaled fin-shaped (report fin/scaled side-of-head structures as scaled_fin); scales can be faint
patches on the cheeks, brow, jaw, or neck. Do NOT report "absent"
for anything you can see even faintly — only use "absent" when the region is clearly bare. If hair
or a hat truly hides a region, use "occluded". The tail and overall body height are NOT in this crop:
always set tail_type and stature to "occluded". Return only the JSON object. Do not use Markdown.
""".strip()

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
- ear_type: human, long_pointed, feline, rabbit_long, leonine, scaled_fin, occluded
- horns: present, absent, occluded
- scales: face, body, absent, occluded
- tail_type: scaled, feline_furred, none, occluded
- stature: child_short, average, large_tall, occluded
- face_type: human, feline_muzzle, occluded
Report fin-shaped or scaled side-of-head ear structures as scaled_fin.
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
