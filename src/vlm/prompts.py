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
Analyze ONLY what is visible in the supplied character screenshots. Images are numbered image_1, image_2, and so on.
Report what you SEE in THIS image. Do not infer game lore and do not name the race.

Return ONE JSON object with exactly the keys below. Every value written as <...> is a PLACEHOLDER telling you what
to put there — replace each one with your own observation of the actual character. Do NOT output the placeholder text,
and do NOT copy these example strings; they are a form to fill in, not an answer.
{
  "traits": {
    "ear_type": "<one of: human | long_pointed | feline | rabbit_long | leonine | scaled_fin | occluded>",
    "horns": "<one of: present | absent | occluded>",
    "scales": "<one of: face | body | absent | occluded>",
    "tail_type": "<one of: scaled | feline_furred | none | occluded>",
    "stature": "<one of: child_short | average | large_tall | occluded>",
    "face_type": "<one of: human | feline_muzzle | occluded>"
  },
  "identity": [
    {"key": "<feature name, e.g. hair_color / eye_color / skin_tone / horns / ears / tail / glasses / headwear / accessory>",
     "value": "<concise description of what you actually see>", "confidence": <number 0.0-1.0>}
  ],
  "outfit": [
    {"key": "<garment slot, e.g. jacket / top / shorts / gloves / boots>",
     "value": "<colors + construction of that garment>", "confidence": <number 0.0-1.0>}
  ],
  "job": {"include": <true or false>, "status": "<detected | uncertain | not_visible | confirmed_none | user_added>"},
  "weapon": {"include": <true or false>,
    "status": "<detected | uncertain | not_visible | confirmed_none | user_added>",
    "candidates": [{"key": "weapon_type", "value": "<shape/description you see>", "confidence": <number 0.0-1.0>}]}
}

Rules:
- "identity" MUST enumerate EVERY visible feature: hair, eyes, skin, any horns / ears / tail, glasses, headwear, and
  accessories. "outfit" MUST enumerate EVERY visible garment with its colors and construction. The character is fully
  visible here, so a near-empty identity or an empty outfit is almost always WRONG — look again and list each item.
- Look carefully for easily-missed details: small horns near the hairline or beside a hat, subtle scales on the cheeks
  or neck, a thin or short tail. Report fin-shaped or scaled side-of-head ear structures as scaled_fin.
- Use "occluded" / "none" when a hat, hair, pose, or framing hides a region; use "absent" / "none" only when the region
  is clearly visible and the feature truly is not there. When you list horns, scales, or a tail in identity, the matching
  traits value must agree.
- Do not infer a job from weapon shape alone; set job include to false unless a job name or icon is visible.
- Judge traits and identity from image_1; image_2 provides weapon or job context. Use concise English. Output ONLY the
  JSON object — no Markdown, no commentary.
""".strip()
