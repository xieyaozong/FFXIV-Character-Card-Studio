from __future__ import annotations


FEATURE_EXTRACTION_PROMPT = """
Analyze only visible evidence in the supplied character screenshots.
Return JSON matching the requested schema. Separate stable identity features from outfit features.
Do not infer a job, weapon, pet, mount, prop, or item when it is not visible.
Use status not_visible when the screenshot provides no evidence and uncertain when evidence is ambiguous.
For every detected candidate include confidence, source_image, and an optional bounding box.
Describe appearance using literal visual attributes rather than game lore.
""".strip()
