# Screenshot Preparation

The first UI can work with one or two screenshots, but consistent character LoRA training needs a broader private dataset.

Recommended character set:

- 20 to 50 screenshots
- front, three-quarter, side, close-up, half-body, and full-body views
- several expressions and poses
- minimal UI overlays
- more than one background and lighting condition
- outfit groups kept separate
- weapon screenshots only when weapon generation is wanted

Keep source images under `private_inputs/`. That directory is ignored by git.

The two blue-background model screenshots are useful for the first background-removal and VLM tests, but they are too similar in pose and framing to train a reliable character LoRA by themselves.
