from __future__ import annotations

from pathlib import Path
from src.config import settings
from src.preprocessing.background import remove_background
from src.preprocessing.image_io import load_image_path
from src.preprocessing.palette import extract_palette
import gradio as gr


def inspect_screenshots(files: list[str] | None, background_backend: str) -> tuple[list, dict]:
    if not files:
        return [], {"status": "waiting_for_images"}

    previews = []
    summaries = []
    for value in files:
        path = Path(value)
        image = load_image_path(path)
        processed = remove_background(image, background_backend)
        previews.append((processed, path.name))
        summaries.append(
            {
                "filename": path.name,
                "width": image.width,
                "height": image.height,
                "palette": extract_palette(processed),
            }
        )

    return previews, {
        "status": "preprocessed",
        "background_backend": background_backend,
        "vlm_backend": settings.vlm_backend,
        "vlm_model_id": settings.vlm_model_id,
        "screenshots": summaries,
        "next_step": "Install the VLM stack, then review structured feature candidates.",
    }


def build_app() -> gr.Blocks:
    with gr.Blocks(title="FFXIV Character Card Studio") as demo:
        gr.Markdown("# FFXIV Character Card Studio")
        gr.Markdown("Import screenshots, review background removal, and prepare evidence for local VLM analysis.")
        with gr.Row():
            screenshots = gr.File(
                label="Character screenshots",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            backend = gr.Dropdown(
                label="Background removal",
                choices=["none", "blue_screen", "rembg"],
                value=settings.background_backend,
            )
        analyze = gr.Button("Prepare screenshots", variant="primary")
        gallery = gr.Gallery(label="Preprocessed images", columns=3, height="auto")
        report = gr.JSON(label="Analysis preparation")
        analyze.click(inspect_screenshots, inputs=[screenshots, backend], outputs=[gallery, report])
    return demo


demo = build_app()


def main() -> None:
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
