from PIL import Image

from src.vlm.qwen_backend import prepare_vlm_images


def test_prepare_vlm_images_limits_long_edge() -> None:
    wide, small = prepare_vlm_images([Image.new("RGB", (4000, 2000)), Image.new("RGB", (400, 300))])

    assert wide.size == (1024, 512)
    assert small.size == (400, 300)
