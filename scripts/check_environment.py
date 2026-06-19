from __future__ import annotations

from importlib.util import find_spec
from platform import python_version


def main() -> None:
    print(f"python={python_version()}")
    for package in ["fastapi", "gradio", "cv2", "rembg", "onnxruntime", "torch", "transformers", "diffusers"]:
        print(f"{package}={'installed' if find_spec(package) else 'missing'}")


if __name__ == "__main__":
    main()
