from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    default_locale: str = "zh-TW"
    background_backend: str = "blue_screen"
    rembg_model: str = "u2net_human_seg"
    vlm_backend: str = "unavailable"
    vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_max_new_tokens: int = 1200
    content_pack_path: Path = Path("content_packs/ffxiv")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
