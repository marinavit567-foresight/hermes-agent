"""Replicate video generation backend.

Uses MiniMax Video-01-Live (minimax/video-01-live) — highest-quality
video model on Replicate: text-to-video and image-to-video, native
audio, up to 6 seconds, 1080p.

Authentication: REPLICATE_API_TOKEN env var.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

REPLICATE_API_BASE = "https://api.replicate.com/v1"
DEFAULT_VIDEO_MODEL = "minimax/video-01-live"
DEFAULT_POLL_INTERVAL = 4.0
DEFAULT_TIMEOUT = 300.0

_MODELS: Dict[str, Dict[str, Any]] = {
    "minimax/video-01-live": {
        "display": "MiniMax Video-01 Live",
        "speed": "~60-120s",
        "strengths": "Best quality T2V + I2V, audio, 1080p, 6s clips",
        "price": "$0.30/video",
        "modalities": ["text", "image"],
    },
    "minimax/video-01": {
        "display": "MiniMax Video-01",
        "speed": "~60-90s",
        "strengths": "High quality T2V, cinematic motion",
        "price": "$0.20/video",
        "modalities": ["text"],
    },
}


def _get_token() -> str:
    return os.getenv("REPLICATE_API_TOKEN", "").strip()


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Request-Id": str(uuid.uuid4()),
    }


async def _generate_async(
    token: str,
    model: str,
    prompt: str,
    image_url: Optional[str],
) -> Dict[str, Any]:
    owner, name = model.split("/", 1)

    inp: Dict[str, Any] = {
        "prompt": prompt,
        "prompt_optimizer": True,
    }
    if image_url:
        inp["first_frame_image"] = image_url

    url = f"{REPLICATE_API_BASE}/models/{owner}/{name}/predictions"

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, headers=_headers(token), json={"input": inp})
        resp.raise_for_status()
        prediction = resp.json()

        prediction_id = prediction.get("id")
        if not prediction_id:
            raise RuntimeError("Replicate did not return a prediction id")

        if prediction.get("status") == "succeeded":
            return prediction

        poll_url = f"{REPLICATE_API_BASE}/predictions/{prediction_id}"
        elapsed = 0.0
        while elapsed < DEFAULT_TIMEOUT:
            await asyncio.sleep(DEFAULT_POLL_INTERVAL)
            elapsed += DEFAULT_POLL_INTERVAL
            resp = await client.get(poll_url, headers=_headers(token))
            resp.raise_for_status()
            prediction = resp.json()
            status = prediction.get("status", "")
            if status == "succeeded":
                return prediction
            if status in {"failed", "canceled"}:
                raise RuntimeError(prediction.get("error") or f"Replicate prediction {status}")

        raise TimeoutError(f"Replicate video timed out after {DEFAULT_TIMEOUT}s")


class ReplicateVideoGenProvider(VideoGenProvider):
    """Replicate video generation (MiniMax Video-01-Live)."""

    @property
    def name(self) -> str:
        return "replicate"

    @property
    def display_name(self) -> str:
        return "Replicate"

    def is_available(self) -> bool:
        return bool(_get_token())

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": mid, **meta} for mid, meta in _MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_VIDEO_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Replicate",
            "badge": "paid",
            "tag": "MiniMax Video-01-Live — T2V + I2V + audio, 1080p",
            "env_vars": [
                {
                    "key": "REPLICATE_API_TOKEN",
                    "prompt": "Replicate API token",
                    "url": "https://replicate.com/account/api-tokens",
                }
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "resolutions": ["720p", "1080p"],
            "max_duration": 6,
            "min_duration": 6,
            "supports_audio": True,
            "supports_negative_prompt": False,
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        token = _get_token()
        if not token:
            return error_response(
                error=(
                    "REPLICATE_API_TOKEN is not set. "
                    "Get a token at https://replicate.com/account/api-tokens"
                ),
                error_type="auth_required",
                provider="replicate",
                prompt=prompt,
            )

        resolved_model = (model or DEFAULT_VIDEO_MODEL).strip()
        prompt = (prompt or "").strip()

        if not prompt:
            return error_response(
                error="prompt is required for Replicate video generation",
                error_type="missing_prompt",
                provider="replicate",
                prompt=prompt,
            )

        modality = "image" if image_url else "text"

        try:
            loop = asyncio.new_event_loop()
            try:
                prediction = loop.run_until_complete(
                    _generate_async(token, resolved_model, prompt, image_url)
                )
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("Replicate video gen failed: %s", exc, exc_info=True)
            return error_response(
                error=f"Replicate video generation failed: {exc}",
                error_type="api_error",
                provider="replicate",
                model=resolved_model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        output = prediction.get("output")
        if isinstance(output, list) and output:
            video_url = output[0]
        elif isinstance(output, str):
            video_url = output
        else:
            return error_response(
                error="Replicate returned no video URL",
                error_type="empty_response",
                provider="replicate",
                model=resolved_model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        return success_response(
            video=video_url,
            model=resolved_model,
            prompt=prompt,
            modality=modality,
            aspect_ratio=aspect_ratio,
            duration=6,
            provider="replicate",
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(ReplicateVideoGenProvider())
