"""Replicate image generation backend.

Uses FLUX 1.1 Pro Ultra (black-forest-labs/flux-1.1-pro-ultra) — the
highest-quality image model on Replicate: 4-megapixel output, photorealism,
prompt upsampling option.

Authentication: ``REPLICATE_API_TOKEN`` env var.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

REPLICATE_API_BASE = "https://api.replicate.com/v1"
DEFAULT_IMAGE_MODEL = "black-forest-labs/flux-1.1-pro-ultra"
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_TIMEOUT = 120.0

# Hermes abstract → Replicate native aspect ratios
_ASPECT_RATIO_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

_MODELS: Dict[str, Dict[str, Any]] = {
    "black-forest-labs/flux-1.1-pro-ultra": {
        "display": "FLUX 1.1 Pro Ultra",
        "speed": "~10-20s",
        "strengths": "4MP photorealism, best quality on Replicate",
        "price": "$0.06/image",
    },
    "black-forest-labs/flux-1.1-pro": {
        "display": "FLUX 1.1 Pro",
        "speed": "~6-10s",
        "strengths": "Fast, high-quality, 1MP",
        "price": "$0.04/image",
    },
}


def _get_token() -> str:
    return os.getenv("REPLICATE_API_TOKEN", "").strip()


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
        "X-Request-Id": str(uuid.uuid4()),
    }


async def _generate_async(
    token: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
) -> Dict[str, Any]:
    owner, name = model.split("/", 1)
    native_ratio = _ASPECT_RATIO_MAP.get(aspect_ratio, "16:9")

    payload: Dict[str, Any] = {
        "input": {
            "prompt": prompt,
            "aspect_ratio": native_ratio,
            "output_format": "webp",
            "output_quality": 90,
            "safety_tolerance": 2,
            "prompt_upsampling": True,
        }
    }

    url = f"{REPLICATE_API_BASE}/models/{owner}/{name}/predictions"

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.post(url, headers=_headers(token), json=payload)
        resp.raise_for_status()
        prediction = resp.json()

        prediction_id = prediction.get("id")
        if not prediction_id:
            raise RuntimeError("Replicate did not return a prediction id")

        # If already done (Prefer: wait may resolve inline)
        if prediction.get("status") == "succeeded":
            return prediction

        # Poll until done
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
                error = prediction.get("error") or f"Replicate prediction {status}"
                raise RuntimeError(error)

        raise TimeoutError(f"Replicate prediction timed out after {DEFAULT_TIMEOUT}s")


class ReplicateImageGenProvider(ImageGenProvider):
    """Replicate image generation (FLUX 1.1 Pro Ultra)."""

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
        return DEFAULT_IMAGE_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Replicate",
            "badge": "paid",
            "tag": "FLUX 1.1 Pro Ultra — 4MP photorealism",
            "env_vars": [
                {
                    "key": "REPLICATE_API_TOKEN",
                    "prompt": "Replicate API token",
                    "url": "https://replicate.com/account/api-tokens",
                }
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
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

        model = (kwargs.get("model") or DEFAULT_IMAGE_MODEL).strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        prompt = (prompt or "").strip()

        if not prompt:
            return error_response(
                error="prompt is required",
                error_type="missing_prompt",
                provider="replicate",
                prompt=prompt,
            )

        try:
            loop = asyncio.new_event_loop()
            try:
                prediction = loop.run_until_complete(
                    _generate_async(token, model, prompt, aspect)
                )
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("Replicate image gen failed: %s", exc, exc_info=True)
            return error_response(
                error=f"Replicate image generation failed: {exc}",
                error_type="api_error",
                provider="replicate",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        output = prediction.get("output")
        if isinstance(output, list) and output:
            image_url = output[0]
        elif isinstance(output, str):
            image_url = output
        else:
            return error_response(
                error="Replicate returned no output URL",
                error_type="empty_response",
                provider="replicate",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=image_url,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="replicate",
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ReplicateImageGenProvider())
