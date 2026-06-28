"""Replicate provider profile."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class ReplicateProviderProfile(ProviderProfile):
    """Replicate — paginated /models endpoint, owner/name model IDs."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        url = "https://api.replicate.com/v1/models"
        results: list[str] = []
        while url:
            req = urllib.request.Request(url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Accept", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as exc:
                logger.debug("replicate fetch_models: %s", exc)
                return results or None
            for m in data.get("results", []):
                owner = m.get("owner", "")
                name = m.get("name", "")
                if owner and name:
                    results.append(f"{owner}/{name}")
            url = data.get("next") or ""
            # stop after first page to avoid very long fetch
            break
        return results or None


replicate = ReplicateProviderProfile(
    name="replicate",
    aliases=("repl",),
    display_name="Replicate",
    description="Replicate — run open-source models in the cloud",
    signup_url="https://replicate.com/account/api-tokens",
    env_vars=("REPLICATE_API_TOKEN",),
    base_url="https://api.replicate.com/v1",
    fallback_models=(
        "meta/meta-llama-3.1-405b-instruct",
        "meta/meta-llama-3.3-70b-instruct",
        "meta/meta-llama-3-70b-instruct",
    ),
    supports_vision=True,
)

register_provider(replicate)
