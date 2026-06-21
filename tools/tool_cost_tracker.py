"""
Tool cost tracking for Hermes.

Tracks per-call costs for paid tools (web search providers, image generation,
etc.), displays them inline after each call, and prints a turn/session/all-time
summary at the end of each turn.

Usage from tool implementations:
    from tools.tool_cost_tracker import COST_TRACKER
    COST_TRACKER.emit_web_search(provider)       # after provider.search()
    COST_TRACKER.emit_web_extract(provider, n)   # after provider.extract()

Usage from cli.py:
    COST_TRACKER.begin_turn()                    # before run_conversation()
    COST_TRACKER.print_turn_summary(llm_delta, llm_total, model)  # after
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

# ---------------------------------------------------------------------------
# Static pricing table (USD per call)
# None = token-based or unknown — provider must report dynamically.
# Sources: public pricing pages as of 2025.
# ---------------------------------------------------------------------------

_WEB_SEARCH_PRICING: dict[str, Optional[Decimal]] = {
    "tavily":     Decimal("0.005"),
    "exa":        Decimal("0.010"),
    "firecrawl":  Decimal("0.003"),
    "parallel":   Decimal("0.005"),
    "brave-free": Decimal("0"),
    "ddgs":       Decimal("0"),
    "searxng":    Decimal("0"),
    "xai":        None,
    "perplexity": None,   # token-based; provider reports dynamically
}

_WEB_EXTRACT_PRICING: dict[str, Optional[Decimal]] = {
    "tavily":     Decimal("0.015"),
    "exa":        Decimal("0.010"),
    "firecrawl":  Decimal("0.003"),
    "parallel":   None,
    "brave-free": Decimal("0"),
    "ddgs":       Decimal("0"),
    "searxng":    Decimal("0"),
    "perplexity": None,
}


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostEvent:
    ts: float
    tool: str
    provider: str
    model: Optional[str]
    cost_usd: float


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ToolCostTracker:
    """Session-scoped cost tracker.  One singleton per process: ``COST_TRACKER``."""

    def __init__(self) -> None:
        self._session_events: list[CostEvent] = []
        self._turn_events: list[CostEvent] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def begin_turn(self) -> None:
        """Call before each run_conversation(). Resets per-turn accumulator."""
        self._turn_events = []

    def emit(
        self,
        tool: str,
        provider: str,
        cost_usd: Decimal,
        model: Optional[str] = None,
    ) -> None:
        """Record a paid tool call and print the inline cost badge."""
        event = CostEvent(
            ts=time.time(),
            tool=tool,
            provider=provider,
            model=model,
            cost_usd=float(cost_usd),
        )
        self._session_events.append(event)
        self._turn_events.append(event)
        self._persist(event)
        self._print_event(tool, provider, model, cost_usd)

    # -- Web-specific helpers ------------------------------------------------

    def emit_web_search(self, provider_obj: object) -> None:
        """Emit cost for a web_search call using the active provider."""
        cost = _resolve_provider_cost(provider_obj, "search")
        if cost is None:
            return
        self.emit("web_search", getattr(provider_obj, "name", "?"), cost)

    def emit_web_extract(self, provider_obj: object, url_count: int = 1) -> None:
        """Emit cost for a web_extract call.  Scales by number of URLs."""
        base = _resolve_provider_cost(provider_obj, "extract")
        if base is None:
            return
        self.emit(
            "web_extract",
            getattr(provider_obj, "name", "?"),
            base * max(1, url_count),
        )

    # -- Turn summary --------------------------------------------------------

    def print_turn_summary(
        self,
        llm_turn_cost_usd: float = 0.0,
        llm_total_cost_usd: float = 0.0,
        model: str = "",
        cost_status: str = "unknown",
    ) -> None:
        """Print the cost footer line at the end of a turn.

        Skipped when both LLM and tool costs are zero (e.g. interrupted turns,
        turns with no API calls, or sessions where pricing is unavailable).
        """
        turn_tool = float(self.turn_tool_cost_usd)
        turn_total = llm_turn_cost_usd + turn_tool
        session_tool = float(self.session_tool_cost_usd)
        session_total = llm_total_cost_usd + session_tool
        alltime_tool = float(self.alltime_tool_cost_usd)

        # Suppress the line entirely when there's nothing meaningful to show.
        llm_unknown = cost_status in ("unknown", "none", "")
        if llm_unknown and turn_tool == 0:
            return

        try:
            from rich.console import Console
            from rich.text import Text

            c = Console(highlight=False)
            parts: list[str] = []

            # Turn cost
            if turn_total > 0 or turn_tool > 0:
                if llm_unknown:
                    turn_str = f"[bold]${turn_tool:.4f}[/bold] tools"
                elif llm_turn_cost_usd > 0 and turn_tool > 0:
                    turn_str = (
                        f"[bold]${turn_total:.4f}[/bold] "
                        f"[dim](LLM ${llm_turn_cost_usd:.4f} + tools ${turn_tool:.4f})[/dim]"
                    )
                elif llm_turn_cost_usd > 0:
                    prefix = "~" if cost_status == "estimated" else ""
                    turn_str = f"[bold]{prefix}${llm_turn_cost_usd:.4f}[/bold] LLM"
                else:
                    turn_str = f"[bold]${turn_tool:.4f}[/bold] tools"
                parts.append(f"Turn {turn_str}")

            # Session total
            if not llm_unknown:
                parts.append(f"Session [bold]${session_total:.4f}[/bold]")
            elif session_tool > 0:
                parts.append(f"Session tools [bold]${session_tool:.4f}[/bold]")

            # All-time tools (from JSONL, omit if zero and no history)
            if alltime_tool > 0:
                parts.append(f"All-time tools [bold]${alltime_tool:.4f}[/bold]")

            if not parts:
                return

            c.print(f"  [dim cyan]{'  ·  '.join(parts)}[/dim cyan]")
        except Exception:
            pass

    # -- /usage extension ----------------------------------------------------

    def format_usage_block(self) -> str:
        """Return a formatted string for inclusion in /usage output."""
        lines: list[str] = []
        session_tool = float(self.session_tool_cost_usd)
        alltime_tool = float(self.alltime_tool_cost_usd)

        if not self._session_events and alltime_tool == 0:
            return ""

        lines.append(f"  {'─' * 40}")
        lines.append("  Tool API Costs (this session)")
        if self._session_events:
            by_provider: dict[str, float] = {}
            for e in self._session_events:
                key = f"{e.tool} · {e.provider}"
                by_provider[key] = by_provider.get(key, 0.0) + e.cost_usd
            for label, cost in sorted(by_provider.items()):
                lines.append(f"    {label:<36}  ${cost:.4f}")
            lines.append(f"  Session tool total:          ${session_tool:>10.4f}")
        else:
            lines.append("    (no paid tool calls this session)")

        if alltime_tool > 0:
            lines.append(f"  All-time tool total:         ${alltime_tool:>10.4f}")

        return "\n".join(lines)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def session_tool_cost_usd(self) -> Decimal:
        return Decimal(str(sum(e.cost_usd for e in self._session_events)))

    @property
    def turn_tool_cost_usd(self) -> Decimal:
        return Decimal(str(sum(e.cost_usd for e in self._turn_events)))

    @property
    def alltime_tool_cost_usd(self) -> Decimal:
        """Total tool costs from ALL sessions (reads JSONL log)."""
        try:
            from hermes_constants import get_hermes_home
            log_path = get_hermes_home() / "tool_costs.jsonl"
            if not log_path.exists():
                return Decimal("0")
            total = Decimal("0")
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        total += Decimal(str(json.loads(line.strip()).get("cost_usd", 0)))
                    except Exception:
                        pass
            return total
        except Exception:
            return Decimal("0")

    # ── Private ─────────────────────────────────────────────────────────────

    def _print_event(
        self,
        tool: str,
        provider: str,
        model: Optional[str],
        cost: Decimal,
    ) -> None:
        try:
            from rich.console import Console
            c = Console(highlight=False)
            model_str = f"/{model}" if model else ""
            if cost == 0:
                cost_str = "[dim]free[/dim]"
            else:
                cost_str = f"[yellow]${float(cost):.4f}[/yellow]"
            c.print(f"  [dim]└─ {tool} · {provider}{model_str} · {cost_str}[/dim]")
        except Exception:
            pass

    def _persist(self, event: CostEvent) -> None:
        try:
            from hermes_constants import get_hermes_home
            log_path = get_hermes_home() / "tool_costs.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "ts": event.ts,
                        "tool": event.tool,
                        "provider": event.provider,
                        "model": event.model,
                        "cost_usd": event.cost_usd,
                    })
                    + "\n"
                )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_provider_cost(
    provider_obj: object,
    kind: str,  # "search" | "extract"
) -> Optional[Decimal]:
    """Return USD cost per call from provider method, then static table, else None."""
    # Try provider-reported cost first (dynamic pricing, token-based, etc.)
    method_name = f"cost_per_{kind}_usd"
    method = getattr(provider_obj, method_name, None)
    if callable(method):
        try:
            result = method()
            if result is not None:
                return Decimal(str(result))
        except Exception:
            pass

    # Fall back to static table
    name = getattr(provider_obj, "name", "")
    table = _WEB_SEARCH_PRICING if kind == "search" else _WEB_EXTRACT_PRICING
    return table.get(name)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

COST_TRACKER = ToolCostTracker()
