"""Instrumented single-run cost measurement.

Runs one full analysis with a per-LLM-call usage logger and prints:
per-call token table (with cache read/write split), per-model totals,
dollar cost, and a portfolio extrapolation. Use it to baseline run cost
before/after config or caching changes.

Usage (conda-base python, keys in env or .env):

    python scripts/measure_run_cost.py --ticker NVDA --date 2026-07-06
    python scripts/measure_run_cost.py --quick-model claude-haiku-4-5 --effort low
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from cli.stats_handler import StatsCallbackHandler

# $/1M tokens: (input, output, cache_read, cache_write_5m). Prefix-matched.
PRICES = {
    "claude-opus-4-8": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-7": (5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5": (3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
    "claude-fable-5": (10.00, 50.00, 1.00, 12.50),
}

RUNS_PER_MONTH_40_WEEKLY = 40 * 52 / 12  # ≈173.3


def price_for(model: str):
    for prefix, p in PRICES.items():
        if model.startswith(prefix):
            return p
    return None


class PerCallStatsHandler(StatsCallbackHandler):
    """StatsCallbackHandler plus a per-call usage log with model + cache split."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self._run_meta: dict[Any, dict[str, Any]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, **kwargs):
        super().on_chat_model_start(serialized, messages, **kwargs)
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or ""
        node = (metadata or {}).get("langgraph_node", "")
        with self._lock:
            self._run_meta[run_id] = {"model": model, "node": node, "t0": time.time()}

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        super().on_llm_end(response, **kwargs)
        try:
            message = response.generations[0][0].message
            usage = message.usage_metadata or {}
        except (IndexError, TypeError, AttributeError):
            return
        details = usage.get("input_token_details") or {}
        meta = self._run_meta.pop(run_id, {})
        model = meta.get("model") or (response.llm_output or {}).get("model_name", "")
        # Cache writes surface either as "cache_creation" or in the TTL
        # buckets (ephemeral_5m/1h) depending on the provider payload shape.
        cache_creation = details.get("cache_creation", 0) or (
            details.get("ephemeral_5m_input_tokens", 0) + details.get("ephemeral_1h_input_tokens", 0)
        )
        with self._lock:
            self.records.append(
                {
                    "seq": len(self.records) + 1,
                    "node": meta.get("node", ""),
                    "model": model,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_read": details.get("cache_read", 0),
                    "cache_creation": cache_creation,
                    "seconds": round(time.time() - meta["t0"], 1) if "t0" in meta else None,
                }
            )


def record_cost(r: dict[str, Any]) -> float | None:
    p = price_for(r["model"])
    if p is None:
        return None
    in_price, out_price, read_price, write_price = p
    uncached = r["input_tokens"] - r["cache_read"] - r["cache_creation"]
    return (
        uncached * in_price
        + r["cache_read"] * read_price
        + r["cache_creation"] * write_price
        + r["output_tokens"] * out_price
    ) / 1_000_000


def report(handler: PerCallStatsHandler, label: str) -> dict[str, Any]:
    print(f"\n=== {label} — per-call usage ===")
    header = f"{'#':>3} {'node':<22} {'model':<20} {'in':>8} {'cache_rd':>9} {'cache_wr':>9} {'out':>7} {'$':>8}"
    print(header)
    print("-" * len(header))
    total_cost = 0.0
    by_model: dict[str, dict[str, float]] = {}
    for r in handler.records:
        cost = record_cost(r) or 0.0
        total_cost += cost
        m = by_model.setdefault(r["model"], {"in": 0, "out": 0, "cache_rd": 0, "cache_wr": 0, "cost": 0.0, "calls": 0})
        m["in"] += r["input_tokens"]
        m["out"] += r["output_tokens"]
        m["cache_rd"] += r["cache_read"]
        m["cache_wr"] += r["cache_creation"]
        m["cost"] += cost
        m["calls"] += 1
        print(
            f"{r['seq']:>3} {r['node'][:22]:<22} {r['model'][:20]:<20} "
            f"{r['input_tokens']:>8,} {r['cache_read']:>9,} {r['cache_creation']:>9,} "
            f"{r['output_tokens']:>7,} {cost:>8.4f}"
        )
    print("\n--- per-model totals ---")
    for model, m in by_model.items():
        print(
            f"{model}: {m['calls']:.0f} calls, in={m['in']:,.0f} "
            f"(cache_rd={m['cache_rd']:,.0f}, cache_wr={m['cache_wr']:,.0f}), "
            f"out={m['out']:,.0f}, cost=${m['cost']:.4f}"
        )
    stats = handler.get_stats()
    print(f"\ncumulative check: {stats}")
    monthly = total_cost * RUNS_PER_MONTH_40_WEEKLY
    print(f"\nRUN COST: ${total_cost:.4f}")
    print(f"40 stocks weekly  ≈ {RUNS_PER_MONTH_40_WEEKLY:.0f} runs/month → ${monthly:,.2f}/month (${monthly * 12:,.0f}/year)")
    return {"label": label, "records": handler.records, "stats": stats, "run_cost_usd": round(total_cost, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--date", default=None, help="trade date YYYY-MM-DD (default: today)")
    parser.add_argument("--quick-model", default="claude-sonnet-5")
    parser.add_argument("--deep-model", default="claude-opus-4-8")
    parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--quick-effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--deep-effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--caching", action="store_true", help="enable Anthropic prompt caching")
    parser.add_argument("--analysts", nargs="+", default=["market", "social", "news", "fundamentals"])
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--json-out", default=None, help="write the per-call log to this JSON file")
    args = parser.parse_args()

    load_dotenv()
    from datetime import date as _date

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    trade_date = args.date or _date.today().isoformat()
    config = DEFAULT_CONFIG.copy()
    config.update(
        llm_provider="anthropic",
        deep_think_llm=args.deep_model,
        quick_think_llm=args.quick_model,
        anthropic_effort=args.effort,
        anthropic_effort_quick=args.quick_effort,
        anthropic_effort_deep=args.deep_effort,
        anthropic_prompt_caching=args.caching,
        max_debate_rounds=args.depth,
        max_risk_discuss_rounds=args.depth,
        checkpoint_enabled=False,
    )

    effort_desc = f"effort={args.effort or 'default'}/q={args.quick_effort or '-'}/d={args.deep_effort or '-'}"
    label = f"{args.ticker} {trade_date} quick={args.quick_model} deep={args.deep_model} {effort_desc} cache={args.caching} depth={args.depth}"
    print(f"Running: {label}")

    handler = PerCallStatsHandler()
    graph = TradingAgentsGraph(args.analysts, config=config, debug=False, callbacks=[handler])

    # Direct-stream (webapp pattern) so tool callbacks are wired too.
    init_state = graph.propagator.create_initial_state(args.ticker, trade_date)
    stream_args = graph.propagator.get_graph_args(callbacks=[handler])
    final_state = None
    for chunk in graph.graph.stream(init_state, **stream_args):
        final_state = chunk

    decision = (final_state or {}).get("final_trade_decision", "")
    print(f"\nDECISION (first 300 chars): {str(decision)[:300]}")

    result = report(handler, label)
    if args.json_out:
        result["decision"] = str(decision)
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
