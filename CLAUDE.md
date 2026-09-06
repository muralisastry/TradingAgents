# CLAUDE.md

This file provides guidance to Claude Code when working with code in this
project.

## What this service does

Vendored fork of upstream `TauricResearch/TradingAgents` (a multi-agent LLM
trading framework), kept in sync with upstream and **minimally diverged**.
Workspace-specific integration and UI live in the separate
`TradingAgents-Core` project (`../TradingAgents-Core/CLAUDE.md` if present,
else `docs/services/TradingAgents-Core/architecture.md`).

### Deliberate divergences from upstream

Preserve these across upstream merges — every one is load-bearing, and the
first three are in files upstream also edits, so they conflict or (worse)
auto-merge:

| What | Where |
| --- | --- |
| Anthropic defaults (Sonnet 5 deep / Haiku 4.5 quick) instead of upstream's OpenAI pair | `default_config.py`, `tests/test_env_overrides.py` |
| `MODEL_OPTIONS` rebuilt from the shared quantai-core catalog when readable | `llm_clients/model_catalog.py` |
| Per-tier Anthropic effort + prompt caching | `graph/trading_graph.py` |
| Prompt caching **on** by default (upstream: off) | `default_config.py`, pinned by `tests/test_per_tier_effort.py` |
| Trader on the deep tier (upstream: quick) | `graph/setup.py`, pinned by `tests/test_graph_tiers.py` |
| `json_schema` structured output for reasoning models | `agents/utils/structured.py` |
| Default token / LLM-call budgets | `default_config.py` |

After any upstream merge, grep that each survives, then run **both** suites —
this one and `qt test tradingagents-core` (the fork is installed editable, so
Core imports it live).

## Running

Conda env. `tradingagents` CLI (entry point `cli.main:app`) or
`python main.py`.

## Tests

conda-base `pytest` (569 tests, +2 skipped). This is upstream's own suite,
not workspace-specific — it doesn't gate workspace CI.

## Verification

After a change, before calling it done:

1. Run this project's own suite: conda-base `pytest`.
2. If you touched vendor integration code (anything `TradingAgents-Core`
   calls into), also run `TradingAgents-Core`'s tests.

## Gotchas

- Git-ignored by the monorepo — it's an in-place, separate git repo
  (`origin` = your fork, `upstream` = `TauricResearch/TradingAgents`), not
  tracked by the monorepo's git history.
- The one genuinely upstream-owned tree in the workspace: stays in sync with
  upstream. Don't hand-edit business logic here without checking the
  upstream diff first — you'll fight future merges.

## Boundaries

- Don't retheme or restructure. If the suite needs different behavior
  (workspace UI, orchestration, data wiring), that belongs in
  `TradingAgents-Core`, not here.

## See also

- `docs/services/TradingAgents-Core/architecture.md`
- Root `CLAUDE.md` §1 (sub-project map) and §5 (venv/python convention)
