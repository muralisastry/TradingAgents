"""Per-tier Anthropic effort resolution and opt-in prompt caching."""

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.anthropic_client import AnthropicClient


def _kwargs(config_overrides, tier):
    graph = object.__new__(TradingAgentsGraph)  # skip __init__ (no LLM construction)
    graph.config = {**DEFAULT_CONFIG, "llm_provider": "anthropic", **config_overrides}
    return graph._get_provider_kwargs(tier=tier)


def test_per_tier_effort_overrides_shared_value():
    overrides = {"anthropic_effort": "high", "anthropic_effort_quick": "low"}
    assert _kwargs(overrides, "quick")["effort"] == "low"
    assert _kwargs(overrides, "deep")["effort"] == "high"


def test_per_tier_effort_falls_back_to_shared():
    overrides = {"anthropic_effort": "medium"}
    assert _kwargs(overrides, "quick")["effort"] == "medium"
    assert _kwargs(overrides, "deep")["effort"] == "medium"


def test_no_effort_when_nothing_set():
    assert "effort" not in _kwargs({}, "quick")
    assert "effort" not in _kwargs({}, "deep")


def test_prompt_caching_flag_threads_through():
    assert _kwargs({"anthropic_prompt_caching": True}, "quick")["prompt_caching"] is True
    # Explicitly off must reach the client as *absent*, not as False — the
    # client tests truthiness, so a stray False would still be a no-op, but
    # absence is what the off path is supposed to produce.
    assert "prompt_caching" not in _kwargs({"anthropic_prompt_caching": False}, "quick")


def test_prompt_caching_is_on_without_an_override():
    """The fork default reaches the client with no config passed.

    Previously this asserted the opposite (absent by default). Flipping the
    default is the whole change, so the assertion flips with it — and this is
    what fails if an upstream merge restores upstream's off default.
    """
    for tier in ("quick", "deep"):
        assert _kwargs({}, tier)["prompt_caching"] is True


def test_anthropic_client_injects_cache_control():
    llm = AnthropicClient("claude-sonnet-5", prompt_caching=True, api_key="test").get_llm()
    assert llm.model_kwargs == {"cache_control": {"type": "ephemeral"}}

    llm_off = AnthropicClient("claude-sonnet-5", api_key="test").get_llm()
    assert not llm_off.model_kwargs


def test_default_config_has_new_keys():
    assert DEFAULT_CONFIG["anthropic_effort_quick"] is None
    assert DEFAULT_CONFIG["anthropic_effort_deep"] is None
    # Fork default: caching on (2026-09-04). Upstream ships it off; this
    # assertion is what catches an upstream merge silently reverting it.
    assert DEFAULT_CONFIG["anthropic_prompt_caching"] is True
