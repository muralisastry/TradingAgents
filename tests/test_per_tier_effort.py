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
    assert "prompt_caching" not in _kwargs({}, "quick")


def test_anthropic_client_injects_cache_control():
    llm = AnthropicClient("claude-sonnet-5", prompt_caching=True, api_key="test").get_llm()
    assert llm.model_kwargs == {"cache_control": {"type": "ephemeral"}}

    llm_off = AnthropicClient("claude-sonnet-5", api_key="test").get_llm()
    assert not llm_off.model_kwargs


def test_default_config_has_new_keys():
    assert DEFAULT_CONFIG["anthropic_effort_quick"] is None
    assert DEFAULT_CONFIG["anthropic_effort_deep"] is None
    assert DEFAULT_CONFIG["anthropic_prompt_caching"] is False
