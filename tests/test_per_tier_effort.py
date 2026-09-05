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


# ── GLM thinking + reasoning effort ───────────────────────────────────────
# GLM uses the OpenAI-compatible client but its provider string is "glm", so it
# never matched the `openai` branch: `thinking` and `reasoning_effort` were
# both dropped and the run reasoned at neither. Nothing errored — the only
# symptom was a model quietly running in default mode.

def _glm_kwargs(config_overrides, tier):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {**DEFAULT_CONFIG, "llm_provider": "glm", **config_overrides}
    return graph._get_provider_kwargs(tier=tier)


def test_glm_thinking_is_on_by_default():
    assert _glm_kwargs({}, "deep")["thinking"] == {"type": "enabled"}


def test_glm_thinking_can_be_disabled():
    assert "thinking" not in _glm_kwargs({"glm_thinking": False}, "deep")


def test_glm_reasoning_effort_is_per_tier():
    overrides = {"glm_reasoning_effort": "medium", "glm_reasoning_effort_deep": "max"}
    assert _glm_kwargs(overrides, "deep")["reasoning_effort"] == "max"
    assert _glm_kwargs(overrides, "quick")["reasoning_effort"] == "medium"


def test_glm_reasoning_effort_absent_when_unset():
    assert "reasoning_effort" not in _glm_kwargs({}, "deep")


def test_glm_cn_gets_the_same_treatment():
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {**DEFAULT_CONFIG, "llm_provider": "glm-cn",
                    "glm_reasoning_effort_deep": "max"}
    kw = graph._get_provider_kwargs(tier="deep")
    assert kw["thinking"] == {"type": "enabled"}
    assert kw["reasoning_effort"] == "max"


def test_anthropic_branch_is_unaffected_by_the_glm_keys():
    """The GLM keys must not leak into an Anthropic run."""
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {**DEFAULT_CONFIG, "llm_provider": "anthropic"}
    kw = graph._get_provider_kwargs(tier="deep")
    assert "thinking" not in kw and "reasoning_effort" not in kw


# ── the kwargs must survive into the constructed client ───────────────────
# Asserting on _get_provider_kwargs alone would pass while the client silently
# discarded both: reasoning_effort is gated by a model regex that rejects every
# GLM name, and `thinking` is not an OpenAI field, so a bare kwarg would be
# dropped by the SDK rather than sent.

def _glm_llm(model="glm-5.3", **kw):
    from tradingagents.llm_clients.openai_client import OpenAIClient

    return OpenAIClient(model, provider="glm", api_key="test", **kw).get_llm()


def test_glm_thinking_reaches_extra_body():
    llm = _glm_llm(thinking={"type": "enabled"})
    assert llm.extra_body == {"thinking": {"type": "enabled"}}


def test_glm_reasoning_effort_survives_the_model_gate():
    assert _glm_llm(reasoning_effort="max").reasoning_effort == "max"


def test_openai_model_gate_still_rejects_non_reasoning_models():
    """The GLM allowance must not loosen the gate for native OpenAI."""
    from tradingagents.llm_clients.openai_client import OpenAIClient

    llm = OpenAIClient("gpt-4o", provider="openai", api_key="test",
                       reasoning_effort="max").get_llm()
    assert not llm.reasoning_effort


def test_glm_47_and_5x_accept_effort_but_older_glm_does_not():
    from tradingagents.llm_clients.openai_client import _supports_reasoning_effort

    assert _supports_reasoning_effort("glm-5.3", "glm")
    assert _supports_reasoning_effort("glm-4.7", "glm-cn")
    assert not _supports_reasoning_effort("glm-4.5-air", "glm")
