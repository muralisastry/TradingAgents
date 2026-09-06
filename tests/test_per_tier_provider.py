"""Per-tier providers: the quick tier and the deep tier may run on different
providers (e.g. Anthropic Haiku for the ~15 quick calls, GLM-5.3 for the ~3
deep synthesis calls). Each tier's kwargs and client come from ITS provider."""

import pytest

from tradingagents.graph import trading_graph as tg


def _graph(config):
    g = tg.TradingAgentsGraph.__new__(tg.TradingAgentsGraph)
    g.config = config
    g.callbacks = None
    return g


BASE = {
    "llm_provider": "glm", "deep_think_llm": "glm-5.3", "quick_think_llm": "claude-haiku-4-5",
    "backend_url": "https://glm.example/v4",
    "glm_thinking": True, "glm_reasoning_effort": None, "glm_reasoning_effort_deep": "max",
    "glm_reasoning_effort_quick": None, "anthropic_effort": None, "anthropic_effort_quick": "low",
    "anthropic_effort_deep": None, "anthropic_prompt_caching": True, "temperature": None,
    "llm_max_retries": None, "max_tokens": None,
}


@pytest.mark.unit
class TestTierProvider:
    def test_tier_falls_back_to_the_shared_provider(self):
        g = _graph({**BASE, "deep_llm_provider": None, "quick_llm_provider": None})
        assert g._tier_provider("deep") == "glm" and g._tier_provider("quick") == "glm"
        assert g._tier_provider() == "glm"

    def test_tier_override_wins_and_is_lowercased(self):
        g = _graph({**BASE, "quick_llm_provider": "Anthropic"})
        assert g._tier_provider("quick") == "anthropic" and g._tier_provider("deep") == "glm"

    def test_kwargs_follow_each_tiers_provider(self):
        g = _graph({**BASE, "quick_llm_provider": "anthropic"})
        deep = g._get_provider_kwargs(tier="deep")
        quick = g._get_provider_kwargs(tier="quick")
        # deep on GLM: thinking + max effort, no Anthropic keys
        assert deep["thinking"] == {"type": "enabled"} and deep["reasoning_effort"] == "max"
        assert "effort" not in deep and "prompt_caching" not in deep
        # quick on Anthropic: effort + caching, no GLM keys
        assert quick["effort"] == "low" and quick["prompt_caching"] is True
        assert "thinking" not in quick and "reasoning_effort" not in quick

    def test_clients_are_built_on_their_own_provider_and_backend_url_stays_with_the_shared_one(self, monkeypatch):
        seen = []
        monkeypatch.setattr(tg, "create_llm_client", lambda **kw: seen.append(kw) or object())
        g = _graph({**BASE, "quick_llm_provider": "anthropic"})
        g._build_tier_client("deep", {"reasoning_effort": "max"})
        g._build_tier_client("quick", {"effort": "low"})
        deep, quick = seen
        assert deep["provider"] == "glm" and deep["model"] == "glm-5.3" and deep["base_url"] == "https://glm.example/v4"
        assert quick["provider"] == "anthropic" and quick["model"] == "claude-haiku-4-5" and quick["base_url"] is None
        assert deep["reasoning_effort"] == "max" and quick["effort"] == "low"

    def test_env_overrides_reach_the_config(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_QUICK_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("TRADINGAGENTS_DEEP_LLM_PROVIDER", "glm")
        import importlib

        from tradingagents import default_config

        cfg = importlib.reload(default_config).DEFAULT_CONFIG
        assert cfg["quick_llm_provider"] == "anthropic" and cfg["deep_llm_provider"] == "glm"
