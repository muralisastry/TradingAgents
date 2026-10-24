"""Per-tier providers: the quick tier and the deep tier may run on different
providers (e.g. Anthropic Haiku for the ~15 quick calls, GLM-5.3 for the ~3
deep synthesis calls). Each tier's kwargs and client come from ITS provider."""

import pytest

from tradingagents import default_config
from tradingagents.llm_clients import factory


def _tier_kwargs(config, tier):
    return factory.build_llm_kwargs({**config, "llm_provider": factory.tier_provider(config, tier)}, tier=tier)


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
        cfg = {**BASE, "deep_llm_provider": None, "quick_llm_provider": None}
        assert factory.tier_provider(cfg, "deep") == "glm" and factory.tier_provider(cfg, "quick") == "glm"

    def test_tier_override_wins_and_is_lowercased(self):
        cfg = {**BASE, "quick_llm_provider": "Anthropic"}
        assert factory.tier_provider(cfg, "quick") == "anthropic" and factory.tier_provider(cfg, "deep") == "glm"

    def test_upstream_name_wins_over_the_fork_alias(self):
        cfg = {**BASE, "deep_think_provider": "openai", "deep_llm_provider": "anthropic"}
        assert factory.tier_provider(cfg, "deep") == "openai"

    def test_kwargs_follow_each_tiers_provider(self):
        cfg = {**BASE, "quick_llm_provider": "anthropic"}
        deep = _tier_kwargs(cfg, "deep")
        quick = _tier_kwargs(cfg, "quick")
        # deep on GLM: thinking + max effort, no Anthropic keys
        assert deep["thinking"] == {"type": "enabled"} and deep["reasoning_effort"] == "max"
        assert "effort" not in deep and "prompt_caching" not in deep
        # quick on Anthropic: effort + caching, no GLM keys
        assert quick["effort"] == "low" and quick["prompt_caching"] is True
        assert "thinking" not in quick and "reasoning_effort" not in quick

    def test_clients_are_built_on_their_own_provider_and_backend_url_stays_with_the_shared_one(self, monkeypatch):
        seen = []
        monkeypatch.setattr(factory, "create_llm_client", lambda **kw: seen.append(kw) or object())
        cfg = {**BASE, "quick_llm_provider": "anthropic"}
        factory.create_tier_client(cfg, "deep")
        factory.create_tier_client(cfg, "quick")
        deep, quick = seen
        assert deep["provider"] == "glm" and deep["model"] == "glm-5.3" and deep["base_url"] == "https://glm.example/v4"
        assert quick["provider"] == "anthropic" and quick["model"] == "claude-haiku-4-5" and quick["base_url"] is None
        assert deep["reasoning_effort"] == "max" and quick["effort"] == "low"

    def test_env_overrides_reach_the_config(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_QUICK_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("TRADINGAGENTS_DEEP_LLM_PROVIDER", "glm")
        # build_default_config(), not importlib.reload: a reload rebinds the
        # module's DEFAULT_CONFIG with this test's env baked in, and monkeypatch
        # restores the env but not the module, so every later test inherited
        # deep_llm_provider="glm".
        cfg = default_config.build_default_config()
        assert cfg["quick_llm_provider"] == "anthropic" and cfg["deep_llm_provider"] == "glm"
