"""Which model tier each graph node is wired to.

The two-tier split is a cost/quality decision worth an assertion: the quick
model carries the bulk of the run and the deep model is reserved for the nodes
whose output is acted on. Nothing else in the suite covers it —
``test_structured_agents`` hands an llm straight to ``create_trader``, so the
wiring in ``GraphSetup.setup_graph`` was previously unobserved and an upstream
merge could re-tier any node silently.

This fork deliberately diverges from upstream on one entry: the **Trader** runs
on the deep tier, because it produces the actual trade decision. If an upstream
merge reverts that line, ``test_trader_runs_on_the_deep_tier`` is what fails.
"""
from __future__ import annotations

import pytest

from tradingagents.graph import setup as setup_mod
from tradingagents.graph.conditional_logic import ConditionalLogic


QUICK = object()
DEEP = object()

# node factory -> the tier it must receive.
EXPECTED_TIER = {
    "create_market_analyst": "quick",
    "create_sentiment_analyst": "quick",
    "create_news_analyst": "quick",
    "create_fundamentals_analyst": "quick",
    "create_bull_researcher": "quick",
    "create_bear_researcher": "quick",
    "create_aggressive_debator": "quick",
    "create_neutral_debator": "quick",
    "create_conservative_debator": "quick",
    "create_research_manager": "deep",
    "create_portfolio_manager": "deep",
    # Fork divergence — upstream wires the Trader to the quick tier.
    "create_trader": "deep",
}


@pytest.fixture
def wiring(monkeypatch):
    """Build the real graph, recording which llm object each factory got.

    The factories are intercepted rather than the llms inspected, because
    ``create_trader`` binds its llm inside a closure at construction time. The
    assertion is still on object identity of the two sentinels, so a node wired
    to the wrong tier fails — this records real wiring, it does not restate it.
    """
    seen: dict[str, object] = {}

    def recorder(name):
        def factory(llm=None, *args, **kwargs):
            seen[name] = llm
            return lambda state, *a, **k: state
        return factory

    for name in EXPECTED_TIER:
        monkeypatch.setattr(setup_mod, name, recorder(name))

    analyst_keys = ("market", "social", "news", "fundamentals")
    graph_setup = setup_mod.GraphSetup(
        quick_thinking_llm=QUICK,
        deep_thinking_llm=DEEP,
        tool_nodes={k: (lambda state, *a, **k: state) for k in analyst_keys},
        conditional_logic=ConditionalLogic(),
    )
    graph_setup.setup_graph(selected_analysts=analyst_keys)
    return seen


def test_trader_runs_on_the_deep_tier(wiring):
    """The node that produces the trade decision gets the better model."""
    assert wiring["create_trader"] is DEEP


def test_every_node_is_wired_to_its_expected_tier(wiring):
    sentinel = {"quick": QUICK, "deep": DEEP}
    actual = {
        name: "deep" if obj is DEEP else "quick" if obj is QUICK else repr(obj)
        for name, obj in wiring.items()
    }
    assert actual == EXPECTED_TIER
    # Guard the guard: both sentinels must actually appear, or an "all quick"
    # regression would still satisfy a map that had drifted to all-quick.
    assert set(wiring.values()) == set(sentinel.values())


def test_the_deep_tier_stays_scarce(wiring):
    """Deep is the expensive tier — keep its membership deliberate.

    Three of twelve nodes. This fails when a node is promoted without the cost
    being considered, which is the whole point of tracking the split.
    """
    deep = sorted(n for n, obj in wiring.items() if obj is DEEP)
    assert deep == [
        "create_portfolio_manager",
        "create_research_manager",
        "create_trader",
    ]
