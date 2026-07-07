"""Structured-output binding picks the right method per Anthropic model.

Anthropic reasoning models (Fable/Mythos family, thinking always on) reject
the forced tool_choice used by langchain's default
``with_structured_output(method="function_calling")`` — every structured call
would 400 and silently fall back to free text. ``bind_structured`` must bind
those models with ``method="json_schema"`` and leave everything else on the
provider default.
"""

from pydantic import BaseModel

from tradingagents.agents.utils.structured import bind_structured
from tradingagents.llm_clients.anthropic_client import (
    requires_json_schema_structured_output,
)


class _Schema(BaseModel):
    action: str


class _RecordingLLM:
    def __init__(self, model: str):
        self.model = model
        self.recorded_kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.recorded_kwargs = kwargs
        return object()


def test_reasoning_model_detection():
    assert requires_json_schema_structured_output("claude-fable-5")
    assert requires_json_schema_structured_output("claude-mythos-5")
    assert requires_json_schema_structured_output("claude-mythos-preview")
    assert requires_json_schema_structured_output("claude-fable-6")  # forward-compatible
    assert not requires_json_schema_structured_output("claude-opus-4-8")
    assert not requires_json_schema_structured_output("claude-sonnet-5")
    assert not requires_json_schema_structured_output("claude-haiku-4-5")
    assert not requires_json_schema_structured_output("gpt-5.5")


def test_bind_structured_uses_json_schema_for_fable():
    llm = _RecordingLLM("claude-fable-5")
    assert bind_structured(llm, _Schema, "test_agent") is not None
    assert llm.recorded_kwargs == {"method": "json_schema"}


def test_bind_structured_default_method_for_opus():
    llm = _RecordingLLM("claude-opus-4-8")
    assert bind_structured(llm, _Schema, "test_agent") is not None
    assert llm.recorded_kwargs == {}


def test_bind_structured_default_for_models_without_model_attr():
    class _NoModelLLM:
        def with_structured_output(self, schema, **kwargs):
            self.recorded_kwargs = kwargs
            return object()

    llm = _NoModelLLM()
    assert bind_structured(llm, _Schema, "test_agent") is not None
    assert llm.recorded_kwargs == {}
