"""`ModelGateway`: a configuration file decides which model backs each port.

The acceptance criterion this guards is the swap: changing models is changing
a `ModelConfig`, and nothing that consumes the ports notices.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from personalos.bootstrap import build_model_gateway
from personalos.models.artifact_drafting import StructuredLLMDraftWriter
from personalos.models.gateway import (
    DecliningSemanticAssessor,
    ModelConfig,
    ModelGateway,
    ModelRole,
    ModelSpec,
    SemanticAssessmentUnavailable,
)
from personalos.models.job_matching import StructuredLLMSemanticAssessor
from personalos.models.recruiter_events import (
    RuleBasedRecruiterEventExtractor,
    StructuredLLMRecruiterEventExtractor,
)
from personalos.models.routing import KeywordIntentClassifier, StructuredLLMIntentClassifier
from tests.fixtures import job_search_fakes as fakes
from tests.fixtures.node_contracts import ReplayChatModel

CONFIGS = Path(__file__).resolve().parents[2] / "evals" / "model_benchmarks" / "configs"


class RecordingFactory:
    """A `ChatModelFactory` that records the specs it was asked for."""

    def __init__(self):
        self.specs: list[ModelSpec] = []

    def __call__(self, spec: ModelSpec) -> ReplayChatModel:
        self.specs.append(spec)
        return ReplayChatModel(name=spec.label())


def test_rules_builds_the_deterministic_ports_and_no_draft_writer():
    gateway = ModelGateway(ModelConfig(name="rules"), chat_model_factory=RecordingFactory())

    assert isinstance(gateway.intent_classifier(), KeywordIntentClassifier)
    assert isinstance(gateway.recruiter_event_extractor(), RuleBasedRecruiterEventExtractor)
    assert isinstance(gateway.semantic_assessor(), DecliningSemanticAssessor)
    assert gateway.draft_writer() is None
    assert gateway._chat_model.specs == []


async def test_the_rules_assessor_declines_so_matching_falls_back_to_keywords():
    with pytest.raises(SemanticAssessmentUnavailable):
        await DecliningSemanticAssessor().assess(fakes.posting(), fakes.profile(), ())


def test_a_chat_provider_backs_every_role_with_the_model_it_names():
    factory = RecordingFactory()
    config = ModelConfig(
        name="local", default=ModelSpec(provider="ollama", model="qwen2.5:7b", max_attempts=4)
    )
    gateway = ModelGateway(config, chat_model_factory=factory)

    classifier = gateway.intent_classifier()
    extractor = gateway.recruiter_event_extractor()

    assert isinstance(classifier, StructuredLLMIntentClassifier)
    assert isinstance(extractor, StructuredLLMRecruiterEventExtractor)
    assert isinstance(gateway.semantic_assessor(), StructuredLLMSemanticAssessor)
    assert isinstance(gateway.draft_writer(), StructuredLLMDraftWriter)
    assert {spec.label() for spec in factory.specs} == {"ollama:qwen2.5:7b"}
    assert classifier.max_attempts == extractor.max_attempts == 4


def test_a_role_override_changes_that_role_only():
    factory = RecordingFactory()
    config = ModelConfig(
        name="hybrid",
        roles={
            ModelRole.RECRUITER_EVENT_EXTRACTOR: ModelSpec(provider="ollama", model="llama3.1:8b")
        },
    )
    gateway = ModelGateway(config, chat_model_factory=factory)

    assert isinstance(gateway.intent_classifier(), KeywordIntentClassifier)
    assert isinstance(gateway.recruiter_event_extractor(), StructuredLLMRecruiterEventExtractor)
    assert [spec.label() for spec in factory.specs] == ["ollama:llama3.1:8b"]
    assert config.resolved()["recruiter_event_extractor"] == "ollama:llama3.1:8b"
    assert config.resolved()["intent_classifier"] == "rules"


def test_a_chat_provider_without_a_model_is_refused_at_construction():
    with pytest.raises(ValueError, match="no model"):
        ModelGateway(ModelConfig(name="broken", default=ModelSpec(provider="anthropic")))


def test_a_misspelled_field_is_refused_rather_than_ignored():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"name": "x", "default": {"provider": "rules", "modle": "y"}})


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.json")), ids=lambda p: p.stem)
def test_every_shipped_benchmark_config_builds_every_port(path):
    config = ModelConfig.load(path)
    gateway = ModelGateway(config, chat_model_factory=RecordingFactory())

    assert config.name == path.stem
    gateway.intent_classifier()
    gateway.recruiter_event_extractor()
    gateway.semantic_assessor()
    gateway.draft_writer()


def test_bootstrap_reads_the_configured_file_and_defaults_to_rules():
    assert build_model_gateway("").config.name == "rules"
    hybrid = build_model_gateway(str(CONFIGS / "hybrid-local-extraction.json"))
    assert hybrid.config.spec_for(ModelRole.RECRUITER_EVENT_EXTRACTOR).provider == "ollama"
