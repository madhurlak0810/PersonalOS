"""The one place a model configuration becomes the model-backed ports.

Four ports in this system are backed by a model: intent routing, recruiter
event extraction, semantic assessment of a posting and artifact drafting.
Each already depends on nothing but the `StructuredChatModel` slice of a chat
model, so which model sits behind them is a composition decision. This module
makes it a *configuration* decision: a `ModelConfig` names a provider and a
model per role, and `ModelGateway` builds the ports from it.

    gateway = ModelGateway(ModelConfig.load("evals/model_benchmarks/configs/rules.json"))
    SupervisorGraph(gateway.intent_classifier(), job_subgraph)

Swapping Claude for a local model, or one local model for another, is a
different file. No graph, node or prompt changes, which is what lets
`evals.harness` run the same golden set against every configuration and
compare them.

Providers:

- `rules`: the deterministic stand-ins. `KeywordIntentClassifier`,
  `RuleBasedRecruiterEventExtractor`, an assessor that always declines (so
  `HybridJobMatcher` scores on keyword overlap, its documented fallback), and
  no draft writer. Needs nothing installed and no network.
- `anthropic`: Claude through `langchain-anthropic` (the `llm` extra).
- `ollama`: a local model served by Ollama, through `langchain-ollama`
  (the `local` extra).
- `openai_compatible`: any server speaking the OpenAI chat API -- llama.cpp,
  vLLM, LM Studio -- through `langchain-openai` (the `local` extra).

Provider packages are imported when a model is built, never at module scope,
so importing this module -- and running on `rules` -- needs none of them.
"""

import json
import os
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from personalos.domain.job_search import (
    EvidenceRecord,
    NormalizedPosting,
    SearchProfile,
    SemanticAssessment,
)
from personalos.models.artifact_drafting import StructuredLLMDraftWriter
from personalos.models.job_matching import StructuredLLMSemanticAssessor
from personalos.models.recruiter_events import (
    RuleBasedRecruiterEventExtractor,
    StructuredLLMRecruiterEventExtractor,
)
from personalos.models.routing import (
    DEFAULT_CLASSIFIER_MODEL,
    KeywordIntentClassifier,
    StructuredChatModel,
    StructuredLLMIntentClassifier,
)

Provider = Literal["rules", "anthropic", "ollama", "openai_compatible"]

#: Providers that need a chat model, and so a `model` name.
CHAT_PROVIDERS: frozenset[str] = frozenset({"anthropic", "ollama", "openai_compatible"})


class ModelRole(str, Enum):
    """The model-backed ports, by the name a configuration overrides them under."""

    INTENT_CLASSIFIER = "intent_classifier"
    RECRUITER_EVENT_EXTRACTOR = "recruiter_event_extractor"
    SEMANTIC_ASSESSOR = "semantic_assessor"
    DRAFT_WRITER = "draft_writer"


class ModelSpec(BaseModel):
    """Which model answers for one role, and how to reach it."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    provider: Provider = "rules"
    #: The provider's model name (`claude-opus-5`, `llama3.1:8b`, ...). Ignored by `rules`.
    model: str | None = None
    #: Server URL for `ollama` and `openai_compatible`; the client's default when unset.
    base_url: str | None = None
    #: Environment variable holding the API key for `openai_compatible`. The key
    #: itself never appears in a configuration file.
    api_key_env: str | None = None
    #: Passed to the chat model's constructor as keyword arguments
    #: (`temperature`, `num_ctx`, `max_tokens`, ...).
    params: dict[str, Any] = Field(default_factory=dict)
    #: Schema-repair attempts; each port's own default when unset.
    max_attempts: int | None = Field(default=None, ge=1)

    def label(self) -> str:
        """`provider:model`, or just `rules`."""
        return self.provider if self.provider == "rules" else f"{self.provider}:{self.model}"


class ModelConfig(BaseModel):
    """A named model configuration: one default spec, and per-role overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    name: str
    description: str = ""
    default: ModelSpec = Field(default_factory=ModelSpec)
    roles: dict[ModelRole, ModelSpec] = Field(default_factory=dict)

    def spec_for(self, role: ModelRole) -> ModelSpec:
        return self.roles.get(role, self.default)

    def resolved(self) -> dict[str, str]:
        """`role -> provider:model`, for every role: what a run actually used."""
        return {role.value: self.spec_for(role).label() for role in ModelRole}

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        """Read a configuration from a JSON file."""
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


#: Builds a chat model for a spec whose provider is in `CHAT_PROVIDERS`.
ChatModelFactory = Callable[[ModelSpec], StructuredChatModel]


class SemanticAssessmentUnavailable(RuntimeError):
    """Raised by `DecliningSemanticAssessor`: this configuration has no assessor model."""


class DecliningSemanticAssessor:
    """The `rules` provider's semantic assessor: there is no model, so it declines.

    `HybridJobMatcher` treats a raising assessor as "score on keyword overlap
    and say so in the match's risks". That is the deterministic behaviour a
    configuration without an assessor model should get, and declining here
    routes to it rather than inventing an assessment that looks model-made.
    """

    async def assess(
        self,
        posting: NormalizedPosting,
        profile: SearchProfile,
        evidence: Sequence[EvidenceRecord],
    ) -> SemanticAssessment:
        raise SemanticAssessmentUnavailable("no semantic assessor model is configured")


def _missing(provider: str, extra: str) -> ImportError:
    return ImportError(
        f"the '{provider}' model provider requires the '{extra}' extra: "
        f"pip install 'personalos[{extra}]'"
    )


def default_chat_model_factory(spec: ModelSpec) -> StructuredChatModel:
    """Build the LangChain chat model a spec names."""
    params = dict(spec.params)
    if spec.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise _missing(spec.provider, "llm") from exc
        return ChatAnthropic(model=spec.model or DEFAULT_CLASSIFIER_MODEL, **params)

    if spec.base_url:
        params["base_url"] = spec.base_url
    if spec.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise _missing(spec.provider, "local") from exc
        return ChatOllama(model=spec.model, **params)

    if spec.provider == "openai_compatible":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise _missing(spec.provider, "local") from exc
        # Local servers usually ignore the key, but the client will not start without one.
        params["api_key"] = os.environ.get(spec.api_key_env, "") if spec.api_key_env else "unused"
        return ChatOpenAI(model=spec.model, **params)

    raise ValueError(f"provider '{spec.provider}' is not backed by a chat model")


class ModelGateway:
    """Builds every model-backed port from one `ModelConfig`.

    Each method returns a fresh port, so two graphs built from one gateway
    share a configuration and nothing else. `chat_model_factory` is the seam
    for a test double or a recording wrapper; by default it is
    `default_chat_model_factory`.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        chat_model_factory: ChatModelFactory | None = None,
    ):
        for role in ModelRole:
            spec = config.spec_for(role)
            if spec.provider in CHAT_PROVIDERS and not spec.model:
                raise ValueError(f"{config.name}: {role.value} uses {spec.provider} but no model")
        self.config = config
        self._chat_model = chat_model_factory or default_chat_model_factory

    def _spec(self, role: ModelRole) -> tuple[ModelSpec, dict[str, Any]]:
        spec = self.config.spec_for(role)
        kwargs = {} if spec.max_attempts is None else {"max_attempts": spec.max_attempts}
        return spec, kwargs

    def intent_classifier(self) -> KeywordIntentClassifier | StructuredLLMIntentClassifier:
        spec, kwargs = self._spec(ModelRole.INTENT_CLASSIFIER)
        if spec.provider == "rules":
            return KeywordIntentClassifier()
        return StructuredLLMIntentClassifier(self._chat_model(spec), **kwargs)

    def recruiter_event_extractor(
        self,
    ) -> RuleBasedRecruiterEventExtractor | StructuredLLMRecruiterEventExtractor:
        spec, kwargs = self._spec(ModelRole.RECRUITER_EVENT_EXTRACTOR)
        if spec.provider == "rules":
            return RuleBasedRecruiterEventExtractor()
        return StructuredLLMRecruiterEventExtractor(self._chat_model(spec), **kwargs)

    def semantic_assessor(self) -> DecliningSemanticAssessor | StructuredLLMSemanticAssessor:
        spec, kwargs = self._spec(ModelRole.SEMANTIC_ASSESSOR)
        if spec.provider == "rules":
            return DecliningSemanticAssessor()
        return StructuredLLMSemanticAssessor(self._chat_model(spec), **kwargs)

    def draft_writer(self) -> StructuredLLMDraftWriter | None:
        """The draft writer, or None: there is no deterministic one to stand in."""
        spec, kwargs = self._spec(ModelRole.DRAFT_WRITER)
        if spec.provider == "rules":
            return None
        return StructuredLLMDraftWriter(self._chat_model(spec), **kwargs)


__all__ = [
    "CHAT_PROVIDERS",
    "ModelRole",
    "ModelSpec",
    "ModelConfig",
    "ChatModelFactory",
    "SemanticAssessmentUnavailable",
    "DecliningSemanticAssessor",
    "default_chat_model_factory",
    "ModelGateway",
]
