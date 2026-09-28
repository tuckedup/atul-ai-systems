"""Adapters for specialised document-claim support checkers (MiniCheck, AlignScore).

These are thin wrappers around upstream code, not reimplementations. The upstream algorithms were
read at pinned commits and are reproduced by *calling* them, because a "reimplementation from the
paper" is a different system with the same name -- and the number that matters here is an empirical
one, so the implementation has to be the one whose published numbers we are comparing against.

    MiniCheck   https://github.com/Liyan06/MiniCheck    b58b9fa  Apache-2.0
    AlignScore  https://github.com/yuh-zha/AlignScore   a0936d5  MIT

Code licences do not license model weights. The MiniCheck README asks for separate commercial
licensing for Bespoke-MiniCheck-7B, so `DEFAULT_SPECS` selects the non-LLM `flan-t5-large` variant
and `CheckerSpec.weights_licence_checked` has to be set deliberately before a run.

**Nothing here imports torch, transformers, nltk or downloads weights at module import.** Every
heavy import is inside a loader function, so the whole test suite runs on a machine with none of
them -- which is the only way tests can cover the segmentation and aggregation logic before anyone
spends a GPU-hour. `FakeSupportChecker` exists for exactly that.

What is faithful and what is adaptation, stated separately because conflating them is how a
borrowed benchmark number becomes an unearned claim:

*   **Faithful.** Document chunking, sentence segmentation, per-chunk scoring and the max-over-
    chunks aggregation are upstream's, performed by upstream's code. MiniCheck's own binary rule
    (`max_support_prob > 0.5`) is available as `aggregation="upstream_binary"`.
*   **Adaptation.** RouteBench needs ONE pass/fail verdict per response, and MiniCheck is a
    sentence-level checker whose README explicitly says "we leave the user to decide how to
    aggregate the results from multiple sentences". That choice is ours, it is declared in
    `CheckerSpec.aggregation`, and which one is right is an empirical question settled on dev --
    never asserted here and never selected on test.

A missing or failed checker output is an ERROR with a status, never a zero score and never a failed
label. That is the same rule `Judgment.usable` enforces for the LLM judge, and for the same reason:
`aisys.evals.run` collapses grader exceptions into `score=0.0`, which converts an out-of-memory into
"the model said unsupported" and biases agreement in a direction that depends on input length.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .dataset import CaseRecord

__all__ = [
    "AGGREGATIONS",
    "DEFAULT_SPECS",
    "CheckerError",
    "CheckerResult",
    "CheckerSpec",
    "FakeSupportChecker",
    "SupportChecker",
    "aggregate_sentence_scores",
    "check_case",
    "load_checker",
    "split_claim_sentences",
]


class CheckerError(RuntimeError):
    pass


#: Upstream revisions these adapters were written against. Recorded on every result: a support
#: score is only comparable to a published number if the code that produced it is the published
#: code, and "we used MiniCheck" without a revision is not a reproducible statement.
UPSTREAM = {
    "minicheck": {
        "repo": "https://github.com/Liyan06/MiniCheck",
        "commit": "b58b9fa69acbd1015ec970fa65dd752413a053d2",
        "code_licence": "Apache-2.0",
    },
    "alignscore": {
        "repo": "https://github.com/yuh-zha/AlignScore",
        "commit": "a0936d5afee642a46b22f6c02a163478447aa493",
        "code_licence": "MIT",
    },
}


# ---------------------------------------------------------------- aggregation (the adaptation)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


#: Ways of turning per-sentence support probabilities into one response-level score.
#:
#: All of these are OURS, not upstream's. MiniCheck scores (document, sentence) pairs and its README
#: says in as many words that aggregating a multi-sentence claim is left to the user. The
#: groundedness annotation question is "is EVERY claim in this response supported", which argues for
#: `min` -- one unsupported sentence should sink the response -- but `mean` and
#: `supported_fraction` give a dev threshold a smoother quantity to cut, and which of those actually
#: agrees best with the labels is not knowable from the annotation guidelines alone. So all three are
#: declared here, compared on dev, and the winner is frozen. Picking one on the strength of the
#: argument above and reporting it as upstream behaviour would be the error this docstring exists to
#: prevent.
AGGREGATIONS: dict[str, Callable[[Sequence[float]], float]] = {
    # Weakest sentence decides. Closest to the annotation question as written.
    "min": min,
    # Average support. Tolerant of one hedged sentence in a long response.
    "mean": _mean,
    # Fraction of sentences clearing upstream's own 0.5 support cutoff. Mirrors the decompose
    # judge's supported-fraction score, so the two mechanisms are compared on the same footing.
    "supported_fraction": lambda vs: sum(1 for v in vs if v > 0.5) / len(vs),
    # Upstream's binary rule applied to the whole response as a single claim, with no sentence
    # splitting. The faithful-reproduction setting; a baseline rather than a candidate.
    "upstream_binary": lambda vs: 1.0 if max(vs) > 0.5 else 0.0,
}


def aggregate_sentence_scores(scores: Sequence[float], aggregation: str) -> float:
    """Reduce per-sentence support probabilities to one response score.

    An empty sentence list raises rather than defaulting. A response with no checkable sentences is
    not "unsupported" and not "perfectly supported"; it is a case the checker could not answer, and
    `check_case` records it as an error. Returning 0.0 here would smuggle it into the metric as a
    failing verdict -- the same collapse this module's docstring refuses for exceptions.
    """
    if aggregation not in AGGREGATIONS:
        raise CheckerError(
            f"unknown aggregation {aggregation!r}; declared options are {sorted(AGGREGATIONS)}"
        )
    if not scores:
        raise CheckerError(
            "no sentence scores to aggregate; a response the checker could not segment is an "
            "error, not a score of zero"
        )
    for i, value in enumerate(scores):
        if not 0.0 <= float(value) <= 1.0:
            raise CheckerError(
                f"sentence score {i} is {value!r}; support probabilities must lie in [0, 1]"
            )
    return float(AGGREGATIONS[aggregation]([float(v) for v in scores]))


# ---------------------------------------------------------------- segmentation


def split_claim_sentences(
    text: str, *, splitter: Callable[[str], list[str]] | None = None
) -> list[str]:
    """Split a candidate response into the sentences the checker will score separately.

    MiniCheck is a `(document, sentence) -> [0, 1]` model and its README is explicit: "In order to
    fact-check a multi-sentence claim, the claim should first be broken up into sentences to achieve
    optimal performance." Skipping this step is not a neutral simplification -- upstream reports
    that whole-response scoring is measurably worse -- so the adapter does it, and records that it
    did.

    The default splitter is upstream's own (`nltk.tokenize.sent_tokenize` via MiniCheck's
    `sent_tokenize_with_newlines`, which preserves paragraph breaks). Substituting a different
    splitter changes the segmentation and therefore the score, so `splitter` exists for tests and
    the choice is recorded in every result rather than left implicit.
    """
    if splitter is not None:
        return [s for s in (part.strip() for part in splitter(text)) if s]
    try:
        from minicheck.inference import (
            sent_tokenize_with_newlines,  # type: ignore[import-not-found]
        )
    except ImportError as e:  # pragma: no cover - requires the upstream package
        raise CheckerError(
            "upstream MiniCheck is not importable, so its sentence splitter is unavailable. "
            "Install the pinned MiniCheck checkout, or pass an explicit `splitter` and accept that "
            "the segmentation is then no longer upstream's."
        ) from e
    return [s for s in (part.strip() for part in sent_tokenize_with_newlines(text)) if s]


# ---------------------------------------------------------------- the checker protocol


class SupportChecker(Protocol):
    """The shape both upstream libraries already expose, so neither needs modifying.

    `score(docs, claims)` takes parallel sequences and returns per-pair support probabilities in
    [0, 1]. MiniCheck returns a 4-tuple whose second element is the max-over-chunks support
    probability; AlignScore returns the list directly. `load_checker` normalises both to this.
    """

    def score(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        ...


@dataclass
class FakeSupportChecker:
    """A checker that returns scripted probabilities. For tests, before any weights are downloaded.

    This is not a convenience. Every property worth testing in this module -- sentence mapping,
    aggregation, empty and oversized input, error propagation, cache separation -- is independent of
    the neural network, and a test suite that can only run after a multi-gigabyte download is a test
    suite that does not run.
    """

    #: Per-claim score, keyed by the claim text. Claims absent from the map use `default`.
    scores: dict[str, float] = field(default_factory=dict)
    default: float = 0.5
    #: Raise on the nth call (1-based) to exercise the error path.
    fail_on_call: int | None = None
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)

    def score(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        self.calls.append((tuple(docs), tuple(claims)))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("scripted checker failure")
        if len(docs) != len(claims):
            raise CheckerError(f"{len(docs)} docs but {len(claims)} claims")
        return [self.scores.get(c, self.default) for c in claims]


# ---------------------------------------------------------------- spec


@dataclass(frozen=True)
class CheckerSpec:
    """One fully-specified checker run. Everything that can change a score lives here."""

    checker_id: str
    #: "minicheck" | "alignscore"
    backend: str
    #: Upstream model name. `flan-t5-large` is the default rather than Bespoke-MiniCheck-7B because
    #: the 7B weights carry a separate commercial-licensing request.
    model_name: str = "flan-t5-large"
    #: Document chunk size in tokens. None uses upstream's default for the model (400, or 500 for
    #: flan-t5-large). Recorded either way, since it changes the chunking and so the score.
    chunk_size: int | None = None
    #: Split the response into sentences and score each separately, as upstream's README requires.
    #: False reproduces the whole-response setting upstream used for AggreFact-CNN only.
    split_sentences: bool = True
    #: How per-sentence scores become one response score. OURS, not upstream's -- see AGGREGATIONS.
    aggregation: str = "min"
    #: AlignScore evaluation mode. `nli_sp` is the published baseline.
    evaluation_mode: str = "nli_sp"
    #: Local checkpoint path for AlignScore, which has no automatic download.
    ckpt_path: str = ""
    #: Set deliberately once the weights' licence has been checked for the intended use. Code
    #: licences do not license weights, and `load_checker` refuses without this.
    weights_licence_checked: bool = False
    notes: str = ""

    @property
    def upstream(self) -> dict[str, str]:
        if self.backend not in UPSTREAM:
            raise CheckerError(
                f"unknown backend {self.backend!r}; known backends are {sorted(UPSTREAM)}"
            )
        return UPSTREAM[self.backend]

    @property
    def config_hash(self) -> str:
        """Everything that can change a score, including the upstream revision.

        The upstream commit is in the hash deliberately. A support score from a different revision
        of the checker is a different measurement, and a cache that reused it across revisions would
        attribute one model's numbers to another -- the failure the judge's `prompt_template_hash`
        exists to prevent, in a new place.
        """
        payload = {
            k: v for k, v in asdict(self).items()
            if k not in ("checker_id", "notes", "weights_licence_checked")
        }
        payload["upstream_commit"] = self.upstream["commit"]
        blob = repr(sorted(payload.items())).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config_hash"] = self.config_hash
        d["upstream"] = self.upstream
        return d


#: Predeclared checker candidates, named before any of them is run. Both are scoped to
#: `summarize`: a per-claim support checker has no coherent reading on a SQL query, and running one
#: there would spend time to produce a number nobody should use.
DEFAULT_SPECS: tuple[CheckerSpec, ...] = (
    CheckerSpec(
        checker_id="minicheck-flan-t5-large-min",
        backend="minicheck",
        model_name="flan-t5-large",
        aggregation="min",
        notes="upstream sentence-level checking; weakest sentence decides the response",
    ),
    CheckerSpec(
        checker_id="minicheck-flan-t5-large-fraction",
        backend="minicheck",
        model_name="flan-t5-large",
        aggregation="supported_fraction",
        notes="same checker, supported-fraction aggregation to match the decompose judge's score",
    ),
    CheckerSpec(
        checker_id="minicheck-flan-t5-large-upstream",
        backend="minicheck",
        model_name="flan-t5-large",
        split_sentences=False,
        aggregation="upstream_binary",
        notes="faithful whole-response reproduction of upstream's own binary rule; the baseline",
    ),
    CheckerSpec(
        checker_id="alignscore-base-nli-sp",
        backend="alignscore",
        model_name="roberta-base",
        aggregation="mean",
        evaluation_mode="nli_sp",
        notes=("AlignScore nli_sp: max over premise chunks per response sentence, then mean over "
               "sentences -- upstream's own aggregation, so `mean` here is faithful rather than "
               "an adaptation"),
    ),
)


# ---------------------------------------------------------------- loading (lazy)


def load_checker(spec: CheckerSpec, *, device: int = -1, batch_size: int = 16) -> SupportChecker:
    """Import and construct the upstream checker. Every heavy import happens here, not above.

    Refuses without `weights_licence_checked`. The code licences (Apache-2.0, MIT) say nothing about
    the weights: the MiniCheck README asks for separate commercial licensing for its 7B model, and
    AlignScore's checkpoints are distributed separately from its MIT-licensed code. A default that
    downloaded weights on first use would make that decision silently, on someone else's behalf.
    """
    if not spec.weights_licence_checked:
        raise CheckerError(
            f"{spec.checker_id}: set weights_licence_checked=True once the MODEL WEIGHTS' terms "
            "have been checked for this use. The upstream CODE licence "
            f"({spec.upstream['code_licence']}) does not cover them."
        )
    if spec.backend == "minicheck":
        # Policy checks run BEFORE the import. Whether this variant is permitted is a decision
        # about the spec, not about the machine, so it must not depend on whether the package
        # happens to be installed -- otherwise the licence-restricted 7B model is refused on a
        # laptop and silently accepted on the box that has upstream installed.
        if spec.model_name not in ("roberta-large", "deberta-v3-large", "flan-t5-large"):
            raise CheckerError(
                f"{spec.model_name!r} is an LLM-backed MiniCheck variant. Those take the optional "
                "vLLM path and, for Bespoke-MiniCheck-7B, a separate commercial licence. Choose a "
                "non-LLM variant or approve that path explicitly."
            )
        try:
            from minicheck.minicheck import MiniCheck  # type: ignore[import-not-found]
        except ImportError as e:
            raise CheckerError(
                "upstream MiniCheck is not installed. Clone "
                f"{spec.upstream['repo']} at {spec.upstream['commit']} into an isolated "
                "environment and install only its audited runtime dependencies -- not its full "
                "benchmark requirements file, which pulls in unrelated packages."
            ) from e
        model = MiniCheck(model_name=spec.model_name, batch_size=batch_size)
        return _MiniCheckAdapter(model, chunk_size=spec.chunk_size)
    if spec.backend == "alignscore":
        if not spec.ckpt_path:
            raise CheckerError(
                "AlignScore has no automatic weight download; set `ckpt_path` to a local "
                "checkpoint whose licence you have checked."
            )
        try:
            from alignscore import AlignScore  # type: ignore[import-not-found]
        except ImportError as e:
            raise CheckerError(
                f"upstream AlignScore is not installed. Clone {spec.upstream['repo']} at "
                f"{spec.upstream['commit']} into an isolated environment."
            ) from e
        return AlignScore(
            model=spec.model_name, batch_size=batch_size, device=device,
            ckpt_path=spec.ckpt_path, evaluation_mode=spec.evaluation_mode, verbose=False,
        )
    raise CheckerError(f"unknown backend {spec.backend!r}")


@dataclass
class _MiniCheckAdapter:
    """Normalises MiniCheck's 4-tuple return to the `SupportChecker` protocol.

    `MiniCheck.score` returns `(pred_labels, max_support_probs, used_chunks,
    support_prob_per_chunk)`. The probability is taken, not the label: upstream's 0.5 cutoff is one
    of the aggregations on offer, and discarding the probability here would remove the continuous
    quantity a dev threshold needs.
    """

    model: Any
    chunk_size: int | None = None

    def score(self, docs: Sequence[str], claims: Sequence[str]) -> list[float]:
        out = self.model.score(
            docs=list(docs), claims=list(claims), chunk_size=self.chunk_size
        )
        _labels, max_support_probs, _chunks, _per_chunk = out
        return [float(p) for p in max_support_probs]


# ---------------------------------------------------------------- the common result record


@dataclass
class CheckerResult:
    """One checker verdict, bound to everything needed to reproduce or to distrust it."""

    case_id: str
    checker_id: str
    config_hash: str
    backend: str
    upstream_commit: str
    model_name: str
    #: Hashes of the exact texts scored, so a result cannot be silently re-attributed to a case
    #: whose document or response was edited afterwards.
    source_hash: str = ""
    candidate_hash: str = ""
    #: How the response was segmented and chunked. Part of the measurement, not metadata: change it
    #: and the score changes.
    segmentation: dict[str, Any] = field(default_factory=dict)
    #: Per-sentence support probabilities, and the sentences they belong to, in order.
    sentence_scores: list[float] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    #: The response-level score after `aggregation`. None when the checker did not produce one.
    score: float | None = None
    aggregation: str = ""
    #: "ok" | "checker_error" | "empty_input" | "invalid_output"
    status: str = "ok"
    error: str | None = None
    latency_ms: float = 0.0
    #: Local inference makes no provider charge, but zero is recorded with its basis rather than
    #: bare: an unexplained 0.00 is indistinguishable from an unpriced model, which is how a spend
    #: cap silently stops binding.
    cost_usd: float = 0.0
    cost_basis: str = "local_inference_no_provider_charge"
    trace_id: str = ""

    @property
    def usable(self) -> bool:
        """A failed check is an error, not an unsupported verdict. Same rule as `Judgment.usable`."""
        return self.status == "ok" and self.score is not None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def check_case(
    case: CaseRecord,
    spec: CheckerSpec,
    checker: SupportChecker,
    *,
    splitter: Callable[[str], list[str]] | None = None,
    max_source_chars: int = 200_000,
    trace_id: str = "",
) -> CheckerResult:
    """Score one case with one checker. Never raises for a checker failure; records it.

    The source is `case.context` alone. Not the reference, not the task input: the groundedness
    question is whether the response is supported by the DOCUMENT, and widening the premise would
    let a claim be "supported" by a gold answer the annotators never saw. That is the same scoping
    error as the evidence-location defect in `judge.py`, one layer out.
    """
    result = CheckerResult(
        case_id=case.case_id,
        checker_id=spec.checker_id,
        config_hash=spec.config_hash,
        backend=spec.backend,
        upstream_commit=spec.upstream["commit"],
        model_name=spec.model_name,
        source_hash=_hash_text(case.context),
        candidate_hash=_hash_text(case.candidate_output),
        aggregation=spec.aggregation,
        trace_id=trace_id,
    )
    source = case.context.strip()
    candidate = case.candidate_output.strip()

    if not source or not candidate:
        result.status = "empty_input"
        result.error = (
            f"source is {len(source)} chars and response is {len(candidate)} chars; a support "
            "check needs both. Recorded as an error rather than scored 0.0, which would enter the "
            "metric as an unsupported verdict."
        )
        return result

    if len(source) > max_source_chars:
        # Upstream chunks long documents rather than truncating, so a document this long is a
        # cost and correctness question for a human, not something to silently cut. Truncating
        # would change which sentences can support a claim, and the score would look normal.
        result.status = "invalid_output"
        result.error = (
            f"source is {len(source)} chars, over the {max_source_chars} limit. Upstream chunks "
            "rather than truncates, so this would be a very slow check; raise the limit "
            "deliberately rather than scoring a truncated document."
        )
        return result

    started = time.perf_counter()
    try:
        if spec.split_sentences:
            sentences = split_claim_sentences(candidate, splitter=splitter)
            if not sentences:
                result.status = "empty_input"
                result.error = "response produced no sentences after segmentation"
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result
        else:
            sentences = [candidate]

        scores = checker.score([source] * len(sentences), sentences)
        if len(scores) != len(sentences):
            raise CheckerError(
                f"checker returned {len(scores)} scores for {len(sentences)} sentences"
            )
        result.sentences = sentences
        result.sentence_scores = [float(s) for s in scores]
        result.score = aggregate_sentence_scores(result.sentence_scores, spec.aggregation)
        result.segmentation = {
            "split_sentences": spec.split_sentences,
            "n_sentences": len(sentences),
            "chunk_size": spec.chunk_size,
            "splitter": "injected" if splitter is not None else "upstream_nltk",
            "source_chars": len(source),
        }
    except CheckerError as e:
        result.status = "invalid_output"
        result.error = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 - a checker failure is data, not a crash
        result.status = "checker_error"
        result.error = f"{type(e).__name__}: {e}"
    result.latency_ms = (time.perf_counter() - started) * 1000
    return result
