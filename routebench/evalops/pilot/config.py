"""Frozen configuration for the bounded RAGTruth LoRA pilot (one training configuration, one untuned baseline).

Everything that can change a result is declared here and hashed. A run refuses to start if the recorded hash differs from the
hash of this file's `PILOT`. Nothing here trains or downloads anything.

Label semantics (unchanged from the ingestion manifest):
    label 1 = SUPPORTED by the source document, label 0 = UNSUPPORTED.
    The model score is P(supported): HIGHER = MORE SUPPORTED. A sentence is predicted UNSUPPORTED iff score < threshold.
    For detection metrics the POSITIVE class is UNSUPPORTED (it is the rare class, about 6.8% of training sentences).
Scoring path (identical for the untuned baseline and the tuned model, so the comparison isolates training):
    text = "predict: " + source_document + EOS + sentence; first decoder step from start token 0; logits of vocabulary ids
    (3 = unsupported, 209 = supported) ; score = softmax over those two logits, index 1. This is the upstream MiniCheck
    Flan-T5 scoring path at commit b58b9fa, except that the source document is passed WHOLE (no chunking, no truncation).
Source-length handling: an example whose tokenised input exceeds `max_input_tokens` is EXCLUDED from training and from every
    evaluation (baseline and tuned identically) and counted in the report; nothing is truncated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PilotConfig:
    name: str = "ragtruth-lora-pilot-v1"
    # ---- model -------------------------------------------------------------------------------------------------------------
    base_model: str = "lytang/MiniCheck-Flan-T5-Large"
    base_revision: str = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
    base_licence: str = "MIT (model card)"
    prompt_prefix: str = "predict: "
    decoder_start_token_id: int = 0
    unsupported_token_id: int = 3
    supported_token_id: int = 209
    # ---- data --------------------------------------------------------------------------------------------------------------
    manifest_version: str = "ragtruth-training-manifest-v5"
    manifest_sha256: str = "1645dd3fd9455002db8bec4b2fa0f2dbe40d69a11cac4133ecfecc821e2c4da2"  # frozen v5 manifest
    select_fraction: float = 0.2  # share of TRAIN groups held out as `train_select` (threshold / epoch choice)
    select_seed: int = 20261102
    max_input_tokens: int = 2048
    # ---- loss and class balance -------------------------------------------------------------------------------------------------
    loss: str = "two-way cross-entropy on the (unsupported, supported) token logits, unweighted"
    weight_unsupported: float = 1.0
    weight_supported: float = 1.0
    # Imbalance (about 6.9% unsupported) is handled by SAMPLING, not by loss weights: a fixed training sample of every unsupported
    # example up to `train_unsupported_max` plus `train_supported_per_unsupported` supported examples per unsupported one, both drawn
    # by a seeded shuffle from train_fit after the length exclusion. The decision threshold is chosen on the natural-prevalence
    # train_select sample, so the sampling ratio does not leak into the operating point.
    train_unsupported_max: int = 500
    train_supported_per_unsupported: int = 2
    sample_seed: int = 20261104
    # ---- LoRA training (bf16, gradient checkpointing) ---------------------------------------------------------------------------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "v")
    learning_rate: float = 2e-4
    epochs: int = 1
    micro_batch: int = 2
    grad_accum: int = 8
    warmup_fraction: float = 0.05
    weight_decay: float = 0.0
    train_seed: int = 20261102
    # ---- threshold and aggregation -----------------------------------------------------------------------------------------------
    threshold_grid: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
    threshold_objective: str = "balanced accuracy, UNSUPPORTED = positive, on train_select only; ties -> closest to 0.5"
    epoch_choice: str = "single epoch; the final checkpoint is used (no checkpoint selection)"
    # Label-blind, seeded, document-group samples used for scoring, sized so that BOTH models can be scored inside the caps:
    select_eval_sentences: int = 800
    validation_eval_sentences: int = 1400
    response_aggregation: str = "response score = min over its in-scope sentence scores; predicted supported iff min >= threshold"
    # ---- evaluation and stop rule ---------------------------------------------------------------------------------------------------
    primary_metric: str = "sentence-level balanced accuracy on VALIDATION (unsupported = positive)"
    secondary_metrics: tuple[str, ...] = ("unsupported recall", "unsupported precision", "PR-AUC (unsupported)", "Cohen kappa",
                                          "response-level balanced accuracy")
    min_improvement: float = 0.03
    max_recall_drop: float = 0.02
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 20261103
    validation_uses: int = 1  # VALIDATION is scored once per model, for the go/no-go decision only; never for tuning
    # ---- resource caps (seconds) ------------------------------------------------------------------------------------------------------
    total_seconds: int = 7200
    startup_seconds: int = 400  # process start + model load, before the first phase begins; counts toward the total
    watchdog_grace_seconds: int = 15  # external watchdog fires this long after a phase cap, so cooperative stops come first
    phase_seconds: dict[str, int] = field(default_factory=lambda: {
        "feasibility": 300, "baseline_eval": 1100, "training": 3300, "tuned_eval": 1100, "slack": 1000})
    # The feasibility probe must exercise REAL optimizer updates: micro_batch x grad_accum examples per update.
    probe_optimizer_updates: int = 2
    probe_score_examples: int = 16
    # After the feasibility probe (a few timed forward/backward steps at the declared micro-batch, inside the cap) the run continues
    # only if seconds/example x planned examples fits every remaining phase cap; otherwise it stops and reports "not feasible".
    feasibility_rule: str = "projected phase time <= remaining phase cap for training, baseline_eval and tuned_eval, else stop"
    api_spend_usd: float = 0.0

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()


PILOT = PilotConfig()


@dataclass(frozen=True)
class PilotConfigV2(PilotConfig):
    """Corrected pilot (v2). The first pilot (`PILOT`, result directory `pilot_run_v1`) stopped INCOMPLETE because its feasibility probe
    scored short examples (mean 573 tokens vs 1,007 in production). Everything about labels, scoring path, LoRA recipe, threshold
    rule, aggregation, success criteria and the 7,200 s total is unchanged; what changes is how feasibility is measured and how the
    run is sized and observed. Sample reductions are predetermined and label/prediction-blind: the validation sample (45 document
    groups, every sampled sentence) is preserved in full; the selection and training samples shrink.
    """

    name: str = "ragtruth-lora-pilot-v2"
    result_dir: str = "pilot_run_v2"
    freeze_file: str = "v5/pilot_freeze_v4.json"
    # ---- sizing (all label/prediction-blind) -------------------------------------------------------------------------------------------
    train_unsupported_max: int = 350
    train_supported_per_unsupported: int = 2
    select_eval_sentences: int = 800  # COVERAGE rule: the same 24 selection groups as the first pilot
    validation_eval_sentences: int = 1400  # COVERAGE rule: the same 45 validation groups as the first pilot
    select_responses_per_group: int = 4  # whole responses kept per selection group (seeded hash order)
    validation_responses_per_group: int = 6  # keeps every sampled validation response, so validation is preserved in full
    # ---- feasibility: stratified probe and the 80% rules ----------------------------------------------------------------------------------
    probe_mode: str = "stratified"
    probe_score_batches: int = 8  # production batches at evenly spaced length positions, plus the most token-heavy batch
    feasibility_fraction: float = 0.8  # each projected phase <= this fraction of its remaining phase budget
    overall_fraction: float = 0.8  # measured startup + probe + projected phases + finalisation allowance <= this fraction of the total
    finalize_allowance_seconds: int = 300  # saving, hashing and verifying artifacts
    # ---- observability ---------------------------------------------------------------------------------------------------------------------
    heartbeat_every: int = 10  # log counts, elapsed time, rate, RAM and VRAM every N batches / micro-batches
    phase_seconds: dict[str, int] = field(default_factory=lambda: {
        "feasibility": 450, "baseline_eval": 1500, "training": 2500, "tuned_eval": 1500, "slack": 850})


PILOT_V2 = PilotConfigV2()
CONFIGS: dict[str, PilotConfig] = {"v1": PILOT, "v2": PILOT_V2}
