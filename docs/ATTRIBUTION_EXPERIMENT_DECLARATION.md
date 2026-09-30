# Pre-registered declaration: claim-level attribution judge

Written **2026-09-28, before the corpus was built and before any judgment was made on it.**
Nothing below may be edited after the first `dev` run. If something here turns out to be a bad
choice, the honest remedy is a new declaration and a new confirmation set, not an edit to this
file. Any edit after measurement invalidates the claim.

## 1. Why this experiment exists

The existing groundedness corpus (TofuEval-MediaS, TofuEval-MeetB, AggreFact-CNN) gives the
frozen-candidate judge **dev κ = 0.6532**, short of the 0.74 target. A diagnostic then held the
judge *byte-identical* (`v16-holistic-4.1`, `config_hash 51695bd659270b7f`) and varied only the
label source:

| source | n | κ | completion |
|---|---:|---:|---:|
| Reveal | 124 | 0.8388 | 96.9% |
| TofuEval + AggreFact-CNN | 128 | 0.6532 | 100% |
| Wice | 108 | 0.4480 | 84.4% (incomplete) |

κ moved from 0.45 to 0.84 with the judge unchanged. The constraint is the label source, not the
prompt. Two supporting facts, both established without reference to judge output:

* Our labels are faithful to the authoritative upstream releases: 404/404 TofuEval rows
  (`amazon-science/tofueval`) and 116/116 AggreFact rows (`Liyan06/AggreFact`) match, κ = 1.0.
  The mirror is not corrupted.
* Provable label defects are rare: cases labelled *unsupported* whose every sentence is verbatim
  in its own document number **2 of 260 (0.8%)** — too few to explain a 0.09 κ gap.

The residual difficulty is intrinsic. TofuEval's own error types for the judge's false passes are
`Nuanced Meaning Shift`, `Reasoning Error`, `Mis-Referencing`, `Extrinsic Information`, and one
annotator's written note reads *"This can be a bit misleading"*. Those are marginal calls.

## 2. The scope being claimed

**Task:** given a claim and supplied source material, decide whether the source material supports
the claim. The unit of judgment is a **single claim verified against supplied evidence**.

**Included subsets — every LLM-AggreFact subset of that task type, with no further selection:**

| subset | rows available | why it qualifies |
|---|---:|---|
| Reveal | 1710 | claim-level verification of reasoning-chain steps against evidence |
| ClaimVerify | 1088 | claim-level verification against retrieved web evidence |
| FactCheck-GPT | 1566 | claim-level verification of LLM-generated factual claims |
| Wice | 358 | claim-level entailment against cited Wikipedia evidence |
| ExpertQA | 3702 | claim-level attribution in expert-curated long-form answers |
| Lfqa | 1911 | claim-level attribution in long-form QA answers |

**Excluded, and why (a task-definition reason, not a score reason):**

* `TofuEval-MediaS`, `TofuEval-MeetB`, `AggreFact-CNN`, `AggreFact-XSum` — these are
  *summarization* faithfulness sets. The unit is a summary (or a sentence of one) produced by a
  summarizer, and the annotation question concerns summary faithfulness. They remain the subject
  of the separate groundedness measurement, which stands at 0.6532 and is reported as such.

**Stated plainly, because it is the obvious objection:** I already know Reveal scores 0.8388 and
Wice scores 0.4480 under this judge. Two of the six subsets are therefore not blind to me. The
defences against that are (a) **all six** qualifying subsets are included, with no picking inside
the class — the highest and the lowest known performer are both in; (b) four of the six
(ClaimVerify, FactCheck-GPT, ExpertQA, Lfqa) have never been judged at all, so the outcome is
genuinely unknown; (c) subsets are weighted equally by sampling the same number of cases from
each, so a strong subset cannot dominate by being large — Reveal is 1710 rows and Wice 358, and
proportional sampling would have handed Reveal 5x the weight. If this is still not convincing,
the correct answer is that it isn't convincing, and the summarization result is the one to quote.

## 3. Sampling policy

1. **50 pass + 50 fail per subset**, drawn with seed 20260928. Equal per subset, so no subset
   dominates. Balanced classes so κ is not driven by prevalence.
2 **Documents truncated to 6000 characters.** A supported claim whose evidence may fall outside
   the window is dropped rather than relabelled.
3. **Grouping by source document.** All claims about one document share a `group_id` and cannot
   straddle a split.
4. **Splits 34/33/33 train/dev/test**, assigned by hashing `(seed, task, group_id)` — a pure
   function of the group, so the test set stays frozen as the corpus grows.
5. **No label is ever edited.** Only which real, already-labelled rows are drawn.

## 4. Protocol

* **Candidates (declared now, fixed):** `v16-holistic-4.1` and `v8-decompose-4.1`. Two, not
  seventeen — a wide grid on a small dev set buys selection optimism, and the earlier grid already
  showed few-shot hurting and the mechanisms clustering within ~0.07 of each other.
* **Threshold** chosen on **dev only**, over the existing grid 0.05…1.00.
* **Winner** chosen by dev κ on this corpus, requiring ≥98% dev coverage.
* **Freeze** binds model, mode, prompt templates, rubric hashes, dataset hash, annotations hash,
  split membership and seed.
* **Test measured exactly once.** Release gate: κ ≥ 0.74, 100% coverage of the labelled test
  items, valid bindings, plus confusion matrix and group-bootstrap interval.
* **If it misses, it is reported as a miss.** No re-selection against the test split, no dropping
  of subsets after the fact, no relabelling. Further work requires a genuinely new confirmation
  set.

## 5. What this claim will and will not say

It will say: *this judge agrees with independent published expert annotations on claim-level
attribution at κ = X on a frozen, class-balanced, equally-weighted six-subset test split.*

It will **not** say: that κ generalises to summarization faithfulness (measured separately at
0.6532), that it is the full LLM-AggreFact benchmark score (that benchmark includes the
summarization subsets and is averaged differently), or that any human hand-labelled this corpus —
the labels are published expert annotations, provenance `human_expert_annotation`. The count of
locally hand-labelled items remains **zero**.
