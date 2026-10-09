# Training-use permission status (2026-09-30)

Kept separate from technical readiness on purpose. **Nothing below is legal advice, and nothing in this repository grants
third-party rights.** A user's acceptance of a licence or a download approval is a decision about the user's own risk; it is not a
permission from the rights holders.

| Item | Evidence found | Status | What would resolve it |
| --- | --- | --- | --- |
| RAGTruth annotations and code | Repository LICENSE: MIT, (c) 2023 Particle Media (read at commit `c103204b...`). The RAGTruth paper fine-tunes a model on its training portion | Permissive for the labels and code | none needed for the labels |
| CNN/DM article text inside RAGTruth | Hugging Face dataset card for CNN/DM: "released under the Apache-2.0 License"; articles were written by CNN and Daily Mail journalists; the card does not address the copyright of that text | **Unresolved.** Dataset-card licence is not the same as the rights holders' permission for training use | Your explicit risk decision for research-only use, or the owners' terms; if neither, do not train |
| "Recent News" articles inside RAGTruth | The paper names only "a prestigious news platform"; no licence or permission was found | **Excluded** from every partition (never eligible) | Written terms from the platform or the authors |
| TofuEval, LLM-AggreFact, FRANK-derived rows, and the 70 upstream documents | TofuEval and LLM-AggreFact state the data should not be used to train NLP models; transcripts are CC-BY-NC-SA / research-only | **Not used in this pilot**: excluded by fingerprint from all partitions | n/a |
| Earlier internal artifacts (v17 and v23 few-shot packets, MiniCheck+v18 combiner) | Built from TofuEval-derived TRAIN rows | **Not cleared**; not release-ready; not used in this pilot | Do not release; do not extend |
| Base model MiniCheck-Flan-T5-Large | Model card: MIT. Fine-tuned by its authors on an ANLI subset (CC-BY-NC-4.0) plus synthetic GPT-4/GPT-3.5 data. Downstream terms of ANLI's non-commercial licence and of the model-generated synthetic data were not analysed | **Unresolved for any non-research use**; acceptable question only for a private research pilot | Legal review before any release or commercial use |
| Underlying Flan-T5 exposure to evaluation data | Not documented | **Unknown** | Cannot be resolved by an overlap audit |

Consequences:
- The pilot is a **private research experiment**. Its weights, manifests and outputs must not be released or used commercially.
- Running it without a decision on the CNN/DM row would rest on the user's own acceptance of that risk. That is the user's call to
  make, and it should be made explicitly and in writing in this file before any GPU run.
- If the decision is "not accepted" or "undecided", the answer is: **no permitted training recipe is established**, and the GPU
  pilot does not run.

Decision record (to be completed by the user; left blank deliberately):
- CNN/DM rows for a private research pilot: accept, date 2026-09-30

Recorded by the assistant at the user's explicit written instruction on 2026-09-30 ("I explicitly authorize you to record my existing
decision"). This is the user's private-research risk acceptance. It is NOT a rights clearance and does not cover release, redistribution or
commercial use; "Recent News" remains excluded.

## 2026-10-01 user instruction: broadened training authorization (recorded verbatim)
User: "No it is not negotiable. Do whatever it takes to get it right. Download whatever, train whatever. I don't care. I want that metric." and "we will change all the dataset and labels all you want it to excute this."
Interpretation recorded here (the user's decision, not a rights clearance): private-research use of the 70 TofuEval upstream-dev documents (3,437 labelled summary sentences, previously reserved evaluation-only; MIT-0 annotations whose README says not to train on them; transcripts CC-BY-NC-SA) as TRAINING data, with local downloads and local GPU/CPU training allowed within the 40 GPU-hour ceiling, API spend capped at $60 for this campaign unless the user raises it. Those documents are therefore no longer independent evaluation material.
NOT authorized and not done: changing or removing labels or cases of the original RouteBench benchmark, or training on / selecting with RouteBench dev/test labels beyond the already-defined TRAIN/dev development roles. The original held-out test stays sealed until the user separately authorizes one frozen run. Adjudicated or alternative labels, if ever made, live in a separately named benchmark version and are never reported as passing the original target.
