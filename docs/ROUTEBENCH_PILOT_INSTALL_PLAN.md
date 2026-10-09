# Pinned, isolated installation plan for the pilot (PROPOSAL; nothing has been installed)

Resolver check only (`uv pip install --dry-run peft` against the existing isolated MiniCheck environment): 38 packages resolved,
**one** new package would be installed, `peft==0.21.1`; nothing would be upgraded. No dependency was installed or changed in this pass.

Existing pins in that environment (the scoring path already ran on them for MiniCheck inference):
`torch==2.6.0+cu124`, `transformers==5.17.0`, `tokenizers==0.23.2`, `accelerate==1.15.0`, `safetensors==0.8.0`, `huggingface-hub==1.33.0`,
`numpy==2.5.3`, `packaging==26.3`, `psutil==7.2.2`, `pyyaml==6.0.3`, `tqdm==4.70.1`.

Proposed steps (to run only after approval; each is reversible by deleting the directory):
1. Do not touch the project environment. Create a copy for the pilot: `uv venv .local/pilot-venv --python 3.12`, then
   `uv pip install --python .local/pilot-venv/Scripts/python.exe --offline <the pins above>` so torch and transformers come from
   the local wheel cache with no network access.
2. `uv pip install --python .local/pilot-venv/Scripts/python.exe "peft==0.21.1"` (the single network fetch, one small wheel).
3. Record `uv pip freeze` output next to the run result. Run everything with `HF_HUB_OFFLINE=1`; the model directory must be the
   already-present snapshot `96eafd01cee2d16cf81aaa2fb226b14f422a37b3` (the backend refuses any other directory name).
4. If `peft` fails to import or behaves differently, stop; do not try other versions during the pilot's time budget.

Not covered: the real backend (`evalops/pilot/torch_backend.py`) has not been run on hardware. Its first execution is the timed
feasibility probe inside the 7,200 s cap. Untested risks: bf16 LoRA parameter dtypes and optimiser behaviour, gradient
checkpointing with the T5 encoder, peak memory at micro-batch 2 with inputs up to 2,048 tokens on the 8 GB card. Any of these can
end the run as NOT_FEASIBLE or INCOMPLETE, which is a valid outcome.

## Running the pilot (after approval and a recorded permission decision)
`python -m evalops.pilot.cli execute .local/ragtruth-ingest-20260930 .local/token_lens.json <pinned model snapshot dir> --python .local/pilot-venv/Scripts/python.exe --confirm RUN-GPU-PILOT`
(run from the repository root with `PYTHONPATH=routebench`). The command supervises the worker under an external watchdog and persists a result even if the worker is killed. It refuses to start without the recorded decision.
