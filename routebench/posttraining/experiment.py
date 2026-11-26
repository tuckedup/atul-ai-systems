"""Small, auditable SFT experiment. ML imports are lazy for offline verification."""
import argparse
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dataset import LABELS, SYSTEM, load_data, sha

HERE = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def metrics(predictions):
    if not predictions:
        raise ValueError('Cannot score an empty evaluation')
    confusion = {a: {b: 0 for b in (*LABELS, 'INVALID')} for a in LABELS}
    for row in predictions:
        if row['label'] not in LABELS:
            raise ValueError('Unknown reference label')
        pred = row['prediction'].strip().lower()
        confusion[row['label']][pred if pred in LABELS else 'INVALID'] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = support - tp
        per_class[label] = dict(support=support, correct=tp,
                               f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return dict(count=len(predictions), correct=sum(confusion[c][c] for c in LABELS),
                accuracy=sum(confusion[c][c] for c in LABELS) / len(predictions),
                macro_f1=sum(c['f1'] for c in per_class.values()) / len(LABELS),
                invalid=sum(confusion[c]['INVALID'] for c in LABELS),
                per_class=per_class, confusion=confusion)

def paired(base, tuned):
    if [r['id'] for r in base] != [r['id'] for r in tuned]:
        raise ValueError('Paired evaluations must have the same ordered IDs')
    if [r['label'] for r in base] != [r['label'] for r in tuned]:
        raise ValueError('Paired evaluations must have the same references')
    result = {'fixed': [], 'regressed': [], 'both_wrong': []}
    for a, b in zip(base, tuned):
        ac = a['prediction'].strip().lower() == a['label']
        bc = b['prediction'].strip().lower() == b['label']
        if not ac and bc:
            result['fixed'].append(a['id'])
        elif ac and not bc:
            result['regressed'].append(a['id'])
        elif not ac and not bc:
            result['both_wrong'].append(a['id'])
    return result

def gate(base_metrics, tuned_metrics, changes):
    """Predeclared demonstration gate; failure remains a reported result."""
    return (tuned_metrics['accuracy'] >= 0.8
            and tuned_metrics['macro_f1'] > base_metrics['macro_f1']
            and len(changes['fixed']) > len(changes['regressed'])
            and tuned_metrics['invalid'] == 0)

def messages(text):
    return [dict(role='system', content=SYSTEM), dict(role='user', content=text)]

def training_example(tokenizer, row, max_length=384):
    prefix = tokenizer.apply_chat_template(messages(row['text']), tokenize=True, add_generation_prompt=True)
    answer = tokenizer.encode(row['label'], add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prefix) + len(answer) > max_length:
        raise ValueError(f"Example {row['id']} exceeds maximum length; refusing truncation")
    return dict(input_ids=prefix + answer, labels=[-100] * len(prefix) + answer)

def collate(examples, pad_id, device):
    import torch
    length = max(len(e['input_ids']) for e in examples)
    return {key: torch.tensor([e[key] + [fill] * (length - len(e[key])) for e in examples], device=device)
            for key, fill in [('input_ids', pad_id), ('labels', -100)]} | {
                'attention_mask': torch.tensor([[1] * len(e['input_ids']) + [0] * (length - len(e['input_ids']))
                                                for e in examples], device=device)}

def predict(model, tokenizer, rows, device, batch_size=4):
    import torch
    model.eval()
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    output = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                prompts = [tokenizer.apply_chat_template(messages(r['text']), tokenize=False,
                                                          add_generation_prompt=True) for r in batch]
                encoded = tokenizer(prompts, padding=True, return_tensors='pt').to(device)
                tokens = model.generate(**encoded, do_sample=False, max_new_tokens=8,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id)
                texts = tokenizer.batch_decode(tokens[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
                output.extend(dict(**row, prediction=text) for row, text in zip(batch, texts))
    finally:
        tokenizer.padding_side = old_padding
    return output

def load_model(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                                                torch_dtype=dtype, attn_implementation='sdpa').to(device)
    return model, tokenizer

def source_hashes():
    return {p.name: sha(p) for p in [HERE / 'experiment.py', HERE / 'dataset.py', HERE / 'Dockerfile']}

def run(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed
    if args.epochs < 1 or args.batch_size < 1 or args.accumulate < 1:
        raise ValueError('Epochs, batch size, and accumulation must be positive')
    data, pins = load_data(HERE / 'data')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # Never silently overwrite an experiment.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    torch.set_num_threads(4)
    start = time.perf_counter()
    config = dict(model=MODEL, revision=REVISION, seed=args.seed, epochs=args.epochs,
                  batch_size=args.batch_size, accumulate=args.accumulate, learning_rate=args.lr,
                  rank=8, alpha=16, targets=['q_proj', 'k_proj', 'v_proj', 'o_proj'], max_length=384,
                  selection='highest validation macro F1, earliest epoch on ties',
                  gate='test accuracy >= .8, macro F1 improves, fixes > regressions, zero invalid outputs',
                  data_sha256=pins, source_sha256=source_hashes(), system_prompt=SYSTEM,
                  device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                  versions={p: importlib.metadata.version(p) for p in
                            ['torch', 'transformers', 'peft', 'accelerate', 'huggingface-hub']},
                  started_utc=datetime.now(timezone.utc).isoformat())
    config['git_head'] = os.environ.get('EXPERIMENT_GIT_HEAD')
    if not config['git_head']:
        try:
            config['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE,
                                                         text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            config['git_head'] = None
    write_json(out / 'config.json', config)
    (out / 'source').mkdir()
    for name in config['source_sha256']:
        (out / 'source' / name).write_bytes((HERE / name).read_bytes())
    model, tokenizer = load_model(device)
    # Validation and test text never enters the optimization batches.
    examples = [training_example(tokenizer, row) for row in data['train']]
    base_dev = predict(model, tokenizer, data['dev'], device)
    write_json(out / 'base_dev.json', base_dev)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=config['targets'], task_type='CAUSAL_LM'))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if any('lora_' not in n for n, _ in trainable):
        raise ValueError('Expected only LoRA weights to be trainable')
    config['trainable_parameters'] = sum(p.numel() for _, p in trainable)
    config['total_parameters'] = sum(p.numel() for p in model.parameters())
    write_json(out / 'config.json', config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.01)
    history, best = [], -1.0
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(order)
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for i, indexes in enumerate(batches):
            batch = collate([examples[k] for k in indexes], tokenizer.pad_token_id, device)
            loss = model(**batch).loss
            if not math.isfinite(loss.item()):
                raise ValueError('Nonfinite training loss')
            losses.append(loss.item())
            # Correctly normalize a final, partial accumulation group.
            group_size = min(args.accumulate, len(batches) - (i // args.accumulate) * args.accumulate)
            (loss / group_size).backward()
            if (i + 1) % args.accumulate == 0 or i + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 30 == 0:
                print(f'epoch={epoch} microbatch={i+1}/{len(batches)} loss={loss.item():.4f}', flush=True)
        dev = predict(model, tokenizer, data['dev'], device)
        scores = metrics(dev)
        history.append(dict(epoch=epoch, mean_training_loss=sum(losses)/len(losses), validation=scores))
        write_json(out / 'history.json', history)
        write_json(out / f'dev_epoch_{epoch}.json', dev)
        print(f"epoch={epoch} dev_accuracy={scores['accuracy']:.3f} dev_f1={scores['macro_f1']:.3f}", flush=True)
        if scores['macro_f1'] > best:
            best = scores['macro_f1']
            config['selected_epoch'] = epoch
            model.save_pretrained(out / 'adapter', safe_serialization=True)
    config['training_seconds'] = time.perf_counter() - train_start
    # Reload the chosen checkpoint, rather than accidentally evaluating the last epoch.
    model = model.unload()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model = PeftModel.from_pretrained(model, out / 'adapter')
    config['adapter_sha256'] = sha(out / 'adapter' / 'adapter_model.safetensors')
    with model.disable_adapter():
        baseline = predict(model, tokenizer, data['test'], device)
    tuned = predict(model, tokenizer, data['test'], device)
    write_json(out / 'base_test.json', baseline)
    write_json(out / 'tuned_test.json', tuned)
    base_metrics, tuned_metrics = metrics(baseline), metrics(tuned)
    changes = paired(baseline, tuned)
    report = dict(baseline=base_metrics, tuned=tuned_metrics, paired=changes,
                  gate_passed=gate(base_metrics, tuned_metrics, changes),
                  selected_epoch=config['selected_epoch'])
    config['elapsed_seconds'] = time.perf_counter() - start
    config['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else None
    write_json(out / 'config.json', config)
    write_json(out / 'report.json', report)
    # Bind evidence to the actual inputs, predictions, and selected adapter.
    files = [p for p in out.rglob('*') if p.is_file()]
    write_json(out / 'manifest.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(files)})
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['gate_passed'] else 1

def verify(directory, require_gate=True):
    """Recompute scores and file hashes without installing torch or downloading a model."""
    out = Path(directory)
    manifest = json.loads((out / 'manifest.json').read_text())
    required = {'config.json', 'history.json', 'report.json', 'base_test.json', 'tuned_test.json',
                'adapter/adapter_model.safetensors', 'adapter/adapter_config.json',
                'source/experiment.py', 'source/dataset.py', 'source/Dockerfile'}
    if not required.issubset(manifest):
        raise ValueError('Evidence manifest is incomplete')
    for name, expected in manifest.items():
        candidate = (out / name).resolve()
        if not candidate.is_relative_to(out.resolve()) or sha(candidate) != expected:
            raise ValueError(f'Evidence hash mismatch: {name}')
    base = json.loads((out / 'base_test.json').read_text())
    tuned = json.loads((out / 'tuned_test.json').read_text())
    expected_rows, pins = load_data(HERE / 'data')
    config = json.loads((out / 'config.json').read_text())
    if config['data_sha256'] != pins or config['model'] != MODEL or config['revision'] != REVISION:
        raise ValueError('Evidence uses different model or data pins')
    for name, expected in config['source_sha256'].items():
        if sha(out / 'source' / name) != expected:
            raise ValueError('Training source snapshot hash mismatch')
    if sha(out / 'adapter' / 'adapter_model.safetensors') != config['adapter_sha256']:
        raise ValueError('Selected adapter hash mismatch')
    history = json.loads((out / 'history.json').read_text())
    if not history or len(history) != config['epochs']:
        raise ValueError('Training history is incomplete')
    for epoch in history:
        name = f"dev_epoch_{epoch['epoch']}.json"
        if name not in manifest:
            raise ValueError('Missing validation predictions')
        predictions = json.loads((out / name).read_text())
        if [{k: r[k] for k in ('id', 'group', 'text', 'label')} for r in predictions] != expected_rows['dev']:
            raise ValueError('Validation predictions do not match the frozen validation split')
        if metrics(predictions) != epoch['validation']:
            raise ValueError('Validation score does not match predictions')
    selected = max(history, key=lambda e: e['validation']['macro_f1'])['epoch']
    if selected != config['selected_epoch']:
        raise ValueError('Checkpoint was not selected by validation macro F1')
    for predictions in (base, tuned):
        if [{k: r[k] for k in ('id', 'group', 'text', 'label')} for r in predictions] != expected_rows['test']:
            raise ValueError('Predictions do not match the complete frozen test split')
    b, t, changes = metrics(base), metrics(tuned), paired(base, tuned)
    report = json.loads((out / 'report.json').read_text())
    if report['selected_epoch'] != selected:
        raise ValueError('Report selects a different checkpoint')
    if (report['baseline'], report['tuned'], report['paired'], report['gate_passed']) != (b, t, changes, gate(b, t, changes)):
        raise ValueError('Report does not match predictions')
    print(f"Verified {len(manifest)} hashes; baseline {b['correct']}/{b['count']}; "
          f"tuned {t['correct']}/{t['count']}; gate={report['gate_passed']}")
    return 0 if not require_gate or report['gate_passed'] else 1

def demo(args):
    import torch
    from peft import PeftModel
    directory = Path(args.run)
    config = json.loads((directory / 'config.json').read_text())
    if config['model'] != MODEL or config['revision'] != REVISION:
        raise ValueError('Run uses a different base model')
    if sha(directory / 'adapter' / 'adapter_model.safetensors') != config['adapter_sha256']:
        raise ValueError('Adapter hash mismatch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model, tokenizer = load_model(device)
    model = PeftModel.from_pretrained(model, directory / 'adapter')
    row = dict(id='demo', text=args.text)
    with model.disable_adapter():
        base = predict(model, tokenizer, [row], device)[0]['prediction']
    tuned = predict(model, tokenizer, [row], device)[0]['prediction']
    print(json.dumps(dict(text=args.text, baseline=base, tuned=tuned), indent=2))
    return 0

def reevaluate(args):
    """Reload an untouched base in a fresh process and replay both conditions."""
    import torch
    from peft import PeftModel
    directory = Path(args.run)
    config = json.loads((directory / 'config.json').read_text())
    if config['model'] != MODEL or config['revision'] != REVISION:
        raise ValueError('Run uses a different base model')
    if sha(directory / 'adapter' / 'adapter_model.safetensors') != config['adapter_sha256']:
        raise ValueError('Adapter hash mismatch')
    destination = Path(args.output)
    if destination.exists():
        raise ValueError('Refusing to overwrite a reload check')
    data, pins = load_data(HERE / 'data')
    if pins != config['data_sha256']:
        raise ValueError('Run uses different data')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model, tokenizer = load_model(device)
    # Evaluate the pristine model BEFORE attaching any LoRA module.
    base = predict(model, tokenizer, data['test'], device)
    model = PeftModel.from_pretrained(model, directory / 'adapter')
    tuned = predict(model, tokenizer, data['test'], device)
    result = dict(model=MODEL, revision=REVISION, adapter_sha256=config['adapter_sha256'],
                  base_predictions_match=base == json.loads((directory / 'base_test.json').read_text()),
                  tuned_predictions_match=tuned == json.loads((directory / 'tuned_test.json').read_text()),
                  baseline=metrics(base), tuned=metrics(tuned))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    print(json.dumps({k: result[k] for k in ('base_predictions_match', 'tuned_predictions_match')}))
    return 0 if result['base_predictions_match'] and result['tuned_predictions_match'] else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('run')
    train.add_argument('--output', default='runs/seed42')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--epochs', type=int, default=4)
    train.add_argument('--batch-size', type=int, default=2)
    train.add_argument('--accumulate', type=int, default=4)
    train.add_argument('--lr', type=float, default=3e-4)
    check = sub.add_parser('verify')
    check.add_argument('directory')
    check.add_argument('--integrity-only', action='store_true',
                       help='Verify recorded evidence even when its promotion gate failed')
    predict_parser = sub.add_parser('demo')
    predict_parser.add_argument('--run', default='runs/seed42')
    predict_parser.add_argument('--text', required=True)
    replay = sub.add_parser('reevaluate')
    replay.add_argument('--run', required=True)
    replay.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        return run(args)
    if args.command == 'verify':
        return verify(args.directory, require_gate=not args.integrity_only)
    if args.command == 'reevaluate':
        return reevaluate(args)
    return demo(args)

if __name__ == '__main__':
    raise SystemExit(main())
