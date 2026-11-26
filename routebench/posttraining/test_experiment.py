"""Offline tests of the experimental contract; no model downloads or ML packages."""
import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dataset import LABELS, load_data, sha, write_data
from experiment import MODEL, REVISION, gate, metrics, paired, training_example, verify, write_json

HERE = Path(__file__).parent

class ExperimentTests(unittest.TestCase):
    def test_dataset_pins_reproduce_and_no_split_overlap(self):
        data, pins = load_data(HERE / 'data')
        self.assertEqual([len(data[s]) for s in ('train', 'dev', 'test')], [240, 24, 40])
        with tempfile.TemporaryDirectory() as temporary:
            write_data(temporary)
            _, generated_pins = load_data(temporary)
            self.assertEqual(pins, generated_pins)
            file = Path(temporary) / 'test.jsonl'
            file.write_text(file.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_data(temporary)

    def test_rejects_leakage_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_data(temporary)
            root = Path(temporary)
            train = json.loads((root / 'train.jsonl').read_text().splitlines()[0])
            path = root / 'test.jsonl'
            test = [json.loads(line) for line in path.read_text().splitlines()]
            test[0]['text'] = train['text']
            path.write_text(''.join(json.dumps(r) + '\n' for r in test))
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['test.jsonl'] = sha(path)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                load_data(root)

    def test_taxonomy_matches_actual_router(self):
        module = ast.parse((HERE.parent / 'gateway' / 'router.py').read_text())
        assignment = next(n for n in module.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'TaskClass' for t in n.targets))
        self.assertEqual(tuple(ast.literal_eval(assignment.value.slice)), LABELS)

    def test_invalid_answer_is_not_fuzzy_matched(self):
        rows = [dict(id='a', label='sql', prediction='sql because it is a query'),
                dict(id='b', label='sql', prediction=' SQL '),
                dict(id='c', label='code', prediction='sql')]
        result = metrics(rows)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['invalid'], 1)
        self.assertEqual(result['per_class']['sql']['f1'], 0.5)
        self.assertAlmostEqual(result['macro_f1'], 0.5 / 8)

    def test_regressions_and_gate(self):
        base = [dict(id=str(i), label=c, prediction=c) for i, c in enumerate(LABELS)]
        tuned = [dict(r) for r in base]
        base[0]['prediction'] = 'chat'
        changes = paired(base, tuned)
        self.assertEqual(changes['fixed'], ['0'])
        self.assertTrue(gate(metrics(base), metrics(tuned), changes))
        tuned[1]['prediction'] = 'chat'
        changes = paired(base, tuned)
        self.assertEqual(changes['regressed'], ['1'])
        self.assertFalse(gate(metrics(base), metrics(tuned), changes))
        with self.assertRaises(ValueError):
            paired(base, list(reversed(tuned)))

    def test_only_completion_and_eos_are_supervised(self):
        class Tokenizer:
            eos_token_id = 99
            def apply_chat_template(self, *args, **kwargs):
                return [10, 20, 30]
            def encode(self, *args, **kwargs):
                return [40, 50]
        row = dict(id='example', label='tool_use', text='Create an event')
        example = training_example(Tokenizer(), row)
        self.assertEqual(example['input_ids'], [10, 20, 30, 40, 50, 99])
        self.assertEqual(example['labels'], [-100, -100, -100, 40, 50, 99])
        with self.assertRaisesRegex(ValueError, 'refusing truncation'):
            training_example(Tokenizer(), row, max_length=4)

    def test_verifier_rejects_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_json(Path(temporary) / 'manifest.json', {})
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                verify(temporary)

    def test_verifier_detects_tampering_and_recomputes_scores(self):
        # A deliberately fabricated unit-test fixture, never reported as model evidence.
        data, pins = load_data(HERE / 'data')
        base = [dict(r, prediction=r['label']) for r in data['test']]
        base[0]['prediction'] = 'invalid fixture answer'
        tuned = [dict(r, prediction=r['label']) for r in data['test']]
        dev = [dict(r, prediction=r['label']) for r in data['dev']]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'source').mkdir()
            sources = {}
            for name in ('experiment.py', 'dataset.py', 'Dockerfile'):
                (root / 'source' / name).write_bytes((HERE / name).read_bytes())
                sources[name] = sha(root / 'source' / name)
            (root / 'adapter').mkdir()
            weights = root / 'adapter' / 'adapter_model.safetensors'
            weights.write_bytes(b'UNIT TEST FIXTURE; NOT MODEL WEIGHTS')
            write_json(root / 'adapter' / 'adapter_config.json', {})
            write_json(root / 'config.json', dict(model=MODEL, revision=REVISION, data_sha256=pins,
                       source_sha256=sources, adapter_sha256=sha(weights), epochs=1, selected_epoch=1))
            write_json(root / 'history.json', [dict(epoch=1, validation=metrics(dev))])
            write_json(root / 'dev_epoch_1.json', dev)
            write_json(root / 'base_test.json', base)
            write_json(root / 'tuned_test.json', tuned)
            report = dict(baseline=metrics(base), tuned=metrics(tuned), paired=paired(base, tuned),
                          gate_passed=True, selected_epoch=1)
            write_json(root / 'report.json', report)
            manifest = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
            write_json(root / 'manifest.json', manifest)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify(root), 0)
            report['baseline']['correct'] = 40
            write_json(root / 'report.json', report)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify(root)
            # Even a rehashed but incorrectly scored report is rejected.
            manifest['report.json'] = sha(root / 'report.json')
            write_json(root / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'does not match predictions'):
                verify(root)
            tuned[0]['prediction'] = 'invalid fixture answer'
            write_json(root / 'tuned_test.json', tuned)
            report = dict(baseline=metrics(base), tuned=metrics(tuned), paired=paired(base, tuned),
                          gate_passed=False, selected_epoch=1)
            write_json(root / 'report.json', report)
            for name in ('tuned_test.json', 'report.json'):
                manifest[name] = sha(root / name)
            write_json(root / 'manifest.json', manifest)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify(root), 1)
                self.assertEqual(verify(root, require_gate=False), 0)

    def test_malformed_output_blocks_promotion_despite_accuracy(self):
        # The real record is checked by CI. Here, assert the gate itself remains
        # strict when an otherwise strong candidate has a malformed output.
        base = [dict(id=str(i), label=c, prediction='wrong') for i, c in enumerate(LABELS)]
        tuned = [dict(id=str(i), label=c, prediction=c) for i, c in enumerate(LABELS)]
        tuned[0]['prediction'] = 'classify plus commentary'
        self.assertGreaterEqual(metrics(tuned)['accuracy'], .8)
        self.assertFalse(gate(metrics(base), metrics(tuned), paired(base, tuned)))

if __name__ == '__main__':
    unittest.main()
