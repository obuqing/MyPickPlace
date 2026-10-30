import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.task_routing import select_skill, RoutingError
from scripts.run_instruction import main
from skills.push.push_language import plan


class RoutingTests(unittest.TestCase):
    def test_legacy_default(self):
        self.assertEqual(select_skill('Put the blue cube on the red goal.'), 'pick_place')

    def test_push_both_languages(self):
        for instruction in ('Push the blue cube forward 10 cm.', '把蓝色方块向前推10厘米。'):
            self.assertEqual(select_skill(instruction), 'push')

    def test_mixed_and_conflicting_requests(self):
        for instruction, requested, offline in (
            ('Push then pick the cube', 'auto', False),
            ('Push the cube', 'pick_place', False),
            ('Push the cube', 'auto', True),
        ):
            with self.assertRaises(RoutingError):
                select_skill(instruction, requested, offline)

    def test_mixed_cli_has_no_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'mixed'
            self.assertEqual(main(['--instruction', 'Push then pick the cube', '--output', str(out)]), 2)
            log = json.loads((out/'run.json').read_text())
            self.assertEqual(log['status'], 'rejected')
            self.assertEqual(log['control_steps'], 0)
            self.assertIsNone(log['physical_success'])

    def test_push_arguments_and_common_log(self):
        def fake_execute(instruction, output, checkpoint, seed, plan_only, **kwargs):
            self.assertEqual(kwargs, dict(provider='ollama', llm_model='test-model', base_url='http://localhost:1234'))
            self.assertTrue(plan_only)
            output.mkdir()
            (output/'llm_exchange.json').write_text('{}')
            return dict(plan=dict(status='execute', reason='test'), control_steps=0,
                        planning_status='execute', execution_status='plan_only')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'push'
            with patch('skills.push.push_instruction.execute', side_effect=fake_execute):
                code = main(['--instruction', 'Push the cube forward 10 cm.', '--plan-only',
                             '--provider', 'ollama', '--model', 'test-model',
                             '--base-url', 'http://localhost:1234', '--output', str(out)])
            self.assertEqual(code, 0)
            log = json.loads((out/'run.json').read_text())
            self.assertEqual((log['skill'], log['executor'], log['status']), ('push', 'rl', 'planned'))
            self.assertEqual(log['schema_version'], 1)
            self.assertEqual(log['control_steps'], 0)

    def test_unsupported_provider_does_not_make_request(self):
        with patch('skills.push.push_language.urlopen') as request:
            result, audit = plan('Push the cube forward 10 cm.', provider='openai')
        request.assert_not_called()
        self.assertEqual(result['status'], 'error')
        self.assertEqual(audit['attempts'], [])
