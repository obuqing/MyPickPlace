import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_carry_generalization import used_scene_seeds


class SeedAuditTests(unittest.TestCase):
    def test_usage_ledger_survives_removal_of_old_result_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'scene_history.json').write_text(json.dumps({'scene_seeds':[41,42]}))
            seeds, _ = used_scene_seeds(root)
            self.assertEqual(seeds,{41,42})
    def test_audit_includes_failed_attempts_and_actual_random_training_scenes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'initializations.jsonl').write_text(
                json.dumps(dict(seed=987654, success=False))+'\n'+json.dumps(dict(seed=12345, success=True))+'\n',
                encoding='utf-8')
            (root/'results.json').write_text(json.dumps([dict(seed=50000, placement_success=False),
                                                        dict(seed=12345, placement_success=True)]), encoding='utf-8')
            seeds, files = used_scene_seeds(root)
            self.assertEqual(seeds, {987654, 12345, 50000})
            self.assertEqual(len(files), 2)
            self.assertTrue(all(len(row['sha256']) == 64 for row in files))

    def test_unrelated_result_objects_do_not_hide_valid_scene_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'results.json').write_text(json.dumps([{}, dict(seed='instruction text'), dict(seed=42)]))
            seeds, _ = used_scene_seeds(root)
            self.assertEqual(seeds, {42})
