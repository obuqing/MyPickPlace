"""Position, correction-scope and polarity checks independent of model inference."""

from copy import deepcopy
import unittest

from algorithms.intent_planning import compile_intent, IntentFormatError
from algorithms.source_spans import source_tokens, correction_candidates
from test_intent_planning import SCENE, intent
from span_fixtures import spanned, referenced


class SourceSpanTests(unittest.TestCase):
    def test_reference_ids_cannot_be_invented_or_reused_from_cancelled_position(self):
        text = "Move red to the goal. Actually, use red instead."
        extracted = intent("red", "the goal")
        extracted["tasks"][0].update(object_evidence="red", destination_color=None)
        payload = referenced(extracted, text)
        self.assertEqual(compile_intent(payload, text, SCENE)[0]["status"], "execute")
        payload["tasks"][0]["color_ref"] = "t1:red"
        self.assertEqual(compile_intent(payload, text, SCENE)[0]["status"], "reject")
        payload["tasks"][0]["color_ref"] = "t999:red"
        with self.assertRaises(IntentFormatError):
            compile_intent(payload, text, SCENE)

    def test_two_positive_replacements_remain_multiple_tasks(self):
        text = "Wait, move red to the green goal and move blue to the green goal."
        red = intent("red", "the green goal")["tasks"][0]
        blue = intent("blue", "the green goal")["tasks"][0]
        red.update(action_evidence="move", object_evidence="red")
        blue.update(action_evidence="move", object_evidence="blue")
        payload = referenced({"tasks": [red, blue], "correction_evidence": "", "reason": "Fixture"}, text)
        self.assertEqual(compile_intent(payload, text, SCENE)[0]["status"], "reject")

    def test_source_negative_task_is_removed_without_discarding_positive_task(self):
        text = "Leave blue alone and carry red to the green goal."
        negative = intent("blue", "the green goal")["tasks"][0]
        negative.update(action_evidence="Leave", object_evidence="blue")
        positive = intent("red", "the green goal")["tasks"][0]
        positive.update(action_evidence="carry", object_evidence="red", action="pick_up")
        extracted = {"tasks": [negative, positive], "correction_evidence": "", "reason": "Fixture"}
        plan, checks = compile_intent(referenced(extracted, text), text, SCENE)
        self.assertEqual(plan["object_id"], "cube_red")
        self.assertTrue(any(p["field"] == "excluded_negative_task" for p in checks[0]["source_positions"]))

    def test_pickup_and_place_inherits_explicit_placement_verb(self):
        text = "Pick up red cube and place it in the green goal."
        extracted = intent("red", "the green goal")
        extracted["tasks"][0].update(action="pick_up", action_evidence="Pick")
        self.assertEqual(compile_intent(referenced(extracted, text), text, SCENE)[0]["status"], "execute")
        text = "Pick up red cube but do not place it in the green goal."
        self.assertEqual(compile_intent(referenced(extracted, text), text, SCENE)[0]["status"], "reject")
        text = "Pick up red cube then move turquoise to the green goal."
        self.assertEqual(compile_intent(referenced(extracted, text), text, SCENE)[0]["status"], "reject")

    def test_cancelled_task_removed_only_with_explicit_source_correction(self):
        text = "Move red to the green goal. Actually, use blue instead."
        old = intent("red", "the green goal")["tasks"][0]
        old.update(object_evidence="red")
        new = intent("blue", "the green goal")["tasks"][0]
        new.update(action_evidence="use", object_evidence="blue")
        extracted = {"tasks": [old, new], "correction_evidence": "", "reason": "Fixture"}
        self.assertEqual(compile_intent(referenced(extracted, text), text, SCENE)[0]["object_id"], "cube_blue")
        text = "Move red to the green goal and use blue for the green goal."
        self.assertEqual(compile_intent(referenced(extracted, text), text, SCENE)[0]["status"], "reject")

    def test_goal_reference_expands_without_dropping_unknown_qualifier(self):
        text = "Move red cube to the silver target."
        extracted = intent("red", "the silver target")
        extracted["tasks"][0]["destination_color"] = None
        payload = referenced(extracted, text)
        target = next(t for t in source_tokens(text) if t["text"] == "target")
        payload["tasks"][0]["destination_ref"] = f"t{target['index']}:target"
        self.assertEqual(compile_intent(payload, text, SCENE)[0]["status"], "reject")

    def test_relative_role_normalization_does_not_allow_next_to_goal(self):
        text = "Move red cube into the green goal."
        extracted = intent("red", "the green goal")
        extracted["tasks"][0]["destination_type"] = "relative"
        self.assertEqual(compile_intent(referenced(extracted, text), text, SCENE)[0]["status"], "execute")
        text = "Move red cube next to the green goal."
        self.assertEqual(compile_intent(referenced(extracted, text), text, SCENE)[0]["status"], "reject")

    def test_color_binding_can_be_separate_from_noun_phrase(self):
        text = "The cube I want moved to the green goal is blue, not red."
        extracted = intent("blue")
        extracted["tasks"][0].update(action_evidence="moved", object_evidence="The cube", destination_evidence="the green goal")
        plan, checks = compile_intent(spanned(extracted, text), text, SCENE, require_spans=True)
        self.assertEqual(plan["object_id"], "cube_blue")
        for position in checks[0]["source_positions"]:
            self.assertEqual(text[position["start_char"]:position["end_char"]], position["text"])

    def test_excluded_color_cannot_be_bound_as_active(self):
        text = "The cube I want moved to the green goal is blue, not red."
        extracted = intent("red")
        extracted["tasks"][0].update(action_evidence="moved", object_evidence="The cube", destination_evidence="the green goal")
        plan, _ = compile_intent(spanned(extracted, text), text, SCENE)
        self.assertEqual(plan["status"], "reject")

    def test_partial_cue_does_not_determine_replacement_scope(self):
        text = "I was about to ask for blue, but instead please carry the red block to the goal."
        extracted = intent("red", "the goal")
        extracted["tasks"][0].update(action_evidence="carry", object_evidence="the red block", destination_color=None)
        payload = spanned(extracted, text)
        plan, _ = compile_intent(payload, text, SCENE)
        self.assertEqual(plan["object_id"], "cube_red")
        payload["correction"]["replacement_span"]["end"] = payload["correction"]["replacement_span"]["start"]
        plan, checks = compile_intent(payload, text, SCENE)
        self.assertEqual(plan["status"], "reject")
        self.assertTrue(checks[-1]["repairable"])

    def test_identical_color_at_old_position_is_not_valid_for_correction(self):
        text = "Move red to the goal. Actually, use red instead."
        extracted = intent("red", "the goal")
        extracted["tasks"][0].update(object_evidence="red", destination_color=None)
        payload = spanned(extracted, text)
        plan, _ = compile_intent(payload, text, SCENE)
        self.assertEqual(plan["status"], "execute")
        old_index = next(t["index"] for t in source_tokens(text) if t["text"] == "red")
        payload["tasks"][0]["color_span"] = {"start": old_index, "end": old_index + 1}
        plan, _ = compile_intent(payload, text, SCENE)
        self.assertEqual(plan["status"], "reject")

    def test_destination_only_scope_inherits_object(self):
        text = "Move red cube to the blue target. Actually, use the green goal."
        extracted = intent("red", "the green goal")
        payload = spanned(extracted, text, "destination")
        plan, _ = compile_intent(payload, text, SCENE)
        self.assertEqual(plan["object_id"], "cube_red")
        payload["correction"]["scope"] = "object"
        self.assertEqual(compile_intent(payload, text, SCENE)[0]["status"], "reject")

    def test_explicit_destination_scope_preserves_original_object_positions(self):
        for color in ('red', 'blue'):
            text = f'Move {color} to the purple target; correction: use the green goal instead.'
            extracted = intent(color, 'the green goal')
            extracted['tasks'][0].update(object_evidence=color)
            payload = referenced(extracted,text)
            self.assertEqual(payload['correction_scope'], 'destination')
            plan, checks = compile_intent(payload,text,SCENE)
            self.assertEqual(plan['object_id'], 'cube_'+color)
            self.assertEqual(checks[0]['correction_scope'], 'destination')
            position = next(p for p in checks[0]['source_positions'] if p['field']=='tasks[0].object_color')
            self.assertEqual(position['text'], color)
            self.assertLess(position['start_char'], text.index('correction'))

    def test_goal_color_cannot_supply_moved_object_color_even_with_both_scope(self):
        text = 'Move red to the purple target; correction: use the green goal instead.'
        extracted = intent('green','the green goal')
        extracted['tasks'][0].update(action_evidence='use',object_evidence='green')
        payload = referenced(extracted,text)
        self.assertEqual(payload['correction_scope'],'both')
        plan, checks = compile_intent(payload,text,SCENE)
        self.assertEqual(plan['status'],'reject')
        self.assertIn('destination phrase',plan['reason'])

    def test_scope_label_cannot_override_grounded_role_positions(self):
        text = 'Move red to the green goal. Actually, use blue instead.'
        extracted = intent('blue','the green goal')
        extracted['tasks'][0].update(action_evidence='use',object_evidence='blue')
        payload = referenced(extracted,text)
        self.assertEqual(payload['correction_scope'],'object')
        payload['correction_scope'] = 'destination'
        plan, checks = compile_intent(payload,text,SCENE)
        self.assertEqual(plan['object_id'],'cube_blue')
        self.assertEqual(checks[0]['correction_scope'],'object')
        event = next(p for p in checks[0]['source_positions'] if p['field']=='resolved_correction_scope')
        self.assertEqual(event['declared_scope'],'destination')

    def test_inherited_unknown_object_and_corrected_unknown_destination_stay_rejected(self):
        for color, destination in [('orange','green'),('red','silver')]:
            text = f'Move {color} to the purple target; correction: use the {destination} goal instead.'
            extracted = intent(color,'the '+destination+' goal')
            extracted['tasks'][0].update(object_evidence=color,destination_color=destination)
            self.assertEqual(compile_intent(referenced(extracted,text),text,SCENE)[0]['status'],'reject')

    def test_correction_both_uses_distinct_object_and_destination_roles(self):
        text = 'Move red to the purple target. Actually, move blue to the green goal.'
        extracted = intent('blue','the green goal')
        extracted['tasks'][0].update(action_evidence='move',object_evidence='blue')
        payload = referenced(extracted,text)
        self.assertEqual(payload['correction_scope'],'both')
        self.assertEqual(compile_intent(payload,text,SCENE)[0]['object_id'],'cube_blue')

    def test_bad_bounds_and_color_disagreement_are_rejected(self):
        text = "Move red cube to green goal."
        original = spanned(intent(), text)
        for start, end in ((-1, 2), (0, 999), (3, 1), (0, 0), (0, 1)):
            payload = deepcopy(original)
            payload["tasks"][0]["color_span"] = {"start": start, "end": end}
            self.assertEqual(compile_intent(payload, text, SCENE)[0]["status"], "reject")

    def test_nested_correction_uses_latest_cue_not_old_color(self):
        text = "Use blue for the green target. Correction: I meant red."
        candidates = correction_candidates(text)
        self.assertEqual(len(candidates), 2)
        extracted = intent("red", "the green target")
        extracted["tasks"][0].update(action_evidence="Use", object_evidence="red")
        payload = spanned(extracted, text)
        self.assertEqual(compile_intent(payload, text, SCENE)[0]["object_id"], "cube_red")
        payload["correction"].update(candidates[0])
        self.assertEqual(compile_intent(payload, text, SCENE)[0]["status"], "reject")


if __name__ == "__main__":
    unittest.main()
