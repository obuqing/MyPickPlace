from copy import deepcopy
import unittest
from algorithms.correction_roles import bind_destination_only
from algorithms.intent_planning import compile_intent
from test_intent_planning import SCENE, intent


class CorrectionRoleTests(unittest.TestCase):
    def test_stale_destination_is_replaced_without_touching_original_object(self):
        for color in ('red','blue'):
            text = f'Move {color} to the purple target; correction: use the green goal instead.'
            extracted = intent(color,'the purple target')
            extracted['tasks'][0].update(object_evidence=color,destination_color='purple')
            previous = deepcopy(extracted)
            plan, checks = compile_intent(extracted,text,SCENE)
            self.assertEqual(plan['object_id'],'cube_'+color)
            check = next(c for c in checks if c.get('correction_scope')=='destination')
            self.assertEqual(check['inherited_object'],color)
            for p in check['source_positions']:
                self.assertEqual(text[p['start_char']:p['end_char']],p['text'])
            self.assertEqual(extracted,previous)

    def test_destination_color_role_error_keeps_original_object_source(self):
        text = 'Move blue to the purple target; correction: use the green goal instead.'
        extracted = intent('green','the green goal')
        extracted['tasks'][0].update(object_evidence='green')
        self.assertEqual(compile_intent(extracted,text,SCENE)[0]['object_id'],'cube_blue')

    def test_unknown_objects_or_targets_are_not_substituted(self):
        for color,destination in [('orange','green'),('red','violet')]:
            text=f'Move {color} to the purple target; correction: use the {destination} goal instead.'
            extracted=intent(color,'the purple target')
            extracted['tasks'][0].update(object_evidence=color)
            self.assertEqual(compile_intent(extracted,text,SCENE)[0]['status'],'reject')

    def test_extra_tasks_quantities_ambiguity_and_object_corrections_never_match(self):
        texts = [
            'Move red and blue to the purple target; correction: use the green goal instead.',
            'Move red to the purple target and move blue to the goal; correction: use the green goal instead.',
            'Move red to the purple target; correction: use the green goal and move blue too.',
            'Move red to the purple target; correction: move blue to the green goal.',
            'Move it to the purple target; correction: use the green goal instead.',
            'Move a cube to the purple target; correction: use the green goal instead.',
            'Move red to the purple target; correction: use blue instead.'
        ]
        for text in texts:
            extracted=intent('red','the purple target')
            normalized,check=bind_destination_only(extracted,text)
            if check is not None:
                # A literal unsupported reference such as "it"/"a" cannot become red.
                self.assertNotEqual(normalized['tasks'][0]['object_color'],'red')
            self.assertIsNone(check)

    def test_model_invented_unrelated_color_is_not_changed_into_supported_object(self):
        text='Move red to the purple target; correction: use the green goal instead.'
        extracted=intent('blue','the purple target')
        self.assertIsNone(bind_destination_only(extracted,text)[1])

    def test_destination_edit_preserves_latest_preceding_object_edit(self):
        text='Put blue on the green target. Actually, put red there. Correction: use the green disk instead.'
        extracted=intent('red','the green target')
        extracted['tasks'][0].update(object_evidence='red')
        normalized, check = bind_destination_only(extracted,text)
        self.assertEqual(normalized['tasks'][0]['object_color'],'red')
        self.assertEqual(normalized['tasks'][0]['destination_evidence'],'the green disk')
        position = next(p for p in check['source_positions'] if p['field']=='object_color')
        self.assertEqual(position['start_char'],text.index('red'))

    def test_relative_and_negated_goal_phrases_are_not_destination_bindings(self):
        for replacement in ('use next to green goal instead', 'select left of green goal instead',
                            'use not the green goal instead', 'choose the goal near blue instead'):
            text='Move red to the purple target. Correction: '+replacement+'.'
            extracted=intent('red','the purple target')
            self.assertIsNone(bind_destination_only(extracted,text)[1])

    def test_declared_relative_or_coordinate_tasks_are_not_changed_to_goal_tasks(self):
        text='Move red to the purple target. Correction: use the green goal instead.'
        for kind in ('relative','coordinates','other','unspecified'):
            extracted=intent('red','the purple target')
            extracted['tasks'][0]['destination_type']=kind
            self.assertIsNone(bind_destination_only(extracted,text)[1])
