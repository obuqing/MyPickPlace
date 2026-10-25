from copy import deepcopy
import unittest
from algorithms.correction_checks import bind_role_correction, independent_task_evidence
from algorithms.intent_planning import compile_intent
from test_intent_planning import SCENE,intent


class CorrectionCheckTests(unittest.TestCase):
    def test_object_predicate_inherits_original_destination(self):
        text='Put blue on the green marker. Wait, red should go there.'
        extracted=intent('blue','the green marker')
        extracted['tasks'][0].update(action_evidence='Put',object_evidence='blue')
        normalized,audit=bind_role_correction(extracted,text)
        self.assertEqual(normalized['tasks'][0]['object_color'],'red')
        self.assertEqual(normalized['tasks'][0]['destination_evidence'],'the green marker')
        self.assertEqual(audit['correction_scope'],'object')
        self.assertEqual(compile_intent(extracted,text,SCENE)[0]['object_id'],'cube_red')

    def test_complete_correction_cancels_old_task_but_not_independent_moves(self):
        text='Take red to the purple target. Scratch that, carry blue onto the green disk.'
        extracted=intent('red','the purple target')
        extracted['tasks'][0].update(action_evidence='Take',object_evidence='red',destination_color='purple')
        new=intent('blue','the green disk')['tasks'][0]
        new.update(action_evidence='carry',object_evidence='blue')
        extracted['tasks'].append(new)
        normalized,audit=bind_role_correction(extracted,text)
        self.assertEqual(len(normalized['tasks']),1)
        self.assertEqual(normalized['tasks'][0]['object_color'],'blue')
        self.assertEqual(audit['correction_scope'],'both')
        self.assertEqual(audit['cancelled_tasks'],1)
        self.assertEqual(compile_intent(extracted,text,SCENE)[0]['object_id'],'cube_blue')
        independent=text.replace('Scratch that,','Then')
        self.assertIsNone(bind_role_correction(extracted,independent)[1])
        self.assertTrue(independent_task_evidence(independent))

    def test_source_extra_move_is_detected_even_when_model_omits_it(self):
        text='First place red in the green area and then move blue there.'
        self.assertEqual(len(independent_task_evidence(text)),2)
        extracted=intent('red','the green area')
        extracted['tasks'][0].update(action_evidence='place',object_evidence='red')
        self.assertEqual(compile_intent(extracted,text,SCENE)[0]['status'],'reject')

    def test_negative_actions_and_same_object_pick_and_place_are_not_extra_moves(self):
        texts=['Do not move red; put blue on the green goal.',
               'Take red and place it in the green goal.',
               'Leave blue alone and carry red to the green goal.',
               'Move red to the green goal. Actually, move blue there.']
        for text in texts:
            self.assertFalse(independent_task_evidence(text))

    def test_additional_move_after_correction_is_never_cancelled(self):
        text='Take red to the purple target. Scratch that, carry blue to the green disk and move red there.'
        self.assertTrue(independent_task_evidence(text))
        extracted=intent('blue','the green disk')
        self.assertIsNone(bind_role_correction(extracted,text)[1])

    def test_unknown_object_replacement_is_preserved_without_scene_substitution(self):
        text='Put blue on the green marker. Wait, orange should go there.'
        normalized,audit=bind_role_correction(intent('blue','the green marker'),text)
        self.assertEqual(normalized['tasks'][0]['object_color'],'orange')
        self.assertEqual(audit['correction_scope'],'object')
        self.assertEqual(compile_intent(intent('blue','the green marker'),text,SCENE)[0]['status'],'reject')

    def test_stay_put_is_stationary_while_actual_second_moves_stay_visible(self):
        for color,other in [('red','blue'),('blue','red')]:
            text=f'The {other} cube must stay put while you relocate the {color} cube to the green circle.'
            self.assertFalse(independent_task_evidence(text))
            extracted=intent(color,'the green circle')
            extracted['tasks'][0].update(action_evidence='relocate',object_evidence=f'the {color} cube')
            self.assertEqual(compile_intent(extracted,text,SCENE)[0]['object_id'],'cube_'+color)
        self.assertTrue(independent_task_evidence('Stay put. Move red to the green goal and move blue there.'))
