import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from skills.push.push_language import validate,source_references,compile_references
from skills.push.push_instruction import execute


def intent(**changes):
    result=dict(status='execute',object_id='cube_blue',direction='+X',distance_value=10,unit='cm',
        action_text='Push',object_text='the blue cube',direction_text='+X',distance_text='10 cm',reason='one push')
    result.update(changes); return result


class LanguageTests(unittest.TestCase):
    instruction='Push the blue cube 10 cm along +X.'

    def test_supported(self):
        self.assertEqual(validate(self.instruction,intent())['distance_m'],.1)

    def test_chinese_without_spaces(self):
        p=intent(action_text='推',object_text='蓝色方块',direction_text='向前',distance_text='10厘米')
        self.assertEqual(validate('把蓝色方块向前推10厘米。',p)['distance_m'],.1)

    def test_unit_conversion(self):
        p=intent(distance_value=100,unit='mm',distance_text='100 mm')
        self.assertEqual(validate(self.instruction.replace('10 cm','100 mm'),p)['distance_m'],.1)

    def test_fabricated_and_changed_distance(self):
        for p in (intent(distance_value=15),intent(distance_text='15 cm'),intent(unit='m'),intent(distance_value=float('nan'))):
            self.assertEqual(validate(self.instruction,p)['status'],'reject')

    def test_negation_and_multiple_actions(self):
        for s in ('Do not '+self.instruction,'不要'+self.instruction,self.instruction+' Then push it again.'):
            self.assertEqual(validate(s,intent())['status'],'reject')

    def test_unknown_color_and_out_of_range(self):
        p=intent(object_text='the red cube')
        self.assertEqual(validate(self.instruction.replace('blue','red'),p)['status'],'reject')
        for value in (-10,0,20):
            p=intent(distance_value=value,distance_text=f'{value} cm')
            self.assertEqual(validate(self.instruction.replace('10 cm',f'{value} cm'),p)['status'],'reject')

    def test_missing_direction_evidence(self):
        self.assertEqual(validate(self.instruction,intent(direction_text='forward'))['status'],'reject')

    def test_rejection_never_enters_simulation(self):
        with TemporaryDirectory() as directory:
            with patch('skills.push.push_instruction.plan',return_value=({'status':'reject','reason':'unsupported'}, {'attempts':[{}]})):
                result=execute('unsupported',Path(directory)/'rejected',Path(directory)/'model-does-not-exist.zip')
            self.assertFalse(result['scene_created'])
            self.assertEqual(result['control_steps'],0)
            self.assertFalse(result['fallback'])

    def test_reference_conversion_uses_source_not_model_numbers(self):
        text='Push the cube forward 100 mm.'
        refs=source_references(text)
        extracted=dict(status='execute',action_ref=0,object_ref=0,direction_ref=0,distance_ref=0,reason='one push')
        self.assertEqual(compile_references(text,extracted,refs)['distance_m'],.1)

    def test_invalid_reference_cannot_select_another_word(self):
        refs=source_references(self.instruction)
        extracted=dict(status='execute',action_ref=99,object_ref=0,direction_ref=0,distance_ref=0,reason='one push')
        self.assertEqual(compile_references(self.instruction,extracted,refs)['status'],'reject')

    def test_extra_source_request_is_not_silently_dropped(self):
        text=self.instruction+'; open the gripper'
        self.assertEqual(validate(text,intent())['status'],'reject')


if __name__=='__main__': unittest.main()
