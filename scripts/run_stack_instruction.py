"""Repository launcher for the stacking skill."""
import runpy
if __name__ == '__main__':
    runpy.run_module('panda_skills.stack.run_stack_instruction', run_name='__main__')
