"""Repository launcher for the stacking skill."""
import runpy
if __name__ == '__main__':
    runpy.run_module('panda_skills.stack.train_stack', run_name='__main__')
