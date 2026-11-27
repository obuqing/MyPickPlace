"""Repository launcher for the push skill."""
import runpy
if __name__ == '__main__':
    runpy.run_module('panda_skills.push.train_push', run_name='__main__')
