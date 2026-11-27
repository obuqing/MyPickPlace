"""Repository launcher for the push skill."""
import runpy
if __name__ == '__main__':
    runpy.run_module('panda_skills.push.run_trained_push', run_name='__main__')
