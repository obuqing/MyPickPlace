"""Repository launcher for the stacking skill."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    runpy.run_module('skills.stack.explore_stack', run_name='__main__')
