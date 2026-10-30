"""Repository launcher for the push skill."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    runpy.run_module('skills.push.train_push', run_name='__main__')
