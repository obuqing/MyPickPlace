"""Forward every launcher argument while keeping script selection inside this repo."""

from pathlib import Path
import runpy
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "inspect_env"
if not name.isidentifier():
    raise SystemExit("Use a script name, without a path or .py extension")
legacy_scripts = Path(__file__).resolve().parent
root = legacy_scripts.parents[2]
script = root / "scripts" / (name + ".py")
if not script.is_file():
    script = legacy_scripts / (name + ".py")
if not script.is_file():
    raise SystemExit(f"Unknown script: {name}")
sys.path[:0] = [str(root / "src"), str(script.parent)]
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
