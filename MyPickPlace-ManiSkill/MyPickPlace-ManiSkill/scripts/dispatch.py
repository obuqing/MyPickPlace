"""Forward every launcher argument while keeping script selection inside this repo."""

from pathlib import Path
import runpy
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "inspect_env"
if not name.isidentifier():
    raise SystemExit("Use a script name, without a path or .py extension")
script = Path(__file__).resolve().parent / (name + ".py")
if not script.is_file():
    raise SystemExit(f"Unknown script: {name}")
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
