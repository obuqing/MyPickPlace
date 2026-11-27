"""Forward every launcher argument while keeping script selection inside this repo."""

from pathlib import Path
import runpy
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "inspect_env"
if not name.isidentifier():
    raise SystemExit("Use a script name, without a path or .py extension")
scripts = Path(__file__).resolve().parent
script = scripts / (name + ".py")
if script == Path(__file__).resolve() or not script.is_file():
    raise SystemExit(f"Unknown script: {name}")
sys.path.insert(0, str(scripts.parent / "src"))
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
