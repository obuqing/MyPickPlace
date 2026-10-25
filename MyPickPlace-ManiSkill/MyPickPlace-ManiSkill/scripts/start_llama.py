"""Start the workspace's portable Ollama without opening a console window."""

import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


def version():
    try:
        with urlopen("http://127.0.0.1:11434/api/version", timeout=2) as response:
            return json.load(response)["version"]
    except OSError:
        return None


def main():
    root = Path(__file__).resolve().parents[3]
    directory = root / ".cache/ollama"
    executable = directory / "bin/ollama.exe"
    if not version():
        if not executable.is_file():
            raise SystemExit("Portable Ollama is missing from .cache/ollama/bin/ollama.exe")
        environment = os.environ.copy()
        environment.update(
            OLLAMA_HOST="127.0.0.1:11434", OLLAMA_MODELS=str(directory / "models"),
            OLLAMA_KEEP_ALIVE="30m", OLLAMA_NUM_PARALLEL="1",
        )
        with (directory / "server.stdout.log").open("ab") as stdout, (directory / "server.stderr.log").open("ab") as stderr:
            process = subprocess.Popen(
                [str(executable), "serve"], cwd=root, env=environment,
                stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        (directory / "server.pid").write_text(str(process.pid), encoding="ascii")
        for _ in range(20):
            if version():
                break
            if process.poll() is not None:
                raise SystemExit("Ollama exited; see .cache/ollama/server.stderr.log")
            time.sleep(0.5)
        else:
            raise SystemExit("Ollama startup timed out; see .cache/ollama/server.stderr.log")
    print(f"Ollama {version()} at http://127.0.0.1:11434")
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
        print("Downloaded models:", ", ".join(model["name"] for model in json.load(response)["models"]))


if __name__ == "__main__":
    main()
