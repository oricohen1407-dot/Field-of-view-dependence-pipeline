#!/usr/bin/env bash
# Sets up the Python environment for Phase_Retrieval_stand_alone (the GUI pipeline) on
# Linux/macOS/Git-Bash. Creates Phase_Retrieval_stand_alone/.venv and installs requirements.txt.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../Phase_Retrieval_stand_alone" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

PYTHON_BIN="python3"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Could not find python3/python on PATH. Install Python 3.10+ and try again." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"   # Windows venv layout (Git Bash)
else
    source "$VENV_DIR/bin/activate"       # Linux/macOS venv layout
fi

# Always go through "python -m pip", never the bare pip executable: Windows security
# tools (e.g. Device Guard) can block pip right after it upgrades/replaces itself.
python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo
echo "Setup complete."
echo "To run the GUI:"
echo "  cd \"$PROJECT_DIR\""
echo "  source .venv/Scripts/activate   (or .venv/bin/activate on Linux/macOS)"
echo "  python main.py"
