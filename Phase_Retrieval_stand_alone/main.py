# main.py
# GUI entry point (see repo root README.md for setup via setup/setup.sh or setup/setup.bat).
# Use --no-gui to run AutoDS3D steps headlessly instead; edit the PARAMS below first.

import argparse
import os
from pathlib import Path
from config.config import Config, UserConfig
from config.emitter_centers import (
    PROJECT_DIR as DATA_ROOT_DIR, ZSTACK_FILES_PATH,
    ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL, RAW_IMAGE_FOLDER
)
from func_utils import characterize_PSF
from gui import build_demo

# TODO (RK): Delete and remove import if unnecessary
# Avoid GUI backends on a headless server
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# EDIT THIS BLOCK to configure your experiment
# =============================================================================
cfg = Config(
    user=UserConfig(
        project_dir=str(DATA_ROOT_DIR),
        zstack_folder=str(ZSTACK_FILES_PATH),
        zstack_file=ZSTACK_FILE,
        central_bead_coordinates_pixel=CENTRAL_BEAD_COORDINATES_PIXEL,
        offaxis_zstack_files=OFFAXIS_ZSTACK_FILES,
        offaxis_coords_pixel=OFFAXIS_COORDS_PIXEL,
        external_mask=None,           # optional starting-guess mask (.npy/.mat) for phase retrieval
    )
)
# =============================================================================


def run_characterize_PSF():
    characterize_PSF(cfg)
    cfg.save(str(PROJECT_DIR / "config" / "config.json"))


def main():
    parser = argparse.ArgumentParser(description="DeepSTORM3D PSF characterization")
    parser.add_argument("--no-gui", action="store_true", help="Run headlessly without opening the browser GUI")
    parser.add_argument("--host", default="127.0.0.1",
                         help="Interface to bind the GUI to. Use 0.0.0.0 to accept connections "
                              "from other machines (LAN or public IP, depending on your network/"
                              "firewall/router setup). Default: 127.0.0.1 (local only).")
    parser.add_argument("--port", type=int, default=None,
                         help="Port to serve the GUI on (default: Gradio's own default, 7860).")
    args = parser.parse_args()
    if args.no_gui:
        run_characterize_PSF()
    else:
        build_demo().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
