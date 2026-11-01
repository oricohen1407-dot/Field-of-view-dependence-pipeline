from pathlib import Path
from config.config import Config
from app_utils import (
    phase_retrieval,
    show_z_psf,
    fit_mask_offset_from_offaxis_stacks
)
import numpy as np
import torch
import scipy.io as sio
from image_model import simulation_grid_size
from DS3Dplus.ds3d_utils import select_device

PROJECT_DIR = Path(__file__).resolve().parent


def _load_external_mask(path: str, expected_shape: tuple) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == '.npy':
        mask = np.load(path)
    elif ext == '.mat':
        mask_dict = sio.loadmat(path)
        real_keys = [k for k in mask_dict.keys() if not k.startswith('__')]
        if len(real_keys) != 1:
            raise ValueError(
                f"external_mask .mat file '{path}' must contain exactly one variable, "
                f"found {len(real_keys)}: {real_keys}"
            )
        mask = mask_dict[real_keys[0]]
    else:
        raise ValueError(
            f"Unsupported external_mask file extension '{ext}' for '{path}'. "
            f"Supported formats: .npy (np.save) or .mat (scipy.io.savemat, single variable)."
        )

    if np.iscomplexobj(mask):
        raise ValueError(f"external_mask must be real-valued (radians), got dtype {mask.dtype}")
    if mask.shape != expected_shape:
        raise ValueError(
            f"external_mask has shape {mask.shape}, but this run's optical parameters require "
            f"a simulation grid of shape {expected_shape}; no auto-resize is done"
        )
    return mask.astype(np.float32)


def characterize_PSF(cfg: Config, live_box=None, stop_event=None):
    param_dict = cfg.generate_param_dict()
    pr_dict = cfg.generate_pr_dict()

    if live_box is not None:
        param_dict['live_box'] = live_box
    if stop_event is not None:
        param_dict['stop_event'] = stop_event

    device = select_device()
    param_dict['device'] = device
    print(f'device used (characterize_PSF): {device}')

    if cfg.user.external_mask is not None:
        N = simulation_grid_size(param_dict['f_4f'], param_dict['lamda'], param_dict['ps_camera'], param_dict['ps_BFP'])
        param_dict['phase_mask_init'] = _load_external_mask(cfg.user.external_mask, (N, N))
        print(f'Using external mask as starting guess for phase retrieval: {cfg.user.external_mask}')

    phase_mask, g_sigma, ccs = phase_retrieval(param_dict, pr_dict)
    print(f'Phase mask is retrieved. blue sigma: {np.round(g_sigma, decimals=2)}.')
    if len(ccs) > 0:
        print(f'PSF modeling accuracy: average cc of {np.round(np.mean(ccs), decimals=4)}.')
    else:
        print('PSF modeling accuracy: N/A (epochs=0, no phase-retrieval training steps were run).')

    param_dict['g_sigma'] = (np.round(0.8*g_sigma, decimals=2), np.round(1.0*g_sigma, decimals=2))
    param_dict['phase_mask'] = phase_mask

    # DS3Dplus/ds3d_utils.py's ImModel/ImModelBase still require a single scalar 'NFP' key
    param_dict['NFP'] = param_dict['nfp_offset_um']

    show_z_psf(param_dict)

    save_dir = param_dict.get('mask_fit_save_dir') or str(PROJECT_DIR / 'mask_fit_outputs')
    fit_mask_offset_from_offaxis_stacks(param_dict, save_dir=save_dir)

    return ("PSF characterization is done. Check "
            "\nphase_retrieval_results.jpg "
            "\nPSFs.jpg")
