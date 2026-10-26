# FOV-dependent 3D SMLM Reconstruction with Displaced-Mask PSF Engineering

![Plug-and-Plane SMLM showoff](https://github.com/user-attachments/assets/5cef3014-fb7b-46d9-aae9-f7ef563285fc)

A field-of-view-aware 3D single-molecule localization microscopy (SMLM) reconstruction
pipeline, built for setups where a phase mask sits in an arbitrary plane between the
objective and the tube lens, displaced from the shift-invariant conjugate Fourier/pupil
plane. In this configuration the engineered PSF can depend on the emitter's position in
the camera field of view. The pipeline covers bead-based calibration of the displaced
phase mask, synthetic training-data generation, network training, and localization of
experimental blinking data.

> **Status:** research code under active development.

---

## Setup & Run the GUI

The actively maintained, GUI-driven version of this pipeline lives in
**`Phase_Retrieval_stand_alone/`**. One-time setup:

```bash
# Linux / macOS / Git Bash
./setup/setup.sh

# Windows (cmd / PowerShell)
.\setup\setup.bat
```

This creates a virtual environment at `Phase_Retrieval_stand_alone/.venv` and installs
everything from `Phase_Retrieval_stand_alone/requirements.txt` (PyTorch, Gradio, etc.).
If your machine is stuck on an old CUDA driver (e.g. CUDA 11.3), use
`requirements-py310-cu113.txt` instead — see the comment at the top of that file.

Then launch the GUI:

```bash
cd Phase_Retrieval_stand_alone
.venv\Scripts\activate       # Windows; use "source .venv/bin/activate" on Linux/macOS
python main.py
```

This opens a browser GUI at `http://127.0.0.1:7860` with one tab per pipeline stage:
Configure/Run (phase retrieval), Generate Training Data, Train Model, and Inference.
Use `--no-gui` to run PSF characterization headlessly instead
(edit the `cfg = Config(...)` block in `main.py` first), or `--host`/`--port` to change
how the GUI is served.

> The sections below describe the repository's original, older `main.py`-driven
> pipeline at the repo root (edit `main.py` directly, no GUI/`setup/` scripts). It's kept
> for reference but superseded by the GUI above for everyday use.

---

## Main Idea

In a conventional shift-invariant PSF model, one calibrated PSF represents every emitter
in the field, because the optical system is space-invariant at the Fourier/pupil plane.

With the mask displaced, off-axis emitters illuminate shifted regions of it, and the beam
can be converging or diverging at the mask plane depending on focal plane and
displacement — producing field-dependent PSF distortions that may need to be modeled.

The pipeline:

1. Retrieves a phase mask from a bead z-stack calibration.
2. Uses a displaced-mask forward model to simulate PSFs at different field positions.
3. Generates synthetic training frames over the full camera field of view.
4. Splits frames into training tiles while preserving their global FOV coordinates.
5. Trains a localization network to predict a 3D emitter volume.
6. Applies the network to experimental frames via overlapping, FOV-aware inference.
7. Exports localization lists as CSV files.

---

## Repository Structure

```text
.
├── setup/                 # setup.sh / setup.bat — one-time env setup for the GUI below
├── Phase_Retrieval_stand_alone/   # actively maintained, GUI-driven pipeline (see above)
├── Microtubules_dSTORM/   # experiment data for the root pipeline below
├── main.py                # root pipeline entry point; edit experiment parameters here
├── func_utils.py          # root pipeline: PSF, preprocessing, training data, training, inference
├── app_utils.py            # root pipeline implementation of the above steps
├── app.py                 # root pipeline's own GUI entry point, if used
└── DS3Dplus/
    ├── ds3d_utils.py       # optical model, sampling, dataset, volume-to-XYZ conversion, losses
    └── training_utils.py   # training loop and checkpoint handling
```

---

## Root Pipeline Reference (`main.py`-driven, no GUI)

This is the repository's original workflow: edit `main.py` directly and run it from an
IDE (or `python main.py`). Keep in mind that paths, microscope parameters, GPU index, and
acquisition-specific settings all live in `main.py` and must be edited before running.

### Environment

Standard scientific-Python + PyTorch stack (`torch`, `numpy`, `scipy`, `scikit-image`,
`matplotlib`, `scikit-learn`, `tqdm`; see `requirements.txt`). GPU is strongly recommended —
phase retrieval, training-data generation, training, and localization are slow on CPU.

### Required input data

1. **Central/on-axis bead z-stack** — a multi-plane TIFF, used for phase retrieval/PSF
   characterization:
   ```python
   zstack_file = "path/to/central_bead_zstack.tif"
   centralBeadCoordinates_pixel = [row, col]   # camera pixels, [row, col] order
   ```
   Demo file: `/Microtubules_dSTORM/Calibration_stack/step0.2um_027_center_x477_y573_2um.tif`

   <img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/da4667d8-77d2-4408-8aa8-db0ee23f576f" />

2. **Off-axis bead z-stacks** — constrain the FOV-dependent phase-retrieval model:
   ```python
   offaxis_zstack_files = ["path/to/offaxis_bead_1.tif", "path/to/offaxis_bead_2.tif", ...]
   offaxis_coords_pixel = [[row1, col1], [row2, col2], ...]   # camera pixels, [row, col]
   ```
   Demo files: `/Microtubules_dSTORM/Calibration_stack/step0.2um_027_*.tif` (7 off-axis stacks)

   <p align="left">
     <img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/1fb1e356-ece1-42ec-8547-4171b391c2c1" />
     <img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/8c3439f4-db17-400b-8660-e610384c5705" />
     <img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/8aed3e0a-337c-4ceb-91af-d956369e9157" />
   </p>

3. **Experimental blinking frames** — a folder of raw TIFF frames:
   ```python
   raw_image_folder = "path/to/blinking_frames"
   ```
   Preprocessing writes a background-removed copy with suffix `_br`. Frame size should be
   divisible by `num_tiles` (e.g. 1200 px divides evenly by 8, 6, 5, 4, 3, or 2). Inspect a
   few background-removed frames and estimate the noise range manually from an ROI (mean =
   background offset, std = noise level):

   <img width="476" height="504" alt="image" src="https://github.com/user-attachments/assets/743870ff-142f-4e5c-a2d5-2090f664bb42" />
   <img width="234.5" height="192" alt="image" src="https://github.com/user-attachments/assets/010f43f8-f19c-46ff-899b-0074cf739aee" />

   Demo folder: `/Microtubules_dSTORM/Experiment_50_frames/`

### Main configuration parameters (edit in `main.py`)

**Optical/acquisition:**
```python
M = 100; NA = 1.45; n_immersion = 1.518; lamda = 0.67   # emission wavelength [um]
n_sample = 1.33; f_4f = 200000; ps_camera = 11; ps_BFP = 80   # [um]
```

**Axial calibration and reconstruction range:**
```python
nfp_text = "-5.0, -1.0, 21"   # bead-stack start, end, number of planes [um]
NFP = -3.0 + 1.6               # nominal focal-plane parameter [um]
zrange = [0.2, 3.2]             # reconstructed emitter z-range [um]
num_z_voxel = 81                # output z voxels
```
`nfp_text` is the bead-stack calibration range; `zrange` is the training/output z-range;
`NFP` accounts for any offset between the two — e.g. if the bead-stack range is centered
at -3.0 µm and the true focal plane is 1.6 µm inside the sample, `NFP = -3.0 + 1.6`. Get
this wrong and the simulated training PSFs are systematically out of focus.

**Training image size and tiling:**
```python
training_im_size = 1200   # full synthetic canvas, px
num_tiles = 8              # 8x8 tiles; tile_size_px = training_im_size / num_tiles
```
Must divide evenly (crop/pad otherwise). Tile filenames (`00800_FOV_00012.tif`) encode
frame number and tile index (0 = top-left, `num_tiles**2 - 1` = bottom-right).

<img width="225" height="225" alt="image" src="https://github.com/user-attachments/assets/4d485e02-802d-42e6-906a-c662f32e81e2" />

**Inference overlap:** `center_fraction = 0.75` — fraction of each tile trusted during
overlapping tiled inference. Lower values increase overlap (fewer edge artifacts, slower).

**Noise/signal simulation:**
```python
shot_noise_background_range = (8.0, 16.0)
noise_offset_range = (10.0, 30.0)
```
`snr_roi`/`max_pv` drive an optional automatic SNR-estimation stage (`func3`, inherited
from AutoDS3D); in practice these are usually set manually after inspecting
background-removed images.

### Important parameters in `func_utils.py`

- `param_dict["g_sigma"] = 1.4` — image-plane blur used in phase retrieval and
  training-data simulation (typical range ~0.8–1.8; also optimized during phase
  retrieval, preferably at a low LR for fine-tuning).
- `d_bounds_um = (15000.0, 35000.0)`, `mask_iris_diameter_mm = 5.3` — phase-retrieval
  bounds; for a Nikon 100x/1.45 objective the expected mask displacement is ~27 mm.
- `fine_defocus_range_um = 0.4`, `fine_defocus_step_um = 0.1`, `max_shift_px = 15` —
  per-bead lateral/axial shift search for robustness. The lateral shift is cyclic, so
  keep the emitter ROI large enough to avoid cropping the PSF.

### Running the pipeline

Edit `CUSTOM_STEPS` at the bottom of `main.py` and press **Run**:

```python
CUSTOM_STEPS = ["psf", "preproc", "snr", "td", "train", "test", "localize"]
if __name__ == "__main__":
    run_custom_steps(CUSTOM_STEPS)
```

| Stage | Description |
|---|---|
| `psf` | Phase retrieval / PSF characterization |
| `preproc` | Background removal for experimental frames |
| `snr` | Optional automatic SNR/noise estimation |
| `td` | Synthetic training-data generation |
| `train` | Network training |
| `test` | Test inference on one simulated and one experimental frame |
| `localize` | Full experimental localization |

Steps run in order — e.g. `"test"` before `"train"` will error. Before training from
scratch, set `resume_net_file = None` in `func_utils.py`. Checkpoints save to
`training_results/`; `last_net_*.pt` can resume training.

### Output files

```text
phase_retrieval_outputs/       # retrieved mask, fitted parameters, sim/exp comparisons
training_data/x/                # saved training tiles
training_data/y.pickle          # labels
training_data/param.pickle      # parameter dictionary used for training data
training_results/               # checkpoints and training curves
sim_exp.tif                     # side-by-side simulated/experimental comparison
loss_curves.jpg                 # training/validation loss curves
sim_loc_gt_rec.jpg              # test localization comparison
exp_im_gt_rec.jpg               # experimental test image/reconstruction overlay
localizations_*_chunk_*.csv     # localization chunks (e.g. one CSV per 5000 frames)
```

### Troubleshooting

- **CUDA/GPU**: set `GPU_INDEX` in `main.py`; falls back to CPU if unavailable (slow for
  large datasets).
- **SNR stage**: inherited from AutoDS3D; if skipped, set the photon-count range manually
  before generating training data.
- **Empty/poor localizations**: check `threshold`, `zrange`, `NFP`, `center_fraction`,
  `training_im_size`, `num_tiles`, the generated tiles in `training_data/x/`, whether
  `sim_exp.tif` resembles the experimental data, and whether the retrieved mask matches
  the bead z-stacks.
- **Tiling artifacts** near tile borders: lower `center_fraction` (0–1) to increase
  inference overlap.

---

## Citation

If this code is used in a publication, cite the associated manuscript/preprint and the
original AutoDS3D work the localization network is based on:

```text
Author list. Plug-and-play 3D localization microscopy with field-dependent reconstruction. bioRxiv, YEAR.
```

## Contact

For questions, bugs, or data-specific configuration issues, contact the repository
maintainer.
