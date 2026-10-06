import os
import csv
import json
import math
import time
import pickle
import torch
import numpy as np
import tifffile
from skimage import io
from scipy import ndimage
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from image_model import ImModel_pr
from DS3Dplus.ds3d_utils import (
    ImModel, ImModelBase, ImModelTraining, Sampling,
    MyDataset, LON as Net, KDE_loss3D, Volume2XYZ, calc_jaccard_rmse, select_device,
)
from DS3Dplus.training_utils import TorchTrainer

def _load_zstack_with_count(path: str):
    """Z (slice count) is authoritative from the file itself — these TIFFs carry no
    z-spacing/calibration metadata. Cross-checks ImageJ's 'slices' tag if present (warn only)."""
    zst = io.imread(path).astype(np.float32)
    if zst.ndim != 3:
        raise ValueError(f"{path}: expected a 3D (Z,H,W) z-stack TIFF, got shape {zst.shape}")
    Z = int(zst.shape[0])
    try:
        with tifffile.TiffFile(path) as tf:
            slices_tag = (tf.imagej_metadata or {}).get('slices')
        if slices_tag is not None and int(slices_tag) != Z:
            print(f"[PR] WARNING: {path}: ImageJ 'slices' tag={slices_tag} but frame count={Z}; using {Z}.")
    except Exception as exc:
        print(f"[PR] NOTE: could not read ImageJ metadata from {path} ({exc}); using frame count {Z}.")
    return zst, Z

def _norm01_sum(im):
    im = im.astype(np.float32, copy=False)
    im = im - im.min()
    s = float(im.sum())
    if s > 0:
        im /= s
    return im

def _norm_zm_unit(x, eps=1e-6):
    x = x - x.mean()
    return x / (x.std() + eps)

@torch.no_grad()
def cc_score(a, b, eps=1e-6):
    a = _norm_zm_unit(a.float(), eps).flatten()
    b = _norm_zm_unit(b.float(), eps).flatten()
    return (a @ b) / (a.norm() * b.norm() + eps)

@torch.no_grad()
def phasecorr_shift_int(a, b, max_shift_px=None, eps=1e-6):
    """
    Returns (dy, dx) integer shift that best aligns b to a (translation only),
    using phase correlation. a,b: [H,W] torch tensors.
    """
    H, W = a.shape
    a0 = _norm_zm_unit(a.float(), eps)
    b0 = _norm_zm_unit(b.float(), eps)

    A = torch.fft.fftn(a0)
    B = torch.fft.fftn(b0)
    R = A * torch.conj(B)
    R = R / (torch.abs(R) + eps)
    cc = torch.fft.ifftn(R).real  # [H,W]

    # peak index
    k = torch.argmax(cc)
    py = (k // W).item()
    px = (k %  W).item()

    # wrap to signed
    if py > H // 2: py -= H
    if px > W // 2: px -= W

    # optional: clamp shifts (prevents crazy jumps)
    if max_shift_px is not None:
        py = int(max(-max_shift_px, min(max_shift_px, py)))
        px = int(max(-max_shift_px, min(max_shift_px, px)))

    return py, px

def calculate_cc(output, target):
    # output: rank 3, target: rank 3
    output_mean = np.mean(output, axis=(1, 2), keepdims=True)
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    ccs = (np.sum((output - output_mean) * (target - target_mean), axis=(1, 2)) /
           (np.sqrt(np.sum((output - output_mean) ** 2, axis=(1, 2)) * np.sum((target - target_mean) ** 2,
                                                                              axis=(1, 2))) + 1e-9))
    return ccs

def phase_retrieval(param_dict, pr_dict, fig_flag=True):
    device = param_dict['device']

    # ----------------------------
    # Collect stacks: on-axis + off-axis
    # ----------------------------
    stacks = []

    # RK: on-axis stack (x=y=0)
    zstack_on, Z = _load_zstack_with_count(pr_dict['zstack_file_path'])
    stacks.append(("onaxis", zstack_on, 0.0, 0.0))

    # off-axis stacks (if provided)
    if 'offaxis_zstack_files' in param_dict and len(param_dict['offaxis_zstack_files']) > 0:
        r0, c0 = param_dict['centralBeadCoordinates_pixel']
        ps_cam = float(param_dict['ps_camera'])
        M = float(param_dict['M'])

        for f, (rr, cc) in zip(param_dict['offaxis_zstack_files'], param_dict['offaxis_coords_pixel']):
            zst = io.imread(f).astype(np.float32)
            dx_pix = float(cc) - float(c0)
            dy_pix = float(rr) - float(r0)

            # ✅ correct physical conversion (sample-plane um)
            x_um = dx_pix * (ps_cam / M)
            y_um = dy_pix * (ps_cam / M)

            stacks.append((os.path.splitext(os.path.basename(f))[0], zst, x_um, y_um))

    # ----------------------------
    # Normalize and pack into one training batch
    # ----------------------------
    y_list = []
    xyz_list = []
    zi_list = []
    bead_id_list = []
    stack_index = 0

    is_onaxis_list = []  # <-- add

    for name, zst, x_um, y_um in stacks:
        stack_index += 1
        bead_id = stack_index - 1  # added on 15/03/2026
        if zst.shape[0] != Z:
            raise ValueError(
                f"{name}: Z mismatch. stack has {zst.shape[0]} but on-axis calibration stack has {Z}"
            )

        # ---------- ORIGINAL PR BACKGROUND CLEANUP ----------
        corner_size = max(7, int(0.1 * zst.shape[1]))

        patches = np.concatenate(
            (
                np.concatenate(
                    (zst[:, :corner_size, :corner_size],
                     zst[:, :corner_size, -corner_size:]),
                    axis=2
                ),
                np.concatenate(
                    (zst[:, -corner_size:, :corner_size],
                     zst[:, -corner_size:, -corner_size:]),
                    axis=2
                ),
            ),
            axis=1
        )

        means = np.mean(patches, axis=(1, 2), keepdims=True)
        stds = np.std(patches, axis=(1, 2), keepdims=True)

        zst = zst - means
        mask = (zst > stds*1)

        struct = ndimage.generate_binary_structure(2, 1)
        mask = np.array([
            ndimage.binary_dilation(
                ndimage.binary_erosion(mask[i], struct),
                struct
            )
            for i in range(mask.shape[0])
        ], dtype=np.float32)

        zst = zst * mask
        # ---------- END CLEANUP ----------
        # Autocorrelation
        '''# after:
        # zst = zst * mask

        if pr_dict.get("recenter_offaxis", True) and (name != "onaxis"):
            zst, shifts = recenter_stack_per_slice(zst, ref_mode="midz", upsample=10)
        #    if name == "onaxis":
        #        print(f"[recenter] {name}: dy={dy:.2f}px dx={dx:.2f}px")'''
        # end autocorrelation


        # normalize AFTER cleanup (as in original PR)
        #zst = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)
        zst = np.clip(zst, 0.0, None).astype(np.float32)
        z_photons = np.sum(zst, axis=(1, 2)).astype(np.float32)
        for zi in range(Z):
            #y_list.append(zst[zi])
            #y_list.append(zst[zi]/z_photons[Z//2])  # normalize according to center

            #if stack_index == 1:
            #norm_factor = z_photons[Z//2]
            norm_factor = z_photons[zi]
            y_list.append(zst[zi] / norm_factor)  # normalize according to center
            #xyz_list.append([x_um, y_um, 0.0, float(z_photons[zi])])  # <-- photons restored
            xyz_list.append([x_um, y_um, 0.0, 1.0])
            zi_list.append(zi)
            bead_id_list.append(bead_id)
            is_onaxis_list.append(name == "onaxis")  # <-- add

        # normalize AFTER masking
        #zst = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)

    y_true = torch.from_numpy(np.stack(y_list, 0)).to(device)  # [B,H,W]
    xyzps = torch.from_numpy(np.asarray(xyz_list, np.float32)).to(device)  # [B,4]

    # per-sample slice index; NFPs itself is recomputed from im_model.nfps(zi_tensor) each epoch
    zi_tensor = torch.tensor(np.asarray(zi_list, np.int64), device=device)
    bead_ids = torch.tensor(np.asarray(bead_id_list, np.int64), device=device)  # added on 15/03/2026
    is_onaxis = torch.tensor(is_onaxis_list, device=device)  # [B] bool

    # ----------------------------
    # Build PR model (now includes d via ASM)
    # ----------------------------

    # zstack is numpy (Z,Hroi,Wroi)
    Hroi, Wroi = y_true.shape[-2], y_true.shape[-1]  # <-- ALWAYS matches the training target
    param_dict['H'] = int(Hroi)
    param_dict['W'] = int(Wroi)

    params_pr = dict(param_dict)
    params_pr['H'] = int(Hroi)
    params_pr['W'] = int(Wroi)

    # initial d: explicit Config override > warm start from a prior run > bounds midpoint
    d_init_um = param_dict['d_init_um']
    if d_init_um is None:
        d_init_um = param_dict.get('mask_offset_in_um')  # warm start, e.g. resuming a prior fit
    if d_init_um is None:
        d_init_um = 0.5 * (param_dict['d_min_um'] + param_dict['d_max_um'])
    d_init_um = float(d_init_um)

    if not (param_dict['d_min_um'] <= d_init_um <= param_dict['d_max_um']):
        raise ValueError(
            f"d_init_um={d_init_um} must be inside "
            f"[{param_dict['d_min_um']}, {param_dict['d_max_um']}]."
        )

    params_pr['mask_offset_in_um'] = d_init_um
    # end ori's edit from 26/01/2026 for improved pr with displacement

    # initial NFP offset: explicit Config override > warm start from a prior run > bounds midpoint
    nfp_range_um = param_dict['nfp_range_um']
    nfp_offset_min_um = param_dict['nfp_offset_min_um']
    nfp_offset_max_um = param_dict['nfp_offset_max_um']

    nfp_offset_init = param_dict['nfp_offset_init_um']
    if nfp_offset_init is None:
        nfp_offset_init = param_dict.get('nfp_offset_um')  # warm start, e.g. resuming a prior fit
    if nfp_offset_init is None:
        nfp_offset_init = 0.5 * (nfp_offset_min_um + nfp_offset_max_um)
    nfp_offset_init = float(nfp_offset_init)

    if not (nfp_offset_min_um <= nfp_offset_init <= nfp_offset_max_um):
        raise ValueError(
            f"nfp_offset_init_um={nfp_offset_init} must be inside "
            f"[{nfp_offset_min_um}, {nfp_offset_max_um}]."
        )

    params_pr['nfp_range_um'] = nfp_range_um
    params_pr['nfp_offset_min_um'] = nfp_offset_min_um
    params_pr['nfp_offset_max_um'] = nfp_offset_max_um
    params_pr['nfp_offset_init_um'] = nfp_offset_init
    params_pr['Z'] = Z

    im_model = ImModel_pr(params_pr).to(device)

    im_model.train()

    opt = torch.optim.Adam(
        [
            {'params': [im_model.phase_mask], 'lr': pr_dict['lr_phase_mult'] * pr_dict['learning_rate']},
            {'params': [im_model.g_sigma],    'lr': pr_dict['lr_sigma_mult'] * pr_dict['learning_rate']},
            {'params': [im_model.d_raw],      'lr': pr_dict['lr_d_mult']     * pr_dict['learning_rate']},
            {'params': [im_model.nfp_offset_raw], 'lr': pr_dict['lr_nfp_mult'] * pr_dict['learning_rate']},
        ],
        betas=tuple(pr_dict['adam_betas'])
    )

    # ----------------------------
    # Live GUI panel: dense per-epoch history + periodic PSF-grid snapshot
    # ----------------------------
    live_box = param_dict.get('live_box')
    live_debug_every_epochs = int(pr_dict['live_debug_every_epochs'])
    bead_names = [name for name, _, _, _ in stacks]

    loss_history = []
    d_history = []
    nfp_offset_history = []
    g_sigma_history = []
    bead_cursor = 0
    live_panel_time_s = 0.0  # cumulative wall-clock time spent inside _refresh_live_panel

    def _refresh_live_panel(step, loss_val, d_now, nfp_offset_now, g_sigma_now,
                             pred_disp, target_disp, local_bead_ids, local_zi):
        """Unconditionally (re)builds the live_box payload from the current state.
        Callers gate on live_box/refresh-cadence; this never appends to history."""
        nonlocal bead_cursor, live_panel_time_s
        _panel_t0 = time.perf_counter()
        available_beads = torch.unique(local_bead_ids).tolist()
        bead = available_beads[bead_cursor % len(available_beads)]
        bead_cursor += 1

        idx = torch.where(local_bead_ids == bead)[0]
        order = torch.argsort(local_zi[idx])
        idx = idx[order]

        pred_bead = pred_disp[idx].detach().cpu().numpy()      # [Zb,H,W]
        target_bead = target_disp[idx].detach().cpu().numpy()  # [Zb,H,W]

        Zb = pred_bead.shape[0]
        n_slices = min(7, Zb)
        slice_idx = np.round(np.linspace(0, Zb - 1, n_slices)).astype(int)

        pred_slices = np.stack([pred_bead[i] / (pred_bead[i].max() + 1e-12) for i in slice_idx], axis=0)
        target_slices = np.stack([target_bead[i] / (target_bead[i].max() + 1e-12) for i in slice_idx], axis=0)

        with torch.no_grad():
            nfp_vals = im_model.nfps(torch.tensor(slice_idx, device=device)).detach().cpu().numpy()

        mid = idx[len(idx) // 2]
        live_box['phase'] = im_model.last_ef_bfp_phase[mid].cpu().numpy()
        live_box['mask_phase'] = im_model.last_mask_plane_phase[mid].cpu().numpy()
        shift_px = im_model.last_mask_shift_px[mid].cpu().tolist()
        live_box['mask_shift_px'] = (int(shift_px[0]), int(shift_px[1]))
        live_box['pred_slices'] = pred_slices
        live_box['target_slices'] = target_slices
        live_box['slice_zi'] = slice_idx.tolist()
        live_box['slice_nfp_um'] = nfp_vals.tolist()
        live_box['bead_name'] = bead_names[bead]
        live_box['loss_history'] = list(loss_history)
        live_box['d_history'] = list(d_history)
        live_box['nfp_offset_history'] = list(nfp_offset_history)
        live_box['g_sigma_history'] = list(g_sigma_history)
        live_box['meta'] = {
            'step': step, 'loss': loss_val, 'd': d_now,
            'nfp_offset': nfp_offset_now, 'g_sigma': g_sigma_now,
            'bead_name': bead_names[bead],
        }
        live_box['version'] = live_box.get('version', 0) + 1
        live_panel_time_s += time.perf_counter() - _panel_t0

    def _update_live_panel(step, loss_val, d_now, nfp_offset_now, g_sigma_now,
                            pred_disp, target_disp, local_bead_ids, local_zi):
        loss_history.append(loss_val)
        d_history.append(d_now)
        nfp_offset_history.append(nfp_offset_now)
        g_sigma_history.append(g_sigma_now)

        if live_box is None or (step % live_debug_every_epochs) != 0:
            return
        _refresh_live_panel(step, loss_val, d_now, nfp_offset_now, g_sigma_now,
                             pred_disp, target_disp, local_bead_ids, local_zi)

    ccs = []
    fine_defocus_range_um = float(pr_dict.get("fine_defocus_range_um", 0.6))
    fine_defocus_step_um = float(pr_dict.get("fine_defocus_step_um", 0.1))
    max_shift_px = int(pr_dict.get("max_shift_px", 10))

    delta_candidates = np.arange(
        -fine_defocus_range_um,
        fine_defocus_range_um + 0.5 * fine_defocus_step_um,
        fine_defocus_step_um,
        dtype=np.float32
    )
    # end
    stop_event = param_dict.get('stop_event')

    _timing_t0 = time.perf_counter()  # covers warmup + main loop only, for the calc-vs-display breakdown below

    # --- Phase A: mask-only warmup from the on-axis bead alone, d/NFP/g_sigma held fixed ---
    # d's gradient depends on the mask having real structure (see phase_retrieval physics
    # notes); this gives the mask (fast LR) a head start before off-axis beads and NFP join in.
    # g_sigma is held fixed too so the optimizer can't lower loss via blur instead of real mask structure.
    mask_warmup_epochs = int(pr_dict['mask_warmup_epochs'])
    if mask_warmup_epochs > 0:
        onaxis_idx = torch.where(is_onaxis)[0]
        xyzps_onaxis = xyzps[onaxis_idx]
        y_onaxis = y_true[onaxis_idx]
        zi_onaxis = zi_tensor[onaxis_idx]

        held_fixed = [im_model.d_raw, im_model.nfp_offset_raw, im_model.g_sigma]

        for warmup_epoch in range(mask_warmup_epochs):
            if stop_event is not None and stop_event.is_set():
                print(f"[PR][warmup] stop requested — halting at epoch {warmup_epoch}")
                break
            im_model.current_epoch = warmup_epoch
            opt.zero_grad()
            pred = im_model(xyzps_onaxis, im_model.nfps(zi_onaxis), targets=y_onaxis)
            loss = F.mse_loss(pred, y_onaxis)
            loss.backward()

            snapshots = [p.detach().clone() for p in held_fixed]
            opt.step()
            with torch.no_grad():
                for p, snap in zip(held_fixed, snapshots):
                    p.copy_(snap)
                im_model.g_sigma.clamp_(min=1e-3, max=20.0)

            is_last_warmup_epoch = warmup_epoch == mask_warmup_epochs - 1
            if (warmup_epoch % 10) == 0 or is_last_warmup_epoch:
                print(f"[PR][warmup] epoch {warmup_epoch:4d} loss={float(loss.item()):.6g}")

            if live_box is not None and ((warmup_epoch % live_debug_every_epochs) == 0 or is_last_warmup_epoch):
                _refresh_live_panel(
                    warmup_epoch, float(loss.item()),
                    float(im_model.d_um().detach().cpu().item()),
                    float(im_model.nfp_offset_um().detach().cpu().item()),
                    float(im_model.g_sigma.item()),
                    pred, y_onaxis,
                    bead_ids[onaxis_idx], zi_onaxis,
                )

        print(f"[PR] mask warmup done ({mask_warmup_epochs} epochs, on-axis only) "
              f"— d/NFP/g_sigma now free to move, Adam momentum already warmed up")

    # Track the best (lowest-loss) main-loop state so the run's final output reflects the
    # best point found, not wherever training happened to end up — loss can tick back up after
    # its minimum (e.g. g_sigma drifting late, noisy per-bead alignment search), so the last
    # epoch isn't necessarily the best one. Only main-loop epochs are compared (all beads, same
    # loss formulation) — warmup loss (on-axis only) isn't comparable, per existing design.
    best_loss = float('inf')
    best_state = None

    pred_display = target_display = d_now = nfp_offset_now = None
    for epoch in range(pr_dict['epochs']):
        if stop_event is not None and stop_event.is_set():
            print(f"[PR] stop requested — halting at epoch {epoch}")
            break
        # continues on from the warmup phase's own 0..mask_warmup_epochs-1 numbering, so
        # on-disk debug filenames never collide between the two phases
        im_model.current_epoch = mask_warmup_epochs + epoch
        opt.zero_grad()
        NFPs = im_model.nfps(zi_tensor)  # depends on the live nfp_offset_raw
        apply_off_axis_space_invariance = (max_shift_px > 0)

        if not apply_off_axis_space_invariance:
            pred = im_model(xyzps, NFPs, targets=y_true)  #original
            loss = F.mse_loss(pred, y_true)  #original
            # added on 15/03/2026 for small defocus robustness in pr
        else:
            # --------------------------------------------------
            # bead-wise robust alignment:
            # 1) one fixed shift per bead across z
            # 2) one fixed fine-defocus offset per bead across z
            # --------------------------------------------------
            with torch.no_grad():
                y_aligned = y_true.clone()
                nfp_offsets = torch.zeros_like(NFPs)

                unique_beads = torch.unique(bead_ids)

                for bid in unique_beads.tolist():
                    idx = torch.where(bead_ids == bid)[0]

                    # keep on-axis fixed
                    if bool(is_onaxis[idx[0]].item()):
                        continue

                    target_bead = y_true[idx]  # [Z,H,W]

                    best_loss = None
                    best_dd = 0.0
                    best_target_shifted = target_bead.clone()

                    for dd in delta_candidates:
                        nfp_cand = NFPs[idx] + float(dd)
                        pred_cand = im_model(xyzps[idx], nfp_cand, targets=target_bead)  # [Z,H,W] removed on 15/03/2026



                        target_shifted = target_bead.clone()

                        # allow a different shift for every z slice
                        # TODO (RK): goal here is to find the best cc per bead and not per z slice, this can potentially break the bead in half
                        for zi in range(pred_cand.shape[0]):
                            a = pred_cand[zi]
                            b = target_bead[zi]

                            dy, dx = phasecorr_shift_int(a, b, max_shift_px=max_shift_px)

                            # TODO (RK): check if roll is needed. probably not
                            b1 = torch.roll(b, shifts=(dy, dx), dims=(0, 1))
                            b2 = torch.roll(b, shifts=(-dy, -dx), dims=(0, 1))
                            if cc_score(a, b2) > cc_score(a, b1):
                                b1 = b2

                            target_shifted[zi] = b1

                        eps = 1e-12
                        pred_cand_n = pred_cand / (pred_cand.sum(dim=(1, 2), keepdim=True) + eps)
                        cand_loss = F.mse_loss(pred_cand_n, target_shifted).item()

                        if (best_loss is None) or (cand_loss < best_loss):
                            best_loss = cand_loss
                            best_dd = float(dd)
                            best_target_shifted = target_shifted.clone()

                    nfp_offsets[idx] = best_dd
                    y_aligned[idx] = best_target_shifted
            ''' replaced to make shift invariant per slice rather than per bead
            with torch.no_grad():
                y_aligned = y_true.clone()
                nfp_offsets = torch.zeros_like(NFPs)

                unique_beads = torch.unique(bead_ids)

                for bid in unique_beads.tolist():
                    idx = torch.where(bead_ids == bid)[0]

                    # on-axis bead: keep nominal NFP, no shift search
                    if bool(is_onaxis[idx[0]].item()):
                        continue

                    best_loss = None
                    best_dd = 0.0
                    best_shift = (0, 0)

                    target_bead = y_true[idx]  # [Z,H,W]

                    for dd in delta_candidates:
                        nfp_cand = NFPs[idx] + float(dd)
                        pred_cand = im_model(xyzps[idx], nfp_cand)  # [Z,H,W]

                        # one shift for the whole bead stack:
                        # use sum over z to estimate a single robust shift
                        a_ref = pred_cand.sum(dim=0)  # [H,W]
                        b_ref = target_bead.sum(dim=0)  # [H,W]

                        dy, dx = phasecorr_shift_int(a_ref, b_ref, max_shift_px=max_shift_px)

                        # sign ambiguity: test both directions on the summed image
                        b_ref_1 = torch.roll(b_ref, shifts=(dy, dx), dims=(0, 1))
                        b_ref_2 = torch.roll(b_ref, shifts=(-dy, -dx), dims=(0, 1))
                        if cc_score(a_ref, b_ref_2) > cc_score(a_ref, b_ref_1):
                            dy, dx = -dy, -dx

                        target_shifted = torch.roll(target_bead, shifts=(dy, dx), dims=(1, 2))

                        eps = 1e-12
                        pred_cand_n = pred_cand / (pred_cand.sum(dim=(1, 2), keepdim=True) + eps)
                        cand_loss = F.mse_loss(pred_cand_n, target_shifted).item()

                        if (best_loss is None) or (cand_loss < best_loss):
                            best_loss = cand_loss
                            best_dd = float(dd)
                            best_shift = (int(dy), int(dx))

                    # save best bead-wise alignment
                    nfp_offsets[idx] = best_dd
                    y_aligned[idx] = torch.roll(
                        target_bead,
                        shifts=best_shift,
                        dims=(1, 2)
                    )
                    ''' # replaced

            # forward again WITH grad, using the chosen per-bead fine defocus
            pred = im_model(xyzps, NFPs + nfp_offsets, targets=y_aligned)

            eps = 1e-12
            pred_n = pred / (pred.sum(dim=(1, 2), keepdim=True) + eps)
            loss = F.mse_loss(pred_n, y_aligned)
           
        # keep some MSE to prevent "degenerate" solutions
        #loss = 0.2 * loss_mse + 0.8 * loss_ac
        #loss = 0.0 * loss_mse + 1.0 * loss_ac

        # whichever pred/target the loss actually used this epoch, for the live panel
        pred_display = pred if not apply_off_axis_space_invariance else pred_n
        target_display = y_true if not apply_off_axis_space_invariance else y_aligned

        loss.backward()

        # snapshot the state that produced this epoch's loss (pre-step — opt.step() below
        # would otherwise move phase_mask/g_sigma/d_raw/nfp_offset_raw past it)
        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                'phase_mask': im_model.phase_mask.detach().clone(),
                'g_sigma': im_model.g_sigma.detach().clone(),
                'd_raw': im_model.d_raw.detach().clone(),
                'nfp_offset_raw': im_model.nfp_offset_raw.detach().clone(),
            }

        if epoch == 0:
            print("d_um:", im_model.d_um().detach().item())
            print("grad(d_raw):", None if im_model.d_raw.grad is None else im_model.d_raw.grad.detach().item())
            print("nfp_offset_um:", im_model.nfp_offset_um().detach().item())
            print("grad(nfp_offset_raw):", None if im_model.nfp_offset_raw.grad is None else im_model.nfp_offset_raw.grad.detach().item())

        opt.step()
        Visualize_mask = True
        # visualization
        if Visualize_mask:
            if epoch % 10 == 0:
                with torch.no_grad():
                    mask = im_model.phase_mask.detach().cpu().numpy()

                    # wrap phase to [-pi, pi] for visualization
                    mask_wrapped = np.angle(np.exp(1j * mask))

                    plt.figure(figsize=(4, 4))
                    plt.imshow(mask, cmap="twilight")
                    plt.colorbar()
                    plt.title(f"Phase mask, epoch {epoch}")
                    plt.tight_layout()

                    path2save = 'phase_retrieval_with_displacement_iteration'
                    if not (os.path.isdir(path2save)):
                        os.mkdir(path2save)
                    plt.savefig(os.path.join(path2save, 'iteration_' + str(epoch) +  '.jpg'), bbox_inches='tight', dpi=300)
                    plt.close()
                    # End visualization

        # keep sigma sane (optional but helps)
        with torch.no_grad():
            im_model.g_sigma.clamp_(min=1e-3, max=20.0)

        # monitor
        with torch.no_grad():
            pred2 = im_model(xyzps, NFPs, targets=y_true)
            cc = calculate_cc(pred2.detach().cpu().numpy(), y_true.detach().cpu().numpy())
            ccs.append(cc)

            d_now = float(im_model.d_um().detach().cpu().item())
            nfp_offset_now = float(im_model.nfp_offset_um().detach().cpu().item())
            if (epoch % 10) == 0:
                print(
                    f"[PR] epoch {epoch:4d} loss={float(loss.item()):.6g}  d={d_now:.2f} um  "
                    f"g_sigma={float(im_model.g_sigma.item()):.4f}  "
                    f"nfp_offset={nfp_offset_now:.3f} um")

        _update_live_panel(
            epoch, float(loss.item()), d_now, nfp_offset_now,
            float(im_model.g_sigma.item()), pred_display, target_display,
            bead_ids, zi_tensor,
        )

    if live_box is not None and pred_display is not None and (epoch % live_debug_every_epochs) != 0:
        _refresh_live_panel(
            epoch, float(loss.item()), d_now, nfp_offset_now,
            float(im_model.g_sigma.item()), pred_display, target_display,
            bead_ids, zi_tensor,
        )

    _total_time_s = time.perf_counter() - _timing_t0
    _calc_time_s = max(0.0, _total_time_s - im_model.debug_save_time_s - live_panel_time_s)
    if _total_time_s > 0:
        print(
            f"[PR] timing breakdown (warmup+main loop): total={_total_time_s:.1f}s  "
            f"calculation={_calc_time_s:.1f}s ({100*_calc_time_s/_total_time_s:.1f}%)  "
            f"debug_png_dump={im_model.debug_save_time_s:.1f}s ({100*im_model.debug_save_time_s/_total_time_s:.1f}%)  "
            f"live_panel={live_panel_time_s:.1f}s ({100*live_panel_time_s/_total_time_s:.1f}%)"
        )

    # Restore the best-loss main-loop state (if any main-loop epoch ran) so everything saved
    # below — d, g_sigma, phase_mask, the fitted NFP offset, and the exported sim stacks — comes
    # from the lowest-loss point found, not just wherever the last epoch happened to land.
    if best_state is not None:
        with torch.no_grad():
            im_model.phase_mask.copy_(best_state['phase_mask'])
            im_model.g_sigma.copy_(best_state['g_sigma'])
            im_model.d_raw.copy_(best_state['d_raw'])
            im_model.nfp_offset_raw.copy_(best_state['nfp_offset_raw'])
        print(f"[PR] restoring best-loss state (loss={best_loss:.6g}) for final outputs")

    # save final values back
    param_dict['mask_offset_in_um'] = float(im_model.d_um().detach().cpu().item())
    print(f"[PR] done. best d = {param_dict['mask_offset_in_um']:.2f} um")
    phase_mask = im_model.phase_mask.detach().cpu().numpy()
    g_sigma = float(im_model.g_sigma.detach().cpu().numpy())

    with torch.no_grad():
        NFPs = im_model.nfps(zi_tensor).detach()

    param_dict['nfp_offset_um'] = float(im_model.nfp_offset_um().detach().cpu().item())
    param_dict['nfp_range_um'] = float(im_model.nfp_range_um)
    param_dict['nfp_fitted_Z'] = int(Z)
    print(f"[PR] done. fitted NFP offset={param_dict['nfp_offset_um']:.3f} um "
          f"(range={param_dict['nfp_range_um']} um, Z={Z})")

    #print(f"[PR] done. best d = {d:.1f} um")

    # ----------------------------
    # SAVE OUTPUTS (after PR is done)
    # ----------------------------
    save_dir = pr_dict.get("save_dir", os.path.join(os.getcwd(), "phase_retrieval_outputs"))
    os.makedirs(save_dir, exist_ok=True)

    # save mask + scalar params
    np.save(os.path.join(save_dir, "phase_mask.npy"), phase_mask)
    with open(os.path.join(save_dir, "g_sigma_and_d.txt"), "w") as f:
        f.write(f"g_sigma = {g_sigma}\n")
        f.write(f"mask_offset_in_um (d) = {float(param_dict['mask_offset_in_um'])}\n")
        f.write(f"nfp_offset_um = {param_dict['nfp_offset_um']}\n")
        f.write(f"nfp_range_um = {param_dict['nfp_range_um']}\n")
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({
            "phase_mask_path": "phase_mask.npy",
            "g_sigma": float(g_sigma),
            "d_um": float(param_dict['mask_offset_in_um']),
            "nfp_offset_um": float(param_dict['nfp_offset_um']),
            "nfp_range_um": float(param_dict['nfp_range_um']),
        }, f, indent=2)

    # helper: float stack -> uint16 for viewing
    def _to_u16(st):
        st = st.astype(np.float32)
        out = np.zeros_like(st, dtype=np.uint16)
        for i in range(st.shape[0]):
            mx = float(st[i].max())
            if mx > 0:
                out[i] = (np.clip(st[i] / mx, 0, 1) * 65535.0).astype(np.uint16)
        return out

    im_model.eval()
    cnt = -1
    with torch.no_grad():
        for name, zst, x_um, y_um in stacks:
            cnt+=1
            # z is always 0 — axial variation flows entirely through NFPs
            xyz_bead = np.stack([[x_um, y_um, 0.0, 1.0] for _ in range(Z)], axis=0).astype(np.float32)
            xyz_bead_t = torch.from_numpy(xyz_bead).to(device)

            pred = im_model(xyz_bead_t, im_model.nfps()).detach().cpu().numpy()  # [Z,H,W]; NFPs is num_beads*Z long, wrong shape here
            exp = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)   # [Z,H,W] (same norm as training)

            exp_u16 = _to_u16(exp)
            sim_u16 = _to_u16(pred)

            io.imsave(os.path.join(save_dir, f"exp_stack_{cnt}_{name}.tif"), exp_u16, check_contrast=False)
            io.imsave(os.path.join(save_dir, f"sim_stack_{cnt}_{name}.tif"), sim_u16, check_contrast=False)

            # montage per z: left=exp, right=sim
            Z0, H0, W0 = exp_u16.shape
            montage = np.zeros((Z0, H0, 2 * W0), dtype=np.uint16)
            montage[:, :, :W0] = exp_u16
            montage[:, :, W0:] = sim_u16
            io.imsave(os.path.join(save_dir, f"montage_{cnt}_{name}.tif"), montage, check_contrast=False)

    # --- Final per-bead debug dump (central z only) ---
    # Put debug outputs inside the same phase_retrieval_outputs folder
    im_model.debug_bfp = True
    im_model.debug_every_num_epoch = 1
    im_model.current_epoch = 0
    im_model._last_debug_epoch = None
    im_model.debug_dir = os.path.join(save_dir, "per_bead_phase")

    # number of beads you used in PR (onaxis + offaxis)
    num_beads = len(stacks)
    im_model.debug_max_emitters = num_beads

    # Optional: give names to beads (so folders aren't emitter_000, emitter_001...)
    im_model.debug_names = [name for (name, _, _, _) in stacks]

    # Take ONLY the central z/NFP slice from each bead:
    zi = Z // 2
    idxs = [b * Z + zi for b in range(num_beads)]  # assumes your packing is bead-major then z

    xyz_mid = xyzps[idxs]

    #xyz_mid[:,0]  =  xyz_mid[:,0] * 100
    #xyz_mid[:,1]  =  xyz_mid[:,1] * 100

    NFPs_mid = NFPs[idxs]

    with torch.no_grad():
        _ = im_model(xyz_mid, NFPs_mid, targets=y_true[idxs])  # triggers _maybe_save_debug once

    return phase_mask, g_sigma, ccs


def show_z_psf(param_dict):
    model = ImModel(param_dict)
    model.model_demo(np.linspace(param_dict['zrange'][0], param_dict['zrange'][1], 5))  # check PSFs

def _center_crop(im, out_hw):
    out_h, out_w = out_hw
    h, w = im.shape
    y0 = max(0, (h - out_h) // 2)
    x0 = max(0, (w - out_w) // 2)
    return im[y0:y0 + out_h, x0:x0 + out_w]

def _cc(a, b):
    a = a.astype(np.float32, copy=False).ravel()
    b = b.astype(np.float32, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / den)


def _to_uint16_stack(stack, mode="per_slice_max"):
    st = stack.astype(np.float32, copy=False)

    if mode == "global_max":
        mx = float(st.max())
        if mx <= 0:
            return np.zeros_like(st, dtype=np.uint16)
        return (np.clip(st / mx, 0, 1) * 65535.0).astype(np.uint16)

    # per_slice_max
    out = np.zeros_like(st, dtype=np.uint16)
    mx = st.reshape(st.shape[0], -1).max(axis=1)
    for i in range(st.shape[0]):
        if mx[i] > 0:
            out[i] = (np.clip(st[i] / float(mx[i]), 0, 1) * 65535.0).astype(np.uint16)
    return out


def fit_mask_offset_from_offaxis_stacks(
    param_dict,
    #d_search_um=(00000.0, 70000.0),#d_search_um=(00000.0, 80000.0),
    d_search_um=(00000.0, 70000.0),#d_search_um=(00000.0, 80000.0),
    d_coarse_step_um=5000.0,
    d_fine_step_um=250.0,
    photons_for_sim=1e4,
    save_dir=None,
    save_uint16_mode="per_slice_max",
    make_montage=True,
):
    if not param_dict.get("offaxis_zstack_files"):
        print("[fit d] No off-axis stacks provided. Skipping.")
        return None

    # --- output directory (ONE place only) ---
    if save_dir is None:
        time_now = datetime.today().strftime("%Y%m%d_%H%M%S")
        #save_dir = os.path.join(os.getcwd(), f"mask_offset_fit_{time_now}")
        save_dir = os.path.join(os.getcwd(), f"mask_fit_outputs")
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # NFP sweep from the offset phase_retrieval() already fitted, not a raw/unfit guess
    nfp_offset_um = param_dict["nfp_offset_um"]
    nfp_range_um = param_dict["nfp_range_um"]
    Z_expected = int(param_dict["nfp_fitted_Z"])
    nfp_start_um = nfp_offset_um - nfp_range_um / 2
    nfp_end_um = nfp_offset_um + nfp_range_um / 2
    nfps = np.linspace(nfp_start_um, nfp_end_um, Z_expected, dtype=np.float32)

    # --- model ---
    from DS3Dplus.ds3d_utils import ImModelTraining
    model = ImModelTraining(param_dict)
    model.eval()

    r0, c0 = map(float, param_dict["centralBeadCoordinates_pixel"])
    ps_cam = float(param_dict["ps_camera"])
    M = (param_dict["M"])

    # --- load stacks once ---
    stacks = []
    for f, (rr, cc) in zip(param_dict["offaxis_zstack_files"], param_dict["offaxis_coords_pixel"]):
        zstack = io.imread(f).astype(np.float32)  # (Z,H,W)

        if zstack.shape[0] != Z_expected:
            raise ValueError(f"[fit d] Z mismatch: {f} has Z={zstack.shape[0]} but nfps has {Z_expected}.")

        dx_pix = float(cc)# - c0
        dy_pix = float(rr)# - r0
        x_um = dx_pix * (ps_cam/M)
        y_um = dy_pix * (ps_cam/M)

        stacks.append({
            "file": f,
            "name": os.path.splitext(os.path.basename(f))[0],
            "exp": zstack,
            "H": zstack.shape[1],
            "W": zstack.shape[2],
            "x_um": float(x_um),
            "y_um": float(y_um),
        })


    def simulate_stack_for_d(d_um, st):
        model.mask_offset_in_um = float(d_um)

        Z, Hroi, Wroi = st["exp"].shape[0], st["H"], st["W"]
        sim_stack = np.zeros((Z, Hroi, Wroi), dtype=np.float32)
        cc_per_z = np.zeros((Z,), dtype=np.float32)

        x_um, y_um = st["x_um"], st["y_um"]

        oldNFP = float(model.NFP)

        for zi, nfp_um in enumerate(nfps):
            exp_im = _norm01_sum(st["exp"][zi])
            model.NFP = float(nfp_um)  # scan -> NFP
            xyzp = np.array([x_um, y_um, 0.0, float(photons_for_sim)], dtype=np.float32)

            sim = model.psf_patch_clean(xyzp)
            # IMPORTANT FIX: float32 (prevents Float vs Double mismatch in torch)
            #xyzp = np.array([x_um, y_um, float(z_um), float(photons_for_sim)], dtype=np.float32)

            sim = _center_crop(sim, (Hroi, Wroi))
            sim = _norm01_sum(sim)


            Blur = True
            if Blur:
                g_sigma = param_dict["g_sigma"]
                g_sigma = torch.tensor(g_sigma)
                g_size = 9 #hard coded! to fix
                g_r = int(g_size / 2)
                #g_xs = torch.linspace(-g_r, g_r, g_size, device=device).type(torch.float64)
                g_xs = torch.linspace(-g_r, g_r, g_size).type(torch.float64)
                g_xx, g_yy = torch.meshgrid(g_xs, g_xs, indexing='xy')

                # blur
                # blur (batched)
                blur_kernel = 1 / (2 * math.pi * g_sigma[0] ** 2) * (
                    torch.exp(-0.5 * (g_xx ** 2 + g_yy ** 2) / g_sigma[0] ** 2)   )
                sim_tensor = torch.tensor(sim)
                sim_tensor = F.conv2d(sim_tensor.unsqueeze(0).unsqueeze(0), blur_kernel.type_as(sim_tensor).unsqueeze(0).unsqueeze(0), padding='same' ).squeeze(1)
                '''sim = F.conv2d(
                    sim.unsqueeze(1),
                    blur_kernel.unsqueeze(0).unsqueeze(0).type_as(sim),
                    padding='same'
                ).squeeze(1)'''
                # photon normalization
                #sim = sim / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(         1)  # photon normalization
                # sim = sim[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
                #sim = sim[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
            sim_tensor = sim_tensor.squeeze(0)
            sim_stack[zi] = sim_tensor
            cc_per_z[zi] = _cc(exp_im, sim_tensor.numpy())



        model.NFP = oldNFP  # turning it back to experimental nfp
        return sim_stack, cc_per_z

    def score_for_d(d_um):
        ccs = []
        for st in stacks:
            _, cc_per_z = simulate_stack_for_d(d_um, st)
            ccs.append(cc_per_z)
        ccs = np.concatenate(ccs) if ccs else np.array([-1e9], dtype=np.float32)
        return float(ccs.mean())

    # ---- coarse search ----
    d_search_um = param_dict["mask_offset_in_um"], param_dict["mask_offset_in_um"]+1e-6
    d0, d1 = map(float, d_search_um)

    d_vals = np.arange(d0, d1 + 1e-6, float(d_coarse_step_um), dtype=np.float32)
    scores = [score_for_d(d) for d in d_vals]
    best_d = float(d_vals[int(np.argmax(scores))])

    # ---- fine search ----
    lo = max(d0, best_d - 2 * float(d_coarse_step_um))
    hi = min(d1, best_d + 2 * float(d_coarse_step_um))
    d_vals2 = np.arange(lo, hi + 1e-6, float(d_fine_step_um), dtype=np.float32)
    scores2 = [score_for_d(d) for d in d_vals2]
    best_d2 = float(d_vals2[int(np.argmax(scores2))])
    best_s2 = float(max(scores2))

    # save to param_dict
    param_dict["mask_offset_in_um"] = best_d2
    param_dict["mask_offset_fit_info"] = {
        "best_d_um": best_d2,
        "best_cc": best_s2,
        "coarse": {"d": d_vals.tolist(), "cc": [float(x) for x in scores]},
        "fine": {"d": d_vals2.tolist(), "cc": [float(x) for x in scores2]},
        "save_dir": save_dir,
        "nfps_used": nfps.tolist(),
    }

    print(f"[fit d] best mask_offset_in_um = {best_d2:.1f} um, mean CC={best_s2:.4f}")
    print(f"[fit d] saving outputs to: {save_dir}")

    # ---- save curves ----
    with open(os.path.join(save_dir, "cc_curve_coarse.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d_um", "mean_cc"])
        w.writerows([[float(d), float(s)] for d, s in zip(d_vals, scores)])

    with open(os.path.join(save_dir, "cc_curve_fine.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d_um", "mean_cc"])
        w.writerows([[float(d), float(s)] for d, s in zip(d_vals2, scores2)])

    plt.figure(figsize=(6, 4))
    plt.plot(d_vals, scores, marker="o", linewidth=1)
    plt.plot(d_vals2, scores2, marker="o", linewidth=1)
    plt.axvline(best_d2, linestyle="--")
    plt.xlabel("mask_offset_in_um (d) [um]")
    plt.ylabel("mean CC")
    plt.title(f"Best d = {best_d2:.1f} um, mean CC={best_s2:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cc_curve.png"), dpi=200)
    plt.close()

    # ---- save exp/sim/montage for best d (ALL in same dir) ----
    for st in stacks:
        sim_stack, cc_per_z = simulate_stack_for_d(best_d2, st)  # <-- removed bogus save_results

        with open(os.path.join(save_dir, f"cc_per_z_{st['name']}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["z_um", "cc"])
            w.writerows([[float(z), float(cc)] for z, cc in zip(nfps, cc_per_z)])

        exp_u16 = _to_uint16_stack(st["exp"], mode=save_uint16_mode)
        sim_u16 = _to_uint16_stack(sim_stack, mode=save_uint16_mode)

        io.imsave(os.path.join(save_dir, f"exp_stack_{st['name']}.tif"), exp_u16, check_contrast=False)
        io.imsave(os.path.join(save_dir, f"sim_stack_bestd_{st['name']}.tif"), sim_u16, check_contrast=False)

        if make_montage:
            Z, H, W = exp_u16.shape
            montage = np.zeros((Z, H, 2 * W), dtype=np.uint16)
            montage[:, :, :W] = exp_u16
            montage[:, :, W:] = sim_u16
            io.imsave(os.path.join(save_dir, f"comparison_montage_{st['name']}.tif"), montage, check_contrast=False)

    return best_d2


# ============================================================
# Generate Training Data
# ============================================================

def load_phase_retrieval_results(results_dir: str):
    """Loads the fitted PSF parameters written by phase_retrieval() (results.json +
    phase_mask.npy). Returns None (not an exception) if a Phase Retrieval run hasn't produced
    them yet in this directory."""
    results_path = os.path.join(results_dir, "results.json")
    if not os.path.isfile(results_path):
        return None
    with open(results_path) as f:
        results = json.load(f)
    mask_path = os.path.join(results_dir, results["phase_mask_path"])
    if not os.path.isfile(mask_path):
        return None
    return {
        "phase_mask": np.load(mask_path),
        "g_sigma": float(results["g_sigma"]),
        "d_um": float(results["d_um"]),
        "nfp_offset_um": float(results["nfp_offset_um"]),
        "nfp_range_um": float(results["nfp_range_um"]),
    }


def sample_experimental_frames(folder: str, n_samples: int, attempts: int = 8, base_delay: float = 0.25) -> dict:
    """Lists `folder` (sorted .tif/.tiff filenames only -- cheap) and reads just an evenly-spaced
    SUBSAMPLE of up to n_samples of those files into memory. The real experimental dataset this
    points at lives on the server hosting the GUI and can be huge (thousands of frames), so this
    must never read the whole folder -- only the sampled subset ever touches memory.

    Each sampled file may itself be a multi-page TIFF; those pages become that sample's own Z
    axis (browsed independently per-sample), while the returned T axis indexes across the
    SAMPLED files themselves -- genuine time, unlike treating one file's own Z-slices as if they
    were time. Retries on PermissionError, same rationale as gui.py's _read_tiff_with_retry: a
    file still being written by a live acquisition can be transiently locked."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"'{folder}' is not a folder accessible from this server.")
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff')))
    if not names:
        raise ValueError(f"No .tif/.tiff files found in '{folder}'.")
    n_samples = max(1, min(int(n_samples), len(names)))
    idxs = sorted(set(np.linspace(0, len(names) - 1, n_samples).round().astype(int).tolist()))
    sampled_names = [names[i] for i in idxs]

    arrays = []
    ref_hw = None
    for name in sampled_names:
        path = os.path.join(folder, name)
        arr, last_exc = None, None
        for attempt in range(attempts):
            try:
                arr = tifffile.imread(path)
                break
            except PermissionError as exc:
                last_exc = exc
                time.sleep(base_delay * (attempt + 1))
        if arr is None:
            raise last_exc
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            raise ValueError(f"'{name}': expected a 2D or 3D (Z,H,W) TIFF, got shape {arr.shape}.")
        if ref_hw is None:
            ref_hw = arr.shape[1:]
        elif arr.shape[1:] != ref_hw:
            raise ValueError(
                f"'{name}' is {arr.shape[1:]}, but the first sampled frame '{sampled_names[0]}' "
                f"is {ref_hw} -- all sampled frames must be the same size."
            )
        arrays.append(arr.astype(np.float32))

    vmin = min(float(a.min()) for a in arrays)
    vmax = max(float(a.max()) for a in arrays)
    return {
        "arrays": arrays, "names": sampled_names, "vmin": vmin, "vmax": vmax,
        "T": len(arrays), "total_files_in_folder": len(names),
    }


def noise_patch_stats(frame: np.ndarray, bbox):
    """Mean/std of pixel values inside a user-marked no-emitter rectangle, bbox=(r0, r1, c0, c1)."""
    r0, r1, c0, c1 = bbox
    patch = frame[r0:r1, c0:c1]
    return float(patch.mean()), float(patch.std())


def temporal_noise_baseline(stack: np.ndarray, bbox):
    """Mean/std of the single darkest-mean pixel within bbox, across every frame of `stack`
    (Z,H,W) -- mirrors the AutoDS3D/root pipeline's mu_std_p(): isolates genuine per-pixel
    temporal (read) noise from one fixed pixel's value over time, rather than conflating it
    with spatial pixel-to-pixel non-uniformity (gain/vignetting) across a patch, as a single-
    frame spatial mean/std would. Assumes that pixel is background in every frame."""
    r0, r1, c0, c1 = bbox
    region = stack[:, r0:r1, c0:c1].astype(np.float64)
    mean_map = region.mean(axis=0)
    r_idx, c_idx = np.unravel_index(np.argmin(mean_map), mean_map.shape)
    bg_trace = region[:, r_idx, c_idx]
    return float(bg_trace.mean()), float(bg_trace.std())


def temporal_peak(stack: np.ndarray, bbox) -> float:
    """Max pixel value within bbox across every frame of `stack` (Z,H,W) -- catches a blinking
    emitter at its brightest automatically, instead of requiring the single best Z-slice to
    already be the one marked."""
    r0, r1, c0, c1 = bbox
    return float(stack[:, r0:r1, c0:c1].max())


def estimate_signal_range(param_dict: dict, baseline_mu: float, exp_maxv: float, photon_count: float = 1e4):
    """Back-solves an (Nsig_range) photon-count range from a real experimental emitter's peak
    brightness: render a reference on-axis emitter at a known photon_count (mid-z), and compare
    its simulated peak pixel value to the real observed peak-above-baseline
    (exp_maxv - baseline_mu, from a user-marked bright-emitter patch vs. a user-marked
    no-emitter patch) to solve for the true photon count. Returns (sig_min, sig_max), rounded to
    the nearest 1000 like root's func3.

    Deliberately NOT root's mu_std_p() formula (p = photon_count / (sim_peak + mu) * exp_maxv):
    that adds the baseline to the simulated (noise-free) reference peak before dividing, which
    only cancels correctly when photon_count happens to be close to the true photon count being
    solved for -- otherwise it's biased (verified: off by ~4.85x in a synthetic case with
    true_photons=2000, photon_count=1e4, baseline comparable to the true peak-above-baseline).
    Since this tab has the user mark both a no-emitter AND a bright-emitter patch (unlike root's
    GUI, which only marks one ROI), baseline can be subtracted from both the real and simulated
    sides before scaling, which is exact regardless of photon_count's value:
        (exp_maxv - baseline_mu) / true_photons == sim_peak(photon_count) / photon_count
        => true_photons = (exp_maxv - baseline_mu) * photon_count / sim_peak(photon_count)
    """
    model = ImModelBase(param_dict)
    zmin, zmax = param_dict['zrange']
    xyzp = np.array([[0.0, 0.0, (zmin + zmax) / 2.0, photon_count]], dtype=np.float32)
    xyzps = torch.from_numpy(xyzp).to(param_dict['device'])
    sim_peak = float(model.get_psfs(xyzps).detach().cpu().numpy().max())
    if sim_peak <= 0:
        raise ValueError("simulated reference PSF peak is non-positive -- check the fitted phase mask/d")
    p = (exp_maxv - baseline_mu) * photon_count / sim_peak
    if p <= 0:
        raise ValueError(
            f"estimated photon count is non-positive ({p:.1f}) -- the marked bright-emitter "
            f"patch's peak ({exp_maxv:.1f}) isn't brighter than the marked baseline ({baseline_mu:.1f})"
        )
    # Rounding to the nearest integer photon (not the nearest 1000, as root's func3 does) --
    # real single-molecule photon counts are often in the hundreds, and round(0.5*p/1e3)*1e3
    # collapses anything below ~1000 photons to 0, which then makes Sampling.__init__ divide by
    # zero (`blob_maxv / Nsig_range[1]`). p is already guarded > 0 above, so rounding to the
    # nearest integer can never produce exactly 0 for a genuinely positive signal.
    return round(0.5 * p), round(1.1 * p)


def _simulate_one_frame(model, sampling, param_dict):
    """One simulated training frame: random emitters -> pasted clean PSF patches -> Poisson
    shot noise + dark offset -> bit-depth clip. Ported from the root pipeline's
    training_data_func, but sized from each patch's own returned shape rather than model.N (the
    optics-derived internal simulation grid, which can differ from the requested (H, W))."""
    xyzps, xyz_ids, blob3d = sampling.xyzp_batch()
    H, W = param_dict['H'], param_dict['W']
    ps_xy = param_dict['ps_camera'] / param_dict['M']
    canvas = np.zeros((H, W), dtype=np.float32)

    for k in range(xyzps.shape[0]):
        x_um, y_um = xyzps[k, 0], xyzps[k, 1]
        c = int(round(x_um / ps_xy + (W - 1) / 2))
        r = int(round(y_um / ps_xy + (H - 1) / 2))
        patch = model.psf_patch_clean(xyzps[k].astype(np.float32))
        ph, pw = patch.shape
        # ph//2 elements before the center row, ph-ph//2 at/after it -- NOT symmetric ("+1")
        # for an even ph. The old `rr1 = r + pr + 1` assumed an odd patch size (true 121x121
        # default); for an even canvas_size_px (e.g. 122) psf_patch_clean's crop is even-sized
        # too, and that "+1" overshoots the patch by one row/col in EVERY placement (not just
        # near edges), so numpy silently truncates the source slice and the +=  shapes mismatch.
        pr_lo, pc_lo = ph // 2, pw // 2
        rr0, rr1 = max(0, r - pr_lo), min(H, r + (ph - pr_lo))
        cc0, cc1 = max(0, c - pc_lo), min(W, c + (pw - pc_lo))
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        pr0, pc0 = rr0 - (r - pr_lo), cc0 - (c - pc_lo)
        canvas[rr0:rr1, cc0:cc1] += patch[pr0:pr0 + (rr1 - rr0), pc0:pc0 + (cc1 - cc0)]

    bg_lo, bg_hi = param_dict['shot_noise_background_range']
    off_lo, off_hi = param_dict['noise_offset_range']
    background = float(np.random.uniform(bg_lo, bg_hi)) ** 2
    offset = float(np.random.uniform(off_lo, off_hi))
    im = np.abs(np.random.poisson(canvas + background) + offset - background)
    im = np.clip(im, 0, 2 ** param_dict['bitdepth'] - 1).astype(np.uint16)
    return im, xyz_ids, blob3d


def generate_training_frame(param_dict: dict) -> np.ndarray:
    """Renders a single sample simulated training frame, for the GUI's live preview."""
    model = ImModelTraining(param_dict)
    sampling = Sampling(param_dict)
    im, _, _ = _simulate_one_frame(model, sampling, param_dict)
    return im


def generate_training_data(param_dict: dict, out_dir: str, n_ims: int, stop_event=None) -> None:
    """Generates n_ims simulated training frames + ground truth into out_dir/{x/, y.pickle,
    param.pickle}, mirroring the root pipeline's training_data_func output format. Unlike root
    (which unconditionally deletes out_dir first), this never deletes a pre-existing folder --
    an interactive GUI where the user can type an arbitrary path must not silently wipe it.
    A rerun into a non-empty out_dir continues frame numbering after the highest existing
    index and merges into the existing y.pickle, rather than restarting at 0 and silently
    overwriting/orphaning earlier frames and their labels."""
    x_dir = os.path.join(out_dir, "x")
    os.makedirs(x_dir, exist_ok=True)
    y_pickle_path = os.path.join(out_dir, "y.pickle")

    existing_indices = [
        int(os.path.splitext(f)[0]) for f in os.listdir(x_dir)
        if os.path.splitext(f)[1].lower() in ('.tif', '.tiff') and os.path.splitext(f)[0].isdigit()
    ]
    start_i = max(existing_indices) + 1 if existing_indices else 0

    H, W = param_dict['H'], param_dict['W']
    new_metadata = {
        'volume_size': (param_dict['D'], param_dict['HH'], param_dict['WW']),
        'us_factor': param_dict['us_factor'],
        'blob_r': param_dict['blob_r'],
        'blob_maxv': param_dict['blob_maxv'],
        'tile_grid': (1, 1),
        'camera_size_px': (H, W),
    }

    if existing_indices and os.path.isfile(y_pickle_path):
        with open(y_pickle_path, "rb") as f:
            labels_dict = pickle.load(f)
        # Earlier frames' xyz_ids/blob3d were computed under whatever settings produced THIS
        # metadata -- silently overwriting it with the current run's settings would desync
        # those labels from the volume they actually describe (and a differing camera_size_px
        # means the earlier frames are even a different pixel size than the new ones). Refuse
        # rather than silently corrupt; the user can pick a fresh output folder instead.
        existing_metadata = {k: labels_dict.get(k) for k in new_metadata}
        if existing_metadata != new_metadata:
            raise ValueError(
                f"{out_dir} already has {len(existing_indices)} frame(s) generated with "
                f"different settings than the current configuration -- appending here would "
                f"desync their ground truth.\n  existing: {existing_metadata}\n  current:  {new_metadata}\n"
                f"Use a different output folder, or regenerate this one from scratch."
            )
        print(f"[TD] {x_dir} already has {len(existing_indices)} frame(s) with matching settings -- "
              f"appending new frames starting at {start_i:05d}.tif.")
    elif existing_indices:
        labels_dict = {}
        print(f"[TD] WARNING: {x_dir} has {len(existing_indices)} existing frame(s) but no "
              f"y.pickle was found -- their ground truth is not recoverable. New frames will "
              f"still be appended starting at {start_i:05d}.tif with their own labels.")
    else:
        labels_dict = {}

    model = ImModelTraining(param_dict)
    sampling = Sampling(param_dict)
    labels_dict.update(new_metadata)

    n_ims = int(n_ims)
    written = 0
    for k in range(n_ims):
        if stop_event is not None and stop_event.is_set():
            print(f"[TD] stopped at frame {k}/{n_ims}")
            break
        im, xyz_ids, blob3d = _simulate_one_frame(model, sampling, param_dict)
        fname = f"{start_i + k:05d}.tif"
        io.imsave(os.path.join(x_dir, fname), im, check_contrast=False)
        labels_dict[fname] = (xyz_ids, blob3d)
        written += 1
        if k % 100 == 0:
            print(f"[TD] training image [{k} / {n_ims}]")

    with open(y_pickle_path, "wb") as f:
        pickle.dump(labels_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "param.pickle"), "wb") as f:
        pickle.dump(param_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[TD] done. wrote {written} frames to {out_dir} "
          f"(index {start_i:05d}-{start_i + max(written, 1) - 1:05d}).")


# ============================================================
# Train Model
# ============================================================

def maybe_build_x_memmap(td_folder, force_rebuild=False):
    """
    Build or reuse a memmap cache for training_data/x/*.tif.
    Keeps TIFFs on disk for debugging, but training can read from one binary file.

    Returns:
        cache_info: dict with keys enabled, data_path, shape, dtype, ids
    """
    x_folder = os.path.join(td_folder, 'x')
    data_path = os.path.join(td_folder, 'x_memmap.dat')
    ids_path = os.path.join(td_folder, 'x_ids.npy')

    ids = sorted([f for f in os.listdir(x_folder) if f.lower().endswith('.tif')])
    if len(ids) == 0:
        raise RuntimeError(f'No TIFF files found in {x_folder}')

    first_im = io.imread(os.path.join(x_folder, ids[0]))
    H, W = first_im.shape
    dtype = first_im.dtype

    rebuild = force_rebuild
    if (not os.path.exists(data_path)) or (not os.path.exists(ids_path)):
        rebuild = True
    else:
        try:
            cached_ids = np.load(ids_path, allow_pickle=True).tolist()
            if cached_ids != ids:
                rebuild = True
            else:
                expected_bytes = len(ids) * H * W * np.dtype(dtype).itemsize
                actual_bytes = os.path.getsize(data_path)
                if actual_bytes != expected_bytes:
                    rebuild = True
        except Exception:
            rebuild = True

    if rebuild:
        print(f'[memmap] building cache from TIFFs in {x_folder}')
        X = np.memmap(data_path, mode='w+', dtype=dtype, shape=(len(ids), H, W))
        for i, fname in enumerate(ids):
            if i % 1000 == 0:
                print(f'[memmap] packing [{i} / {len(ids)}]')
            im = io.imread(os.path.join(x_folder, fname))
            if im.shape != (H, W):
                raise ValueError(f'Image shape mismatch for {fname}: {im.shape} vs {(H, W)}')
            if im.dtype != dtype:
                raise ValueError(f'Image dtype mismatch for {fname}: {im.dtype} vs {dtype} -- '
                                  f'packing it into the shared memmap would silently cast/rescale its values.')
            X[i] = im
        X.flush()
        np.save(ids_path, np.array(ids, dtype=object), allow_pickle=True)
        print(f'[memmap] cache saved: {data_path}')
    else:
        print(f'[memmap] reusing existing cache: {data_path}')

    return dict(enabled=True, data_path=data_path, shape=(len(ids), H, W),
                dtype=np.dtype(dtype).str, ids=ids)


def load_training_data_metadata(td_folder: str):
    """Loads <td_folder>/param.pickle + y.pickle, written by generate_training_data(). Returns
    None (not an exception) if Generate Training Data hasn't produced them yet, or x/ is empty."""
    param_path = os.path.join(td_folder, "param.pickle")
    y_path = os.path.join(td_folder, "y.pickle")
    x_dir = os.path.join(td_folder, "x")
    if not (os.path.isfile(param_path) and os.path.isfile(y_path) and os.path.isdir(x_dir)):
        return None
    if not any(f.lower().endswith(('.tif', '.tiff')) for f in os.listdir(x_dir)):
        return None
    with open(param_path, "rb") as f:
        param_dict = pickle.load(f)
    with open(y_path, "rb") as f:
        labels = pickle.load(f)
    return {"param_dict": param_dict, "labels": labels}


_TRAINING_DATA_LABEL_METADATA_KEYS = {
    'volume_size', 'us_factor', 'blob_r', 'blob_maxv', 'tile_grid', 'camera_size_px',
}


def check_training_data_folder(td_folder: str, meta: dict) -> list:
    """Minimal sanity checks that td_folder's actual on-disk files match what its own
    param.pickle/y.pickle claim -- catches a folder copied/moved from a different run (wrong
    image size), pointed at the wrong path, or only partially transferred (missing frames),
    with a clear message instead of an opaque failure deep inside the DataLoader once training
    starts. Cheap by design (a set comparison over filenames + reading exactly one sample
    image), not a full per-file validation. Returns a list of warning strings; empty means
    nothing suspicious was found. Never raises -- a failure here should not block loading,
    only warn."""
    warnings = []
    labels = meta['labels']
    x_dir = os.path.join(td_folder, 'x')

    try:
        entries = os.listdir(x_dir)
    except OSError as exc:
        return [f"Could not list {x_dir}: {exc}"]

    tif_files = sorted(f for f in entries if f.lower().endswith(('.tif', '.tiff')))
    other_files = [f for f in entries if os.path.isfile(os.path.join(x_dir, f))
                   and not f.lower().endswith(('.tif', '.tiff'))]
    if other_files:
        warnings.append(f"{len(other_files)} non-TIFF file(s) in x/ (e.g. '{other_files[0]}') -- ignored.")

    expected_frames = set(labels.keys()) - _TRAINING_DATA_LABEL_METADATA_KEYS
    on_disk = set(tif_files)
    missing = expected_frames - on_disk
    extra = on_disk - expected_frames
    if missing:
        sample = ', '.join(sorted(missing)[:3])
        warnings.append(f"{len(missing)} frame(s) listed in y.pickle are missing from x/ (e.g. {sample}).")
    if extra:
        sample = ', '.join(sorted(extra)[:3])
        warnings.append(f"{len(extra)} .tif file(s) in x/ have no matching entry in y.pickle (e.g. {sample}) -- ignored.")

    if tif_files:
        sample_name = tif_files[0]
        try:
            sample_im = io.imread(os.path.join(x_dir, sample_name))
        except Exception as exc:
            warnings.append(f"Could not read '{sample_name}' to check its size: {exc}")
        else:
            expected_hw = labels.get('camera_size_px')
            if expected_hw is not None and tuple(sample_im.shape) != tuple(expected_hw):
                warnings.append(
                    f"Image size mismatch: '{sample_name}' is {tuple(sample_im.shape)}, but "
                    f"y.pickle's camera_size_px is {tuple(expected_hw)}."
                )

    return warnings


def _make_post_epoch_fn(live_box, validate_ds, param_dict, t0, sample_viz_every_epochs):
    """Builds the per-epoch callback passed as Trainer.fit(post_epoch_fn=...). Tracks loss/LR
    history and, every sample_viz_every_epochs epochs (or on a new best), renders a fixed
    validation tile's predicted-vs-ground-truth max-projection (+ a best-effort Jaccard/RMSE
    readout) into live_box for the GUI's live monitor. Never raises -- a failure here must not
    interrupt training."""
    device = param_dict['device']
    train_loss_history, test_loss_history, lr_history = [], [], []

    # Defensive even though train_model() already passes a non-empty dataset here (falling
    # back to a train-partition sample when the validation split is empty) -- this function's
    # own docstring promises "never raises", so guard again rather than rely solely on the
    # caller getting that right.
    if len(validate_ds) > 0:
        viz_x, viz_y = validate_ds[0]
    else:
        viz_x = viz_y = None
    volume2xyz = None
    try:
        volume2xyz = Volume2XYZ(params={
            'blob_r': param_dict['blob_r'], 'vs_xy': param_dict['vs_xy'],
            'vs_z': param_dict['vs_z'], 'zrange': param_dict['zrange'],
            'threshold': param_dict['threshold'], 'device': device,
        })
    except Exception as exc:
        print(f"[TRAIN] WARNING: could not build Volume2XYZ for the debug panel: {exc}")

    def post_epoch_fn(epoch, total_epochs, train_loss, test_loss, is_best, best_metric,
                       epochs_without_improvement, lr, model):
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        lr_history.append(lr)

        snapshot = {
            'epoch': epoch, 'total_epochs': total_epochs,
            'train_loss_history': list(train_loss_history), 'test_loss_history': list(test_loss_history),
            'lr_history': list(lr_history), 'lr': lr,
            'best_metric': best_metric, 'epochs_without_improvement': epochs_without_improvement,
            'elapsed_s': time.time() - t0,
        }
        avg_epoch_s = snapshot['elapsed_s'] / max(epoch, 1)
        snapshot['eta_s'] = avg_epoch_s * max(total_epochs - epoch, 0)

        if viz_x is not None and (is_best or (sample_viz_every_epochs > 0 and epoch % sample_viz_every_epochs == 0)):
            try:
                model.eval()
                with torch.no_grad():
                    x_t = torch.from_numpy(viz_x).unsqueeze(0).to(device)
                    pred = model(x_t)
                snapshot['pred_proj'] = pred[0].max(dim=0).values.cpu().numpy()
                snapshot['target_proj'] = viz_y.max(axis=0)
                snapshot['viz_epoch'] = epoch

                if volume2xyz is not None:
                    xyz_rec, _ = volume2xyz(pred)
                    xyz_ids = np.asarray(param_dict['_viz_xyz_ids'])
                    WW, HH = param_dict['WW'], param_dict['HH']
                    x_gt = (xyz_ids[:, 0] - (WW - 1) / 2) * param_dict['vs_xy']
                    y_gt = (xyz_ids[:, 1] - (HH - 1) / 2) * param_dict['vs_xy']
                    z_gt = (xyz_ids[:, 2] + 0.5) * param_dict['vs_z'] + param_dict['zrange'][0]
                    xyz_gt = np.c_[x_gt, y_gt, z_gt]
                    if xyz_rec is not None and len(xyz_rec) > 0 and len(xyz_gt) > 0:
                        jacc, rmse_xy, rmse_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.1)
                        snapshot['sample_jaccard'] = jacc
                        snapshot['sample_rmse_xy'] = rmse_xy
                        snapshot['sample_rmse_z'] = rmse_z
            except Exception as exc:
                print(f"[TRAIN] WARNING: sample-viz/Jaccard readout failed this epoch: {exc}")
            finally:
                model.train()

        if live_box is not None:
            # Bump version as part of the SAME snapshot dict rather than a second, separate
            # live_box mutation afterward -- the GUI polling thread reads live_box concurrently,
            # and a reader landing between two separate top-level writes could see a torn state
            # (e.g. train_loss_history already updated but test_loss_history not yet, or version
            # bumped before the new histories are actually in place). One dict with everything,
            # applied via one update() call, shrinks that window.
            snapshot['version'] = live_box.get('version', 0) + 1
            live_box.update(snapshot)

    return post_epoch_fn


def train_model(param_dict: dict, training_dict: dict, live_box: dict = None, stop_event=None):
    """Trains a localization CNN (LON) on data written by generate_training_data(), faithfully
    porting the root pipeline's training_func(): same MyDataset/LON/KDE_loss3D/Adam+
    ReduceLROnPlateau/checkpointing/resume/90-10 sequential split behavior. Adds stop_event
    support and a live_box progress-snapshot mechanism (see _make_post_epoch_fn) on top --
    neither changes the training math itself.
    Returns (net_file, fit_file), matching root's own training_func()."""
    np.random.seed(training_dict['numpy_seed'])
    torch.manual_seed(training_dict['torch_seed'])

    device = torch.device(param_dict['device'])
    print(f'device used (train_model): {device}')
    torch.backends.cudnn.benchmark = True

    td_folder = param_dict['td_folder']
    path_save = param_dict['path_save']
    os.makedirs(path_save, exist_ok=True)

    batch_size = training_dict['batch_size']
    lr = training_dict['lr']
    num_epochs = training_dict['num_epochs']
    num_workers = training_dict['num_workers']

    params_train = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=(device.type == 'cuda'))
    if num_workers > 0:
        params_train.update(persistent_workers=True, prefetch_factor=4)
    params_validate = dict(params_train, shuffle=False)

    x_folder = os.path.join(td_folder, 'x')
    x_cache = maybe_build_x_memmap(td_folder, force_rebuild=False)
    x_list = list(x_cache['ids'])
    num_x = len(x_list)

    with open(os.path.join(td_folder, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    if training_dict['shuffle_split']:
        rng = np.random.RandomState(training_dict['numpy_seed'])
        x_list = list(x_list)
        rng.shuffle(x_list)

    split = training_dict['train_val_split']
    partition = {'train': x_list[:int(num_x * split)], 'validate': x_list[int(num_x * split):]}
    train_ds = MyDataset(x_folder, partition['train'], labels, cache_info=x_cache)
    validate_ds = MyDataset(x_folder, partition['validate'], labels, cache_info=x_cache)

    train_dl = DataLoader(train_ds, **params_train)
    validate_dl = DataLoader(validate_ds, **params_validate)

    D, us_factor, maxv = labels['volume_size'][0], labels['us_factor'], labels['blob_maxv']

    resume_file = training_dict['resume_net_file']
    if resume_file == 'None':
        resume_file = None

    model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)
    optimizer = Adam(list(model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, min_lr=1e-6)

    start_epoch = 0
    best_metric = None
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    resume_checkpoint = None

    if resume_file is not None:
        ckpt_path = os.path.join(path_save, resume_file)
        # weights_only=False: this checkpoint is the app's own locally-created file (contains a
        # raw LON module instance under 'net', not just tensors), and PyTorch 2.6+ defaults
        # torch.load to weights_only=True, which would otherwise refuse to unpickle it.
        resume_checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        state_dict = resume_checkpoint.get('model_state_dict', resume_checkpoint.get('state_dict'))
        if state_dict is None:
            raise ValueError(
                f"{ckpt_path} doesn't look like a checkpoint this app wrote -- missing both "
                f"'model_state_dict' and 'state_dict'. Point 'Resume from checkpoint' at a "
                f"net_*.pt/last_net_*.pt file from this Train Model tab."
            )
        model.load_state_dict(state_dict)

        if resume_checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr  # force the new run's LR, matching root's deliberate resume behavior
        if resume_checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])

        start_epoch = int(resume_checkpoint.get('epoch', 0))
        best_metric = resume_checkpoint.get('best_metric', None)
        epochs_without_improvement = int(resume_checkpoint.get('epochs_without_improvement', 0))
        history = resume_checkpoint.get('fit_history', history)

        rng_state = resume_checkpoint.get('torch_rng_state', None)
        if rng_state is not None:
            try:
                if isinstance(rng_state, torch.Tensor):
                    rng_state = rng_state.detach().cpu()
                    if rng_state.dtype != torch.uint8:
                        rng_state = rng_state.to(torch.uint8)
                    torch.set_rng_state(rng_state)
                elif isinstance(rng_state, np.ndarray):
                    torch.set_rng_state(torch.from_numpy(rng_state.astype(np.uint8)))
                elif isinstance(rng_state, (list, tuple)):
                    torch.set_rng_state(torch.tensor(rng_state, dtype=torch.uint8))
                else:
                    print(f"[resume] skipping torch RNG restore: unsupported type {type(rng_state)}")
            except Exception as e:
                print(f"[resume] skipping torch RNG restore: {e}")

        np_state = resume_checkpoint.get('numpy_rng_state', None)
        if np_state is not None:
            try:
                np.random.set_state(np_state)
            except Exception as e:
                print(f"[resume] skipping NumPy RNG restore: {e}")

        cuda_state = resume_checkpoint.get('cuda_rng_state_all', None)
        if torch.cuda.is_available() and cuda_state is not None:
            try:
                fixed_cuda_state = []
                for st in cuda_state:
                    if isinstance(st, torch.Tensor):
                        st = st.detach().cpu()
                        if st.dtype != torch.uint8:
                            st = st.to(torch.uint8)
                    elif isinstance(st, np.ndarray):
                        st = torch.from_numpy(st.astype(np.uint8))
                    elif isinstance(st, (list, tuple)):
                        st = torch.tensor(st, dtype=torch.uint8)
                    else:
                        raise TypeError(f"unsupported CUDA RNG state type: {type(st)}")
                    fixed_cuda_state.append(st)
                torch.cuda.set_rng_state_all(fixed_cuda_state)
            except Exception as e:
                print(f"[resume] skipping CUDA RNG restore: {e}")

        print(f'[resume] loaded full training state: {ckpt_path}')
        print(f'[resume] continuing from epoch {start_epoch}')
    else:
        print('[resume] starting from scratch')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of trainable parameters: {n_params}')

    tv_z_weight = 0  # dead/inert in root (multiplied by 0 there too) -- ported faithfully as-is
    if param_dict['us_factor'] == 1:
        my_loss_func = KDE_loss3D(sigma=1.0, device=device, tv_z_weight=tv_z_weight)
    else:
        my_loss_func = KDE_loss3D(sigma=0.5 * (param_dict['us_factor'] / 2), device=device,
                                   tv_z_weight=tv_z_weight)

    trainer = TorchTrainer(model, my_loss_func, optimizer, lr_scheduler=scheduler, device=device)

    if resume_checkpoint is not None:
        best_file_path = resume_checkpoint.get('file_name', None)
        last_file_path = resume_checkpoint.get('last_file_name', None)
        if best_file_path is None:
            time_now = datetime.today().strftime('%m-%d_%H-%M')
            net_file = 'net_' + time_now + '.pt'
            best_file_path = os.path.join(path_save, net_file)
        else:
            net_file = os.path.basename(best_file_path)
        if last_file_path is None:
            last_net_file = ('last_' + net_file if net_file.startswith('net_')
                              else 'last_net_' + datetime.today().strftime('%m-%d_%H-%M') + '.pt')
            last_file_path = os.path.join(path_save, last_net_file)
        else:
            last_net_file = os.path.basename(last_file_path)
    else:
        time_now = datetime.today().strftime('%m-%d_%H-%M')
        net_file = 'net_' + time_now + '.pt'
        last_net_file = 'last_net_' + time_now + '.pt'
        best_file_path = os.path.join(path_save, net_file)
        last_file_path = os.path.join(path_save, last_net_file)

    checkpoints = dict(
        file_name=best_file_path, last_file_name=last_file_path,
        net=Net(D=D, us_factor=us_factor, maxv=maxv), state_dict=None,
        note='resume-capable checkpoint',
    )

    # cache one validation tile's GT voxel-index positions for the debug panel's Jaccard readout
    # -- fall back to a train-partition sample when the validation split is empty, and use the
    # matching DATASET object too (previously only viz_id had this fallback; _make_post_epoch_fn
    # was always handed validate_ds, which would itself be empty and crash on validate_ds[0]).
    param_dict = dict(param_dict)
    if partition['validate']:
        viz_id, viz_ds = partition['validate'][0], validate_ds
    else:
        viz_id, viz_ds = partition['train'][0], train_ds
    param_dict['_viz_xyz_ids'] = labels[viz_id][0]

    t0 = time.time()
    post_epoch_fn = _make_post_epoch_fn(
        live_box, viz_ds, param_dict, t0,
        training_dict['sample_viz_every_epochs'],
    )
    fit_results = trainer.fit(
        train_dl, validate_dl, num_epochs=num_epochs, checkpoints=checkpoints,
        early_stopping=training_dict['early_stopping'],
        start_epoch=start_epoch, history=history, best_metric=best_metric,
        epochs_without_improvement=epochs_without_improvement,
        post_epoch_fn=post_epoch_fn, stop_event=stop_event,
    )

    fit_stamp = datetime.today().strftime('%m-%d_%H-%M')
    fit_file = 'fit_' + fit_stamp + '.pickle'
    with open(os.path.join(path_save, fit_file), 'wb') as handle:
        pickle.dump(fit_results, handle)

    t1 = time.time()
    print(f'training results in {net_file}, {last_net_file} and {fit_file}')
    print(f'finished training in {t1 - t0}s.')

    return net_file, fit_file
