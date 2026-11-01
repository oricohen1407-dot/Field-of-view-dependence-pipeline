import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from DS3Dplus.ds3d_utils import asm_propagate


def simulation_grid_size(f_4f, lamda, ps_camera, ps_BFP) -> int:
    N = np.floor(f_4f * lamda / (ps_camera * ps_BFP))
    return int(N + 1 - (N % 2))


def _logit_init(x_init, lo, hi, eps=1e-3):
    """Map an initial value into an unconstrained logit parameter, bounded to [lo, hi]."""
    p = (x_init - lo) / (hi - lo + 1e-12)
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class ImModel_pr(torch.nn.Module):
    def __init__(self, params):
        """
        a scalar model for air or oil objective in microscopy
        """
        super().__init__()
        self.device = params['device']
        self.M = params['M']  # magnification
        self.NA = params['NA']
        self.n_immersion = params['n_immersion']  # refractive index of the immersion of the objective
        self.lamda = params['lamda']  # wavelength
        self.n_sample = params['n_sample']   # refractive index of the sample
        self.f_4f = params['f_4f']  # focal length of 4f system
        self.ps_camera = params['ps_camera']  # pixel size of the camera
        self.ps_BFP = params['ps_BFP']  # pixel size at back focal plane
        
        # ---- learnable d (mask displacement) ----
        self.d_min_um = params["d_min_um"]
        self.d_max_um = params["d_max_um"]
        d_init = params["mask_offset_in_um"]

        self.centralBeadCoordinates_pixel = list(params['centralBeadCoordinates_pixel'])

        # map initial d into an unconstrained parameter via inverse-sigmoid (logit)
        eps = 1e-3
        p = (d_init - self.d_min_um) / (self.d_max_um - self.d_min_um + 1e-12)
        p = min(max(p, eps), 1.0 - eps)
        d_raw_init = math.log(p / (1.0 - p))

        self.d_raw = torch.nn.Parameter(torch.tensor(d_raw_init, device=self.device, dtype=torch.float32))

        # ---- learnable NFP center offset (range length is fixed, not fitted) ----
        self.nfp_range_um = float(params["nfp_range_um"])
        if self.nfp_range_um <= 0:
            raise ValueError(f"nfp_range_um must be positive; got {self.nfp_range_um}")
        self.nfp_offset_min_um = params["nfp_offset_min_um"]
        self.nfp_offset_max_um = params["nfp_offset_max_um"]
        self.Z = int(params["Z"])
        if self.Z < 2:
            raise ValueError(f"ImModel_pr requires Z >= 2 to fit an NFP sweep; got Z={self.Z}")

        nfp_offset_init = params["nfp_offset_init_um"]
        nfp_offset_raw_init = _logit_init(nfp_offset_init, self.nfp_offset_min_um, self.nfp_offset_max_um)
        self.nfp_offset_raw = torch.nn.Parameter(torch.tensor(nfp_offset_raw_init, device=self.device, dtype=torch.float32))
        # image
        H, W = params['H'], params['W']  # FOV size
        g_size = 9  # size of the gaussian blur kernel
        g_sigma = params['g_sigma']  # std of the gaussian blur kernel
        ###################

        N = simulation_grid_size(self.f_4f, self.lamda, self.ps_camera, self.ps_BFP)
        print(f'Simulation size of the imaging model is {N} which must be larger than image size (PSF z-stack and training images)!')

        # pupil/aperture at back focal plane
        d_pupil = 2 * self.f_4f * self.NA / np.sqrt(self.M ** 2 - self.NA ** 2)  # diameter [um]
        #print('d_pupil = ' + str(d_pupil))
        pn_pupil = d_pupil / self.ps_BFP  # pixel number of the pupil diameter should be smaller than the simulation size N
        if N < pn_pupil:
            raise Exception('Simulation size is smaller than the pupil!')
        # cartesian and polar grid in BFP
        x_phys = np.linspace(-N / 2, N / 2, N) * self.ps_BFP
        xi, eta = np.meshgrid(x_phys, x_phys)  # cartesian physical coordinates
        r_phys = np.sqrt(xi ** 2 + eta ** 2)
        pupil = (r_phys < d_pupil / 2).astype(np.float32)

        x_ang = np.linspace(-1, 1, N) * (N / pn_pupil) * (self.NA / self.n_immersion)  # angular coordinate
        xx_ang, yy_ang = np.meshgrid(x_ang, x_ang)
        r = np.sqrt(
            xx_ang ** 2 + yy_ang ** 2)  # normalized angular coordinates, s.t. r = NA/n_immersion at edge of E field support

        k_immersion = 2 * math.pi * self.n_immersion / self.lamda  # [1/um]
        sin_theta_immersion = r

        circ_scale = float(params.get("circ_scale", 1.0))  # 1.0 = default behavior
        r_lim = (self.NA / self.n_immersion)
        r_lim_scaled = (self.NA / self.n_immersion) * circ_scale
        #r_lim_scaled = circ_scale
        print("rlim_scaled = " + str(r_lim_scaled))
        print("r_lim = " + str(r_lim))
        r_lim = min(r_lim, 1.0)  # can't exceed sin(theta)=1
        #r_lim_scaled = min(r_lim, 1.0)

        circ_NA = (sin_theta_immersion < r_lim).astype(np.float32)
        circ_NA_scaled = (sin_theta_immersion < r_lim_scaled).astype(np.float32)  # 28/01/2026

        cos_theta_immersion = np.sqrt(1 - (sin_theta_immersion * circ_NA) ** 2) * circ_NA

        k_sample = 2 * math.pi * self.n_sample / self.lamda
        sin_theta_sample = self.n_immersion / self.n_sample * sin_theta_immersion
        # note: when circ_sample is smaller than circ_NA, super angle fluorescence apears
        circ_sample = (sin_theta_sample < 1).astype(np.float32)  # if all the frequency of the sample can be captured
        cos_theta_sample = np.sqrt(1 - (sin_theta_sample * circ_sample) ** 2) * circ_sample * circ_NA

        # circular aperture to impose on BFP, SAF is excluded
        circ = circ_NA * circ_sample
        circ_scaled = circ_NA_scaled

        pn_circ = np.floor(np.sqrt(np.sum(circ) / math.pi) * 2)
        pn_circ = int(pn_circ + 1 - (pn_circ % 2))
        Xgrid = 2 * math.pi * xi * self.M / (self.lamda * self.f_4f)
        Ygrid = 2 * math.pi * eta * self.M / (self.lamda * self.f_4f)
        Zgrid = k_sample * cos_theta_sample
        NFPgrid = k_immersion * (-1) * cos_theta_immersion  # -1

        self.Xgrid = torch.from_numpy(Xgrid).to(self.device)
        self.Ygrid = torch.from_numpy(Ygrid).to(self.device)
        self.Zgrid = torch.from_numpy(Zgrid).to(self.device)
        self.NFPgrid = torch.from_numpy(NFPgrid).to(self.device)
        self.circ = torch.from_numpy(circ).to(self.device)
        self.circ_NA = torch.from_numpy(circ_NA).to(self.device)
        self.circ_sample = torch.from_numpy(circ_sample).to(self.device)
        self.idx05 = int(N / 2)
        self.N = N
        self.pn_pupil = pn_pupil
        self.pn_circ = pn_circ
        self.circ_scaled = torch.from_numpy(circ_scaled).to(self.device)
        # for a blur kernel
        g_r = int(g_size / 2)
        g_xs = torch.linspace(-g_r, g_r, g_size, device=self.device).type(torch.float64)
        self.g_xx, self.g_yy = torch.meshgrid(g_xs, g_xs, indexing='xy')

        # crop settings
        self.r0, self.c0 = int(np.round((N-H)/2)), int(np.round((N-W)/2))
        self.H, self.W = H, W

        # -------------------------
        # DEBUG: BFP logging  #27/01/2026
        # -------------------------
        self.debug_bfp = bool(params.get("debug_bfp", True))  # hard-code True if you want
        self.debug_every_num_epoch = int(params.get("debug_every_num_epoch", 10))  # every N epochs
        self.debug_dir = str(params.get("debug_dir", os.path.join("debug", "bfp")))
        self.debug_max_emitters = int(params.get("debug_max_emitters", 5 ))  # save first K in batch  number of beads
        self.current_epoch = 0  # set by the caller (app_utils.py) before each epoch's forward calls
        self._last_debug_epoch = None  # guards against dumping more than once per qualifying epoch
        self.debug_save_time_s = 0.0  # cumulative wall-clock time spent inside _maybe_save_debug
        phase_mask_init = params.get('phase_mask_init')
        if phase_mask_init is not None:
            self.phase_mask = torch.tensor(phase_mask_init, device=self.device, dtype=torch.float32, requires_grad=True)
        else:
            self.phase_mask = torch.zeros((N, N), device=self.device, requires_grad=True)

        self.g_sigma = torch.tensor(g_sigma, device=self.device, requires_grad=True)


    def d_um(self):
        # bounded to [d_min_um, d_max_um]
        return self.d_min_um + (self.d_max_um - self.d_min_um) * torch.sigmoid(self.d_raw)

    def nfp_offset_um(self):
        return self.nfp_offset_min_um + (self.nfp_offset_max_um - self.nfp_offset_min_um) * torch.sigmoid(self.nfp_offset_raw)

    def nfps(self, zi=None):
        """Per-slice NFP sweep, fixed-length window centered on nfp_offset_um().
        zi: slice indices in [0, Z-1]; default arange(Z)."""
        center = self.nfp_offset_um()
        start = center - self.nfp_range_um / 2
        step = self.nfp_range_um / (self.Z - 1)
        if zi is None:
            zi = torch.arange(self.Z, device=self.device, dtype=start.dtype)
        else:
            zi = zi.to(dtype=start.dtype)
        return start + step * zi

    def _maybe_save_debug(self, ef_bfp_eff, psfs, xyzps, NFPs, targets=None):
        if not self.debug_bfp:
            return
        if (self.current_epoch % self.debug_every_num_epoch) != 0:
            return
        if self._last_debug_epoch == self.current_epoch:
            return  # already dumped this epoch — forward() runs many times per epoch
        self._last_debug_epoch = self.current_epoch

        os.makedirs(self.debug_dir, exist_ok=True)

        d_now = float(self.d_um().detach().cpu().item())
        g_now = float(self.g_sigma.detach().cpu().item()) if hasattr(self, "g_sigma") else float("nan")

        B = ef_bfp_eff.shape[0]
        K = self.debug_max_emitters #min(B, self.debug_max_emitters)
        if B == K:
            num_of_stacks_per_bead = B // K
        else:
            num_of_stacks_per_bead = B
            K=1

        for i in range(K):
            #subdir = os.path.join(self.debug_dir, f"emitter_{i:03d}")
            label = f"emitter_{i:03d}"
            if hasattr(self, "debug_names") and i < len(self.debug_names):
                label = str(self.debug_names[i])
            subdir = os.path.join(self.debug_dir, label)

            os.makedirs(subdir, exist_ok=True)
            #inx = int( (i+0.5*1) * num_of_stacks_per_bead)
            if B == K:
                inx = int( (0.5*1) * num_of_stacks_per_bead) + i
            else:
                inx = int( (0.5*1) * num_of_stacks_per_bead) + 1

            #phase_eff = (torch.angle(ef_bfp_eff[inx]) * self.circ).detach().cpu().numpy()
            phase_eff = (torch.angle(ef_bfp_eff[inx])).detach().cpu().numpy()

            # PSF normalize for display (not for training!)
            psf = psfs[inx].detach().cpu().numpy()
            psf_disp = psf / (psf.max() + 1e-12)

            # experimental (ground-truth) PSF this prediction is trying to match, if available
            target_disp = None
            if targets is not None and inx < targets.shape[0]:
                target = targets[inx].detach().cpu().numpy()
                target_disp = target / (target.max() + 1e-12)

            # xyz + nfp for title
            x_um = float(xyzps[inx, 0].detach().cpu().item())
            y_um = float(xyzps[inx, 1].detach().cpu().item())
            z_um = float(xyzps[inx, 2].detach().cpu().item())
            nfp = float(NFPs[inx].detach().cpu().item()) if NFPs is not None else float("nan")

            n_panels = 3 if target_disp is not None else 2
            fig = plt.figure(figsize=(15 if n_panels == 3 else 10, 4), constrained_layout=True)

            ax1 = fig.add_subplot(1, n_panels, 1)
            im1 = ax1.imshow(phase_eff, cmap="twilight")
            ax1.set_title("effective BFP phase")
            ax1.axis("off")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = fig.add_subplot(1, n_panels, 2)
            im2 = ax2.imshow(psf_disp, cmap="gray")
            ax2.set_title("PSF (display norm)")
            ax2.axis("off")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            if target_disp is not None:
                ax3 = fig.add_subplot(1, n_panels, 3)
                im3 = ax3.imshow(target_disp, cmap="gray")
                ax3.set_title("experimental PSF (target)")
                ax3.axis("off")
                fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            fig.suptitle(
                f"epoch={self.current_epoch}  d={d_now:.1f}um  g={g_now:.3f}  "
                f"x={x_um:.3f} y={y_um:.3f} z={z_um:.3f}  NFP={nfp:.3f}"
            )
            out = os.path.join(subdir, f"epoch_{self.current_epoch:06d}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)

    def forward(self, xyzps, NFPs, targets=None):

        def shift2d_integer(img2d: torch.Tensor, shift_x_px: int, shift_y_px: int):
            """
            Integer-pixel shift with zero fill.
            Positive shift_x_px -> right
            Positive shift_y_px -> down
            """
            H, W = img2d.shape
            out = torch.zeros_like(img2d)

            src_x0 = max(0, -shift_x_px)
            src_x1 = min(W, W - shift_x_px) if shift_x_px >= 0 else W
            dst_x0 = max(0, shift_x_px)
            dst_x1 = min(W, W + shift_x_px) if shift_x_px < 0 else W

            src_y0 = max(0, -shift_y_px)
            src_y1 = min(H, H - shift_y_px) if shift_y_px >= 0 else H
            dst_y0 = max(0, shift_y_px)
            dst_y1 = min(H, H + shift_y_px) if shift_y_px < 0 else H

            out[dst_y0:dst_y1, dst_x0:dst_x1] = img2d[src_y0:src_y1, src_x0:src_x1]
            return out

        def shift_complex_field_integer(field2d: torch.Tensor, shift_x_px: int, shift_y_px: int):
            real_shifted = shift2d_integer(field2d.real, shift_x_px, shift_y_px)
            imag_shifted = shift2d_integer(field2d.imag, shift_x_px, shift_y_px)
            return torch.complex(real_shifted, imag_shifted)

        xyzp = xyzps  # [B,4], um in object space

        # -----------------------------------
        # coordinates
        # -----------------------------------
        x_pix = xyzp[:, 0:1] / self.ps_camera * self.M
        y_pix = xyzp[:, 1:2] / self.ps_camera * self.M

        z = xyzp[:, 2:3]
        photons = xyzp[:, 3:4].unsqueeze(1)
        # already relative to optical axis!
        cx = 0 # float(self.centralBeadCoordinates_pixel[1])  # col
        cy = 0 # float(self.centralBeadCoordinates_pixel[0])  # row

        x_pix_rel = x_pix - cx
        y_pix_rel = y_pix - cy

        x_rel = x_pix_rel * self.ps_camera / self.M
        y_rel = y_pix_rel * self.ps_camera / self.M

        x_coarse = torch.round(x_pix_rel) * self.ps_camera / self.M
        y_coarse = torch.round(y_pix_rel) * self.ps_camera / self.M

        x_sub = x_rel - x_coarse
        y_sub = y_rel - y_coarse

        # -----------------------------------
        # BFP phase: axial + delicate sub-pixel lateral phase only
        # -----------------------------------
        NFPs_b = NFPs.to(self.NFPgrid.dtype).view(-1, 1, 1)

        phase_axial = self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFPs_b
        phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)

        circ_final_bfp = self.circ_NA
        ef_bfp = torch.exp(1j * (phase_axial + phase_lateral_sub_pixel)).to(torch.complex64)
        ef_bfp = ef_bfp * circ_final_bfp
        ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)

        # optional debug field
        #ebfp_on_axis = torch.exp(1j * phase_axial).to(torch.complex64) * self.circ

        # -----------------------------------
        # propagate to mask plane
        # -----------------------------------
        d = self.d_um()
        ef_mask = asm_propagate(ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, d, n=1.0, bandlimit=True).to(torch.complex64)
        
        # -----------------------------------
        # convert coarse lateral position to mask-plane shift
        # NOTE:
        # Current version uses a small-angle geometric approximation.
        #
        # x_coarse, y_coarse are in object-space um
        # For a microscopic system with no 4f: the 4f value should be tube lens/M (e.g. f_obj)
        # -----------------------------------
        theta_x = x_coarse / (self.f_4f / self.M)
        theta_y = y_coarse / (self.f_4f / self.M)

        dx_mask_um = d * theta_x
        dy_mask_um = d * theta_y

        dx_mask_px = (dx_mask_um / self.ps_BFP).squeeze(1)
        dy_mask_px = (dy_mask_um / self.ps_BFP).squeeze(1)

        dx_mask_px = torch.round(dx_mask_px).to(torch.int64)
        dy_mask_px = torch.round(dy_mask_px).to(torch.int64)
        self.last_mask_shift_px = torch.stack([dx_mask_px, dy_mask_px], dim=1).detach()  # [B,2]
        # -----------------------------------
        # shift complex field at mask plane
        ef_mask_shifted = []
        for i in range(ef_mask.shape[0]):
            #ef_mask_shifted.append(shift_complex_field(ef_mask[i], dx_mask_px[i], dy_mask_px[i]))
            ef_mask_shifted.append(shift_complex_field_integer(ef_mask[i], int(dx_mask_px[i].item()), int(dy_mask_px[i].item())))
        ef_mask_shifted = torch.stack(ef_mask_shifted, dim=0)

        # -----------------------------------
        # apply phase mask at mask plane
        phase = torch.exp(1j * self.phase_mask.to(ef_mask.device).to(torch.float32))
        circ_phase = (self.circ_scaled > 0.5).unsqueeze(0)

        ef_mask_shifted = ef_mask_shifted * phase * circ_phase
        ef_mask_shifted = torch.where(circ_phase > 0.5, ef_mask_shifted, 0)
        self.last_mask_plane_phase = torch.angle(ef_mask_shifted).detach()

        # -----------------------------------
        # shift back
        ef_mask_unshifted = []
        for i in range(ef_mask_shifted.shape[0]):
            #ef_mask_unshifted.append(shift_complex_field(ef_mask_shifted[i], -dx_mask_px[i], -dy_mask_px[i]))
            ef_mask_unshifted.append(shift_complex_field_integer(ef_mask_shifted[i], int(-dx_mask_px[i].item()), int(-dy_mask_px[i].item())))
        ef_mask_unshifted = torch.stack(ef_mask_unshifted, dim=0)

        # -----------------------------------
        # propagate back to BFP
        if abs(d) > 0:
            ef_bfp_after = asm_propagate(ef_mask_unshifted,self.lamda,self.ps_BFP,self.ps_BFP,-d, n=1.0,bandlimit=True).to(torch.complex64)
        else:
            ef_bfp_after = ef_mask_unshifted

        ef_bfp_after = torch.where(circ_final_bfp > 0.5, ef_bfp_after, 0)
        self.last_ef_bfp_phase = torch.angle(ef_bfp_after).detach()

        # -----------------------------------
        # image plane FFT
        psf_field = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ef_bfp_after, dim=(1, 2)), dim=(1, 2)),dim=(1, 2))
        psf = torch.abs(psf_field) ** 2
        psfs = psf / (torch.sum(psf, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # blur
        #if len(self.g_sigma) == 1:
        #g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
        #else:
        #    g_sigma = self.g_sigma
        #sigma = g_sigma[0] + torch.rand(1).to(self.device) * (g_sigma[1] - g_sigma[0])

        sigma = self.g_sigma
        blur_kernel = 1 / (2 * math.pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs), padding='same').squeeze(1)

        # renormalize after blur
        psfs = psfs / (torch.sum(psfs, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # crop
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        # debug
        if self.debug_bfp:
            _debug_t0 = time.perf_counter()
            self._maybe_save_debug(ef_bfp_after, psfs, xyzps, NFPs, targets)
            self.debug_save_time_s += time.perf_counter() - _debug_t0

        return psfs