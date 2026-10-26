import numpy as np
from math import pi
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.fft as fft
import torch.nn.functional as F
import os
from skimage import io
from torch.utils.data import Dataset
from torch.nn import MaxPool3d, ConstantPad3d
from torch.nn.functional import conv3d, interpolate
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
# ori's edit
import torch
import torch.fft as fft

### ori's edit from 25/01/2026 - subtituting the asm model with shifted circ
import torch
import torch.nn.functional as F
import numpy as np
from math import pi


def select_device() -> torch.device:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            candidate = torch.device(f"cuda:{i}")
            try:
                torch.zeros(1, device=candidate)
                return candidate
            except Exception:
                continue
    return torch.device("cpu")


def _shift2d_subpixel(img2d, shift_x_px, shift_y_px):
    """
    img2d: (N,N) or (1,1,N,N) float tensor
    shift_x_px: + moves content to the right  (cols)
    shift_y_px: + moves content down          (rows)

    Returns shifted img2d same shape.
    Uses grid_sample (subpixel, bilinear).
    """
    if img2d.dim() == 2:
        x = img2d[None, None]  # (1,1,N,N)
    elif img2d.dim() == 4:
        x = img2d
    else:
        raise ValueError("img2d must be (N,N) or (1,1,N,N)")

    _, _, N, M = x.shape
    assert N == M, "expected square pupil grid"

    # normalized translation in [-1,1]
    tx = 2.0 * (shift_x_px / (N - 1))
    ty = 2.0 * (shift_y_px / (N - 1))

    theta = x.new_tensor([[[1, 0, tx],
                           [0, 1, ty]]])  # (1,2,3)

    grid = F.affine_grid(theta, x.size(), align_corners=True)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    if img2d.dim() == 2:
        return y[0, 0]
    return y

# end ori's edit
def phase_show(name, E):
    import numpy as np, matplotlib.pyplot as plt
    ph = torch.angle(E.squeeze(0)).detach().cpu().numpy()
    plt.imshow(ph, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.show()
    plt.savefig(f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()
def asm_propagate(u0: torch.Tensor,
                  wavelength_um: float,
                  dx_um: float,
                  dy_um: float,
                  z_um: float,
                  n: float = 1.0,
                  bandlimit: bool = True) -> torch.Tensor:
    """
    Angular Spectrum propagation for complex field u0 by distance z_um (micrometers).
    u0: (..., H, W) complex tensor
    dx_um, dy_um: pixel pitch in that plane [um]
    n: refractive index of the medium (use 1.0 for air at the mask plane)
    """
    if z_um == 0:
        return u0

    device = u0.device
    H, W = u0.shape[-2], u0.shape[-1]

    k = 2 * torch.pi * (n / wavelength_um)  # [1/um]
    fx = torch.fft.fftfreq(W, d=dx_um).to(device)
    fy = torch.fft.fftfreq(H, d=dy_um).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    FX, FY = FX.T, FY.T  # (H,W)

    kx = 2 * torch.pi * FX
    ky = 2 * torch.pi * FY
    kt2 = kx**2 + ky**2
    kz2 = k**2 - kt2

    if bandlimit:
        kz = torch.sqrt(torch.clamp(kz2, min=0.0))
    else:
        kz = torch.sqrt(kz2.to(torch.complex64))

    Hprop = torch.exp(1j * kz * z_um)
    U0 = torch.fft.fft2(u0)
    U1 = U0 * Hprop
    return torch.fft.ifft2(U1)
# end ori's edit

class NonUniformBg(nn.Module):
    def __init__(self, HW=(121, 121), xy_offset=(10, 10), angle_range=(-pi / 4, pi / 4)):
        super().__init__()
        self.H, self.W = HW
        m, n = [(ss - 1.) / 2. for ss in (self.H, self.W)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        self.Xbg = torch.from_numpy(x).type(torch.FloatTensor)
        self.Ybg = torch.from_numpy(y).type(torch.FloatTensor)
        self.offsetX, self.offsetY = xy_offset  # pixel
        self.angle_min, self.angle_max = angle_range

    def forward(self, ):
        # center
        x0 = -self.offsetX + torch.rand(1) * self.offsetX * 2
        y0 = -self.offsetY + torch.rand(1) * self.offsetY * 2

        # two stds
        sigmax = self.W / 4 + torch.rand(1) * self.W / 4  # empirical
        sigmay = self.H / 4 + torch.rand(1) * self.H / 4

        # cast a new angle
        theta = self.angle_min + torch.rand(1) * (self.angle_max - self.angle_min)

        # calculate rotated gaussian coefficients
        a = torch.cos(theta) ** 2 / (2 * sigmax ** 2) + torch.sin(theta) ** 2 / (2 * sigmay ** 2)
        b = -torch.sin(2 * theta) / (4 * sigmax ** 2) + torch.sin(2 * theta) / (4 * sigmay ** 2)
        c = torch.sin(theta) ** 2 / (2 * sigmax ** 2) + torch.cos(theta) ** 2 / (2 * sigmay ** 2)

        # calculate rotated gaussian and scale it
        h = torch.exp(
            -(a * (self.Xbg - x0) ** 2 + 2 * b * (self.Xbg - x0) * (self.Ybg - y0) + c * (self.Ybg - y0) ** 2) ** 2)
        maxh = h.max()
        minh = h.min()
        h = (h - minh) / (maxh - minh)

        return h


class ImModel(nn.Module):
    def __init__(self, params):
        """
        a scalar model for air or oil objective in microscopy
        """
        super().__init__()

        ################### set parameters: unit:um
        device = params['device']

        # oil objective
        M = params['M']  # magnification
        NA = params['NA']  # NA
        n_immersion = params['n_immersion']  # refractive index of the immersion of the objective
        lamda = params['lamda']  # wavelength
        n_sample = params['n_sample']  # refractive index of the sample
        f_4f = params['f_4f']  # focal length of 4f system
        ps_camera = params['ps_camera']  # pixel size of the camera
        ps_BFP = params['ps_BFP']  # pixel size at back focal get_psfplane
        NFP = params['NFP']  # location of the nominal focal plane
        #mask_offset_in_um = params['mask_offset_in_um']  # refractive index of the immersion of the objective

        # mask at BFP
        phase_mask = params['phase_mask']

        self.non_uniform_noise_flag = False  # can be switched on

        # image
        H, W = params['H'], params['W']  # FOV size
        g_size = 9  # size of the gaussian blur kernel
        g_sigma = params['g_sigma']  # std of the gaussian blur kernel
        bg = params['bg']  # photon counts of background noise
        baseline = params['baseline']  # cannot be really certain, so should be a range
        read_std = params['read_std']  # standard deviation of readout noise
        ###################

        N = np.floor(f_4f * lamda / (ps_camera * ps_BFP))  # simulation size
        N = int(N + 1 - (N % 2))  # make it odd0

        # pupil/aperture at back focal plane
        d_pupil = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # diameter [um]
        pn_pupil = d_pupil / ps_BFP  # pixel number of the pupil diameter should be smaller than the simulation size N
        if N < pn_pupil:
            raise Exception('Simulation size is smaller than the pupil!')
        # cartesian and polar grid in BFP
        x_phys = np.linspace(-N / 2, N / 2, N) * ps_BFP
        xi, eta = np.meshgrid(x_phys, x_phys)  # cartesian physical coordinates
        r_phys = np.sqrt(xi ** 2 + eta ** 2)
        pupil = (r_phys < d_pupil / 2).astype(np.float32)

        x_ang = np.linspace(-1, 1, N) * (N / pn_pupil) * (NA / n_immersion)  # angular coordinate
        xx_ang, yy_ang = np.meshgrid(x_ang, x_ang)
        r = np.sqrt(xx_ang ** 2 + yy_ang ** 2)  # normalized angular coordinates, s.t. r = NA/n_immersion at edge of E field support

        k_immersion = 2 * pi * n_immersion / lamda  # [1/um]
        sin_theta_immersion = r
        circ_NA = (sin_theta_immersion < (NA / n_immersion)).astype(np.float32)  # the same as pupil, NA / n_immersion < 1
        cos_theta_immersion = np.sqrt(1 - (sin_theta_immersion * circ_NA) ** 2) * circ_NA

        k_sample = 2 * pi * n_sample / lamda
        sin_theta_sample = n_immersion / n_sample * sin_theta_immersion
        # note: when circ_sample is smaller than circ_NA, super angle fluorescence apears
        circ_sample = (sin_theta_sample < 1).astype(np.float32)  # if all the frequency of the sample can be captured
        cos_theta_sample = np.sqrt(1 - (sin_theta_sample * circ_sample) ** 2) * circ_sample * circ_NA

        # circular aperture to impose on BFP, SAF is excluded
        circ = circ_NA * circ_sample

        pn_circ = np.floor(np.sqrt(np.sum(circ)/pi)*2)
        pn_circ = int(pn_circ + 1 - (pn_circ % 2))
        Xgrid = 2 * pi * xi * M / (lamda * f_4f)
        Ygrid = 2 * pi * eta * M / (lamda * f_4f)
        Zgrid = k_sample * cos_theta_sample
        NFPgrid = k_immersion * (-1) * cos_theta_immersion  # -1

        self.device = device
        self.Xgrid = torch.from_numpy(Xgrid).to(device)
        self.Ygrid = torch.from_numpy(Ygrid).to(device)
        self.Zgrid = torch.from_numpy(Zgrid).to(device)
        self.NFPgrid = torch.from_numpy(NFPgrid).to(device)
        self.circ = torch.from_numpy(circ).to(device)
        self.circ_NA = torch.from_numpy(circ_NA).to(device)
        self.circ_sample = torch.from_numpy(circ_sample).to(device)
        self.idx05 = int(N / 2)
        self.N = N
        self.NFP = NFP
        self.phase_NFP = self.NFPgrid * NFP
        if phase_mask is not None:
            self.phase_mask = torch.from_numpy(phase_mask).to(device)
        else:
            self.phase_mask = torch.from_numpy(circ).to(device)
        self.pn_pupil = pn_pupil
        self.pn_circ = pn_circ

        # build a blur kernel
        g_r = int(g_size / 2)
        g_xs = np.linspace(-g_r, g_r, g_size)
        g_xx, g_yy = np.meshgrid(g_xs, g_xs)
        self.g_xx, self.g_yy = torch.from_numpy(g_xx).to(device), torch.from_numpy(g_yy).to(device)
        self.g_sigma = g_sigma
        # crop settings
        # h05, w05 = int(H / 2), int(W / 2)
        # self.h05, self.w05 = h05, w05
        self.r0, self.c0 = int(np.round((N - H) / 2)), int(np.round((N - W) / 2))
        self.H, self.W = H, W

        # noise settings, background, shot, and readout
        self.non_uniform_noise = NonUniformBg(HW=(H, W), xy_offset=(10, 10), angle_range=(-pi / 4, pi / 4))

        self.bg = bg
        self.baseline = baseline
        self.read_std = read_std
        # image bitdepth
        self.bitdepth = 16

        #added on 10/02/2026
        self.ps_camera = ps_camera
        self.M = M
        self.d_um = params['mask_offset_in_um']
        self.circ_scale = float(params.get("circ_scale", 1.0))  # 1.0 = default behavior
        r_lim_scaled = (NA / n_immersion) * self.circ_scale
        circ_NA_scaled = (sin_theta_immersion < r_lim_scaled).astype(np.float32)  # 28/01/2026
        circ_scaled = circ_NA_scaled
        self.circ_scaled = torch.from_numpy(circ_scaled).to(device)
        self.centralBeadCoordinates_pixel = params['centralBeadCoordinates_pixel']
        self.n_immersion = n_immersion
        self.NA = NA
        self.lamda = lamda
        self.ps_BFP = ps_BFP
        #NFPs = torch.tensor(np.asarray(nfp_list, np.float32), device=device)
    def get_psfs(self, xyzps):  # each batch can only have the same number of particles
        xyzp = xyzps  # um in object space
        # pixel coordinates
        x_pix = xyzp[:, 0:1] / self.ps_camera * self.M
        y_pix = xyzp[:, 1:2] / self.ps_camera * self.M

        # optical axis in pixels (col, row)
        cx = self.centralBeadCoordinates_pixel[1]
        cy = self.centralBeadCoordinates_pixel[0]

        # central correction: the input center is now center of frame
        cx = self.W / 2 - cx
        cy = self.H / 2 - cy

        # pixel offset from optical axis
        x_pix_rel = x_pix - cx  #
        y_pix_rel = y_pix - cy

        # convert to sample-plane micrometers
        x = x_pix_rel * self.ps_camera / self.M  #
        y = y_pix_rel * self.ps_camera / self.M

        z = xyzp[:, 2:3]

        photons = xyzp[:, 3:4].unsqueeze(1)

        # --- finite differences (center pixel) ---
        i0 = self.Xgrid.shape[0] // 2
        j0 = self.Xgrid.shape[1] // 2
        dX_col = (self.Xgrid[i0, j0 + 1] - self.Xgrid[i0, j0]).abs()  # phase coef step per +1 col
        dY_row = (self.Ygrid[i0 + 1, j0] - self.Ygrid[i0, j0]).abs()  # phase coef step per +1 row

        # true per-pixel phase increments
        dphi_col = dX_col * i0  # ~ |ΔX|*|x|
        dphi_row = dY_row * j0  # ~ |ΔY|*|y|

        # cycles/pixel
        ax = (dphi_col / (2 * torch.pi)).item()
        ay = (dphi_row / (2 * torch.pi)).item()

        ax = np.floor(np.abs(ax)) * 2 + 1
        ay = np.floor(np.abs(ay)) * 2 + 1
        s = int(np.sqrt(ax ** 2 + ay ** 2))  # ori's edit from 16/12/2025! to fix error in edges

        # --- scale lateral tilt & distance only ---
        x_s = x / s
        y_s = y / s
        z_s = xyzp[:, 2:3] / s  # note -should be absolute value??
        d = self.d_um  # torch scalar, has grad
        d_eff = d * s
        # print('Debug!! d=' + str(d))

        # calculating round phase shift to keep sub pixel shift
        '''
        phase_lateral_actual = self.Xgrid * x.unsqueeze(1) + self.Ygrid * y.unsqueeze(1)
        x_round = torch.floor(x * self.ps_camera) / self.ps_camera
        y_round = torch.floor(y * self.ps_camera) / self.ps_camera
        phase_lateral_round = self.Xgrid * x_round.unsqueeze(1) + self.Ygrid * y_round.unsqueeze(1)
        phase_lateral_sub_pixel = phase_lateral_actual - phase_lateral_round
        '''



        #

        # phases with scaled lateral tilt, original axial
        NFP = self.NFP  # NFPs.to(self.NFPgrid.dtype).view(-1, 1, 1)
        NFP_s = NFP / s
        phase_axial = (self.Zgrid * z_s.unsqueeze(1) + self.NFPgrid * NFP_s)
        #phase_nfp = (self.NFPgrid * NFPs_b)

        actual_phase_axial = (self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFP)
        phase_lateral = self.Xgrid * x_s.unsqueeze(1) + self.Ygrid * y_s.unsqueeze(1)



        #x_s_round = torch.floor(x_s * self.ps_camera) / self.ps_camera
        #y_s_round = torch.floor(y_s * self.ps_camera) / self.ps_camera
        #phase_lateral_round = self.Xgrid * x_s_round.unsqueeze(1) + self.Ygrid * y_s_round.unsqueeze(1)
        #phase_lateral_sub_pixel = phase_lateral - phase_lateral_round
        # sub pixel shift
        x_sub = x - torch.round(x *self.M / self.ps_camera) * self.ps_camera / self.M
        y_sub = y - torch.round(y *self.M / self.ps_camera) * self.ps_camera / self.M
        phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)
        phase_lateral_sub_pixel = phase_lateral_sub_pixel

        circ_final_bfp = self.circ_sample  # *self.circ_sample or self.circ_NA

        # ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)
        #ebfp_on_axis = torch.exp(1j * (actual_phase_axial)).to(torch.complex64)

        inx1 = torch.where(self.circ[np.shape(self.circ)[0] // 2, :] == 1)
        inx1 = inx1[0][0]
        inx1 = (np.shape(self.circ)[0] / 2 - inx1).to(self.device)

        x_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
        x_norm = x_phys / inx1
        y_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
        y_norm = y_phys / inx1

        # xx, yy = meshgrid(x_norm, y_norm); need to do meshgrid
        # rho2 = xx ** 2 + yy ** 2
        rho2 = x_norm ** 2 + y_norm ** 2
        amp = 1 / torch.abs(((1 - rho2 * (self.NA / self.n_immersion) ** 2 + 1e-16)) ** (1 / 4))
        ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64) * circ_final_bfp
        ef_bfp[torch.isnan(ef_bfp)] = 0

        ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)  # removed on 12/17/2025 - probably it is better this way!

        # forward propagate
        if d_eff != 0.0:
            ef_off = asm_propagate(ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0, bandlimit=True)
            # ef_off_on_axis =  asm_propagate(ebfp_on_axis, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0, bandlimit=True)

        else:
            ef_off = ef_bfp

        # mask & circ
        phase = torch.exp(1j * self.phase_mask.to(ef_off.device).to(torch.float32))
        circ_phase = (self.circ_scaled > 0.5).unsqueeze(0)
        ef_off = ef_off * phase * circ_phase
        ef_off = torch.where(circ_phase, ef_off, 0)


        # back propagate
        if d_eff != 0.0:
            ef_bfp = asm_propagate(ef_off, self.lamda, self.ps_BFP, self.ps_BFP, -d_eff, n=1.0, bandlimit=True).to(
                torch.complex64)
        else:
            ef_bfp = ef_off

        # remove only the (scaled) lateral phase before FFT (same as your original flow)
        ef_bfp = ef_bfp * torch.exp(1j * (-phase_lateral + phase_lateral_sub_pixel)).to(torch.complex64)  # * self.circ
        ef_bfp = ef_bfp * torch.exp(1j * (actual_phase_axial - phase_axial)).to(torch.complex64) * circ_final_bfp
        ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)  # removed on 12/17/2025 - probably it is better this way!


        psf_field = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))

        #psf_on_axis = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ebfp_on_axis * self.circ, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))

        psf = torch.abs(psf_field) ** 2
        psfs = psf / torch.sum(torch.abs(psf), dim=(1, 2), keepdims=True) * photons

        # blur
        if len(self.g_sigma)==1:
            g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
        else:
            g_sigma = self.g_sigma
        sigma = g_sigma[0] + torch.rand(1).to(self.device) * (g_sigma[1] - g_sigma[0])

        blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs), padding='same').squeeze(1)

        # photon normalization
        # psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)  # photon normalization
        # psfs = psfs[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]


        return psfs

        ''' original
        phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
        phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.NFPgrid * self.NFP
        ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
        psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))  # FT
        psfs = torch.abs(psf_field) ** 2
        # blur
        sigma = self.g_sigma[0]+torch.rand(1).to(self.device)*(self.g_sigma[1]-self.g_sigma[0])
        blur_kernel = 1/(2*pi*sigma ** 2)*(torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        psfs = psfs.squeeze(1)

        # photon normalization
        psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)  # photon normalization
        # psfs = psfs[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
       
        return psfs
        '''

    def forward(self, xyzps):
        """
        image of point sources
        :param xyzps: spatial locations and photon counts, tensor, rank 2 [n 4]
        :return: tensor, image
        """
        psfs = self.get_psfs(xyzps)

        im = torch.sum(psfs, dim=0)

        # noise: background, shot, readout
        im = torch.poisson(im + self.bg)  # rounded

        read_baseline = self.baseline[0]+torch.rand(1, device=self.device)*(self.baseline[1]-self.baseline[0])
        read_std = self.read_std[0]+torch.rand(1, device=self.device)*(self.read_std[1]-self.read_std[0])

        if self.non_uniform_noise_flag:
            # choose a range in the preset read_std range to reshape the non-uniform distribution
            std_pv = torch.rand(1, device=self.device)*(self.read_std[1]-read_std)  # read_std--valley
            std = self.non_uniform_noise().to(self.device)*std_pv + read_std
            im = im + torch.round(read_baseline + torch.randn(im.shape, device=self.device) * std)
        else:
            im = im + torch.round(read_baseline + torch.randn(im.shape, device=self.device) * read_std)

        im[im < 0] = 0
        max_adu = 2**self.bitdepth - 1
        im[im > max_adu] = max_adu
        im = im.type(torch.int32)

        return im

    def show_circs(self):
        """
        plot several windows/circles in BFP
        :return: plot the windows
        """
        plt.figure(figsize=(4, 3))
        plt.plot(self.circ_NA.cpu().numpy()[self.idx05, :] + 0.5)
        plt.plot(self.circ_sample.cpu().numpy()[self.idx05, :] + 0.25)
        plt.plot(self.circ.cpu().numpy()[self.idx05, :])
        plt.plot(self.phase_mask.cpu().numpy()[self.idx05, :])
        plt.legend(['immersion', 'sample', 'aper', 'mask'])
        plt.title('circles in BFP')
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        plt.show()

    def model_demo(self, zs):
        xyzps = np.c_[np.zeros(zs.shape[0]), np.zeros(zs.shape[0]), zs, np.ones(zs.shape[0])*1e4]
        zstack = self.get_psfs(torch.from_numpy(xyzps).to(self.device)).cpu()
        plt.figure(figsize=(6, 2))
        plt.imshow(torch.cat([zstack[i] for i in range(5)], dim=1))
        plt.title(f'z positions [um]: {zs}')
        plt.axis('off')
        plt.savefig('PSFs.jpg', bbox_inches='tight', dpi=300)
        plt.clf()
        print('Imaging model: PSFs.jpg')


class ImModelBase(nn.Module):
    def __init__(self, params):
        """
        a scalar model for air or oil objective in microscopy
        """
        super().__init__()

        ################### set parameters: unit:um
        device = params['device']
        M = params['M']  # magnification
        NA = params['NA']  # NA
        n_immersion = params['n_immersion']  # refractive index of the immersion of the objective
        lamda = params['lamda']  # wavelength
        n_sample = params['n_sample']  # refractive index of the sample
        f_4f = params['f_4f']  # focal length of 4f system
        ps_camera = params['ps_camera']  # pixel size of the camera
        ps_BFP = params['ps_BFP']  # pixel size at back focal plane
        NFP = params['NFP']  # location of the nominal focal plane
        H, W = params['H'], params['W']  # FOV size
        ###################
        self.M = M
        self.ps_BFP = ps_BFP
        self.f_4f = f_4f
        # BFP calculation
        N = np.floor(f_4f * lamda / (ps_camera * ps_BFP))  # simulation size
        N = int(N + 1 - (N % 2))  # make it odd0

        # pupil/aperture at back focal plane
        d_pupil = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # diameter [um]
        pn_pupil = d_pupil / ps_BFP  # pixel number of the pupil diameter should be smaller than the simulation size N
        if N < pn_pupil:
            raise Exception('Simulation size is smaller than the pupil!')
        # cartesian and polar grid in BFP
        x_phys = np.linspace(-N / 2, N / 2, N) * ps_BFP
        xi, eta = np.meshgrid(x_phys, x_phys)  # cartesian physical coordinates
        r_phys = np.sqrt(xi ** 2 + eta ** 2)
        pupil = (r_phys < d_pupil / 2).astype(np.float32)

        x_ang = np.linspace(-1, 1, N) * (N / pn_pupil) * (NA / n_immersion)  # angular coordinate
        xx_ang, yy_ang = np.meshgrid(x_ang, x_ang)
        r = np.sqrt(
            xx_ang ** 2 + yy_ang ** 2)  # normalized angular coordinates, s.t. r = NA/n_immersion at edge of E field support

        k_immersion = 2 * pi * n_immersion / lamda  # [1/um]
        sin_theta_immersion = r
        circ_NA = (sin_theta_immersion < (NA / n_immersion)).astype(
            np.float32)  # the same as pupil, NA / n_immersion < 1
        cos_theta_immersion = np.sqrt(1 - (sin_theta_immersion * circ_NA) ** 2) * circ_NA

        k_sample = 2 * pi * n_sample / lamda
        sin_theta_sample = n_immersion / n_sample * sin_theta_immersion
        # note: when circ_sample is smaller than circ_NA, super angle fluorescence apears
        circ_sample = (sin_theta_sample < 1).astype(np.float32)  # if all the frequency of the sample can be captured
        cos_theta_sample = np.sqrt(1 - (sin_theta_sample * circ_sample) ** 2) * circ_sample * circ_NA

        # circular aperture to impose on BFP, SAF is excluded
        circ = circ_NA * circ_sample

        pn_circ = np.floor(np.sqrt(np.sum(circ) / pi) * 2)
        pn_circ = int(pn_circ + 1 - (pn_circ % 2))
        Xgrid = 2 * pi * xi * M / (lamda * f_4f)
        Ygrid = 2 * pi * eta * M / (lamda * f_4f)
        Zgrid = k_sample * cos_theta_sample
        NFPgrid = k_immersion * (-1) * cos_theta_immersion  # -1

        self.x_ang = x_ang
        self.device = device
        #self.centralBeadCoordinates_pixel = [849, 854]  # test2 - Should be like in ImModelTraining __init__
        #self.centralBeadCoordinates_pixel = [965, 753]  #Fov6 Should be like in ImModelTraining __init__
        #self.centralBeadCoordinates_pixel = [562, 600]  # Mitochondria_flat_from_January2026
        #self.centralBeadCoordinates_pixel = [561, 647]  # Mitochondria_from_21January2026

        self.ps_camera = params['ps_camera']
        #self.mask_offset_in_um = 40000 * 1 #38000
        #self.mask_offset_in_um = params['mask_offset_in_um']  #25/01/2026
        self.mask_offset_in_um = float(params.get('mask_offset_in_um', 0.0)) #25/01/2026
        self.centralBeadCoordinates_pixel = list(params.get('centralBeadCoordinates_pixel', [600,600]))

        ## GOOD: read from param_dict
        #self.mask_offset_in_um = float(param_dict.get("mask_offset_in_um", 0.0))
        #self.centralBeadCoordinates_pixel = list(param_dict.get("centralBeadCoordinates_pixel", [0, 0]))

        #self.centralBeadCoordinates_pixel = params['centralBeadCoordinates_pixel']  #25/01/2026

        self.lamda = params['lamda']
        self.ps_BFP = params['ps_BFP']
        # end ori's edit

        self.Xgrid = torch.from_numpy(Xgrid).to(device)
        self.Ygrid = torch.from_numpy(Ygrid).to(device)
        self.Zgrid = torch.from_numpy(Zgrid).to(device)
        self.NFPgrid = torch.from_numpy(NFPgrid).to(device)
        self.circ = torch.from_numpy(circ).to(device)
        self.circ_NA = torch.from_numpy(circ_NA).to(device)
        self.circ_sample = torch.from_numpy(circ_sample).to(device)
        self.idx05 = int(N / 2)
        self.N = N
        self.NFP = NFP
        self.phase_NFP = self.NFPgrid * NFP

        if 'phase_mask' in params:
            self.phase_mask = torch.tensor(params['phase_mask'], device=device)
        else:
            self.phase_mask = torch.tensor(circ, device=device).type(torch.float32)

        if 'g_sigma' in params:
            self.g_sigma = torch.tensor(params['g_sigma'], device=device)
        else:
            self.g_sigma = torch.tensor(1.0, device=device)

        self.pn_pupil = pn_pupil
        self.pn_circ = pn_circ

        g_size = 9  # size of the gaussian blur kernel
        # build a blur kernel
        g_r = int(g_size / 2)
        g_xs = np.linspace(-g_r, g_r, g_size)
        g_xx, g_yy = np.meshgrid(g_xs, g_xs)
        self.g_xx, self.g_yy = torch.from_numpy(g_xx).to(device), torch.from_numpy(g_yy).to(device)

        # crop settings
        self.r0, self.c0 = int(np.round((N - H) / 2)), int(np.round((N - W) / 2))
        self.H, self.W = H, W

    def get_psfs(self, xyzps):  # each batch can only have the same number of particles
        import torch
        import torch.nn.functional as F
        from math import pi

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

        xx = np.linspace(-0.5, 0.5, self.circ.shape[1]) * self.circ.shape[1]
        yy = np.linspace(-0.5, 0.5, self.circ.shape[0]) * self.circ.shape[0]
        XX, YY = np.meshgrid(xx, yy)
        circ_mask = torch.ones_like(self.circ)
        circ_mask[((XX) ** 2 + (YY) ** 2 > (5300 / self.ps_BFP) ** 2)] = 0
        self.circ_scaled = circ_mask
        #xyzp = torch.from_numpy(xyzp_np).to(self.device)  # here it is um in object space (relative? probably  center removal is not required)
        xyzp = xyzps
        # -----------------------------------
        # coordinates
        # -----------------------------------
        x_pix = xyzp[:,0:1] / self.ps_camera * self.M
        y_pix = xyzp[:,1:2] / self.ps_camera * self.M

        cx = float(self.centralBeadCoordinates_pixel[1])  # col
        cy = float(self.centralBeadCoordinates_pixel[0])  # row

        # central correction: the input center is now center of frame
        cx = cx - self.W / 2
        cy = cy - self.H / 2

        x_pix_rel = x_pix - cx
        y_pix_rel = y_pix - cy

        x = x_pix_rel * self.ps_camera / self.M
        y = y_pix_rel * self.ps_camera / self.M
        z = xyzp[:,2:3]
        photons = xyzp[:,3:4].unsqueeze(1)

        # -----------------------------------
        # coarse + sub-pixel split
        # -----------------------------------
        x_coarse = torch.round(x * self.M / self.ps_camera) * self.ps_camera / self.M
        y_coarse = torch.round(y * self.M / self.ps_camera) * self.ps_camera / self.M

        x_sub = x - x_coarse
        y_sub = y - y_coarse

        # -----------------------------------
        # BFP phase: axial + delicate sub-pixel lateral phase only
        # -----------------------------------
        NFPs_b = self.NFP

        phase_axial = self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFPs_b
        phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)
        phase_lateral_coarse = self.Xgrid * (x_coarse +  cx* self.ps_camera / self.M).unsqueeze(1) + self.Ygrid * (y_coarse + cy* self.ps_camera / self.M).unsqueeze(1)

        circ_final_bfp = self.circ
        ef_bfp = torch.exp(1j * (phase_axial + phase_lateral_sub_pixel)).to(torch.complex64)
        #ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)
        ef_bfp = ef_bfp * circ_final_bfp
        ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)

        # optional debug field
        # ebfp_on_axis = torch.exp(1j * phase_axial).to(torch.complex64) * self.circ

        # -----------------------------------
        # propagate to mask plane
        # -----------------------------------
        # d = self.d_um()  # if callable(self.d_um) else self.d_um
        d = self.mask_offset_in_um  # self.d_um()  # torch scalar, has grad
        d_scalar = d  # float(d.detach().cpu().item()) if torch.is_tensor(d) else float(d)

        if abs(d_scalar) > 0:
            ef_mask = asm_propagate(ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, +d_scalar, n=1.0, bandlimit=True).to(
                torch.complex64)
        else:
            ef_mask = ef_bfp

        # -----------------------------------
        # convert coarse lateral position to mask-plane shift
        # NOTE:
        # This mapping is the part you should calibrate physically.
        # Current version uses a small-angle geometric approximation.
        #
        # x_coarse, y_coarse are in object-space um
        # f_4f is used here as a placeholder effective propagation geometry. for a microscopic system with no 4f: the 4f value should be tube lens/M (e.g. f_obj)
        # -----------------------------------
        theta_x = x_coarse / (self.f_4f / self.M)
        theta_y = y_coarse / (self.f_4f / self.M)

        dx_mask_um = d_scalar * theta_x
        dy_mask_um = d_scalar * theta_y

        dx_mask_px = (dx_mask_um / self.ps_BFP)  # .squeeze(1)
        dy_mask_px = (dy_mask_um / self.ps_BFP)  # .squeeze(1)

        dx_mask_px = torch.round(dx_mask_px).to(torch.int64)
        dy_mask_px = torch.round(dy_mask_px).to(torch.int64)
        # -----------------------------------
        # shift complex field at mask plane
        ef_mask_shifted = []
        # for i in range(ef_mask.shape[0]):
        #ef_mask_shifted.append(shift_complex_field_integer(ef_mask, int(dx_mask_px.item()), int(dy_mask_px.item())))
        for ii in range(len(dx_mask_px)):
            ef_mask_shifted.append(shift_complex_field_integer(ef_mask[ii,:,:], int(dx_mask_px[ii].item()), int(dy_mask_px[ii].item())))
        ef_mask_shifted = torch.stack(ef_mask_shifted, dim=0)

        # -----------------------------------
        # apply phase mask at mask plane
        phase = torch.exp(1j * self.phase_mask.to(ef_mask.device).to(torch.float32))
        circ_phase = (circ_mask > 0.5).unsqueeze(0)

        ef_mask_shifted = ef_mask_shifted * phase * circ_phase
        ef_mask_shifted = torch.where(circ_phase > 0.5, ef_mask_shifted, 0)

        # -----------------------------------
        # shift back
        ef_mask_unshifted = []
        # for i in range(ef_mask_shifted.shape[0]):
        # ef_mask_unshifted.append(shift_complex_field_integer(ef_mask_shifted[i], int(-dx_mask_px[i].item()), int(-dy_mask_px[i].item())))
        for ii in range(len(dx_mask_px)):
            ef_mask_unshifted.append(shift_complex_field_integer(ef_mask_shifted[ii,:,:], int(-dx_mask_px[ii].item()), int(-dy_mask_px[ii].item())))

        ef_mask_unshifted = torch.stack(ef_mask_unshifted, dim=0)

        # -----------------------------------
        # propagate back to BFP
        if abs(d_scalar) > 0:
            ef_bfp_after = asm_propagate(ef_mask_unshifted, self.lamda, self.ps_BFP, self.ps_BFP, -d_scalar, n=1.0,
                                         bandlimit=True).to(torch.complex64)
        else:
            ef_bfp_after = ef_mask_unshifted

        ef_bfp_after = ef_bfp_after * torch.exp(1j * (phase_lateral_coarse)).to(torch.complex64) # adding coarse lateral phase
        ef_bfp_after = torch.where(circ_final_bfp > 0.5, ef_bfp_after, 0)
        # -----------------------------------
        # image plane FFT
        psf_field = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ef_bfp_after, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
        psf = torch.abs(psf_field) ** 2
        psfs = psf / (torch.sum(psf, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # blur
        if len(self.g_sigma) == 1:
            g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
        else:
            g_sigma = self.g_sigma
        sigma = g_sigma[0] + torch.rand(1).to(self.device) * (g_sigma[1] - g_sigma[0])
        blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs), padding='same').squeeze(1)

        # renormalize after blur
        psfs = psfs / (torch.sum(psfs, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # crop
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        # debug

        #return psfs[0].detach().cpu().numpy().astype(np.float32)
        return psfs.squeeze(1) #psf[0].detach().cpu().numpy()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
        phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.NFPgrid * self.NFP
        ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
        psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))  # FT
        psfs = torch.abs(psf_field) ** 2
        # blur
        #if self.g_sigma.dim() == 0:
        if len(self.g_sigma) == 0:
            sigma = self.g_sigma
        else:
            sigma = self.g_sigma[0] + torch.rand(1).to(self.device) * (self.g_sigma[1] - self.g_sigma[0])

        blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        psfs = psfs.squeeze(1)
        # photon normalization
        psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)  # photon normalization
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
        return psfs
        '''

    def show_circs(self):
        """
        plot several windows/circles in BFP
        """
        plt.figure(figsize=(4, 3))
        plt.plot(self.x_ang, self.circ_NA.cpu().numpy()[self.idx05, :] + 0.2)
        plt.plot(self.x_ang, self.circ_sample.cpu().numpy()[self.idx05, :] + 0.1)
        plt.plot(self.x_ang, self.circ.cpu().numpy()[self.idx05, :])
        # plt.plot(self.phase_mask.cpu().numpy()[self.idx05, :])
        plt.legend(['NA', 'no SAF', 'practical aper'])
        plt.title('circles in BFP')
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        plt.xlabel('"general sin_theta" of incidence light to objective ')
        plt.show()

    def model_demo(self, zs):
        xyzps = np.c_[np.zeros(zs.shape[0]), np.zeros(zs.shape[0]),
        zs,
        np.ones(zs.shape[0]) * 2e4]
        zstack = self.get_psfs(torch.from_numpy(xyzps).to(self.device)).cpu()
        plt.figure(figsize=(6, 2))
        plt.imshow(torch.cat([zstack[i] for i in range(5)], dim=1))
        plt.title(f'z positions [$\mu$m]: {zs}')
        plt.axis('off')
        plt.show()


class ImModelBead(ImModelBase):
    def __init__(self, param_dict):
        super().__init__(param_dict)

    def forward(self, xyzps, nfps):
        # xyzps: tensor, rank 2 [x, 4]
        # nfps: tensor, rank 2 [x, 1]
        print('wrong model! stops')
        return 0

        phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
        phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.NFPgrid * nfps.unsqueeze(1)
        ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
        psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))  # FT
        psfs = torch.abs(psf_field) ** 2
        # blur
        sigma = self.g_sigma
        blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0), padding='same')
        psfs = psfs.squeeze(1)
        # photon normalization
        psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)  # photon normalization
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
        return psfs


class ImModelTraining(ImModelBase):
    # Ori's edit
    # inside class ImModelTraining
    '''
    def psf_patch_clean(self, xyzp_np):
        """
        xyzp_np: np.array shape (4,) -> [x_um, y_um, z_um, photons]
        Returns a single PSF patch of shape (N, N) on the small grid, no noise.
        """
        xyzp = torch.from_numpy(xyzp_np[None, :]).to(self.device)
        psf = self.get_psfs(xyzp)  # [1, N, N], already normalized to photons and cropped to (H,W) earlier
        # IMPORTANT: for patching we want the *full* small grid, not the big-FOV crop:
        # so compute on the base model without the final crop:
        # -> quick workaround: temporarily bypass crop by recomputing at full N:
        phase_lateral = self.Xgrid * (xyzp[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzp[:, 1:2].unsqueeze(1))
        phase_axial = self.Zgrid * (xyzp[:, 2:3].unsqueeze(1)) + self.NFPgrid * self.NFP
        ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
        psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
        psf = torch.abs(psf_field) ** 2
        # photon norm
        psf = psf / torch.sum(psf, dim=(1, 2), keepdims=True) * xyzp[:, 3:4].unsqueeze(1)
        # emitter-wise blur (use a single sigma draw)
        std_min, std_max = self.g_sigma
        sigma = (std_min + (std_max - std_min) * torch.rand(1).to(self.device))
        blur = 1 / (2 * pi * sigma ** 2) * torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2)
        psf = F.conv2d(psf.unsqueeze(1), blur.unsqueeze(0).unsqueeze(0), padding='same').squeeze(1)  # [1,N,N]
        return psf[0].detach().cpu().numpy()  # (N,N)

    #'''
    #ori's edit:
    # in ds3d_utils.py (inside ImModelTraining)

    # --- at top of class, optional limits for stability ---
    # self.tilt_aa_smax = getattr(self, "tilt_aa_smax", 16)

    def _center_phase_steps(self):
        # finite differences at the center pixel (row i0, col j0)
        i0 = self.Xgrid.shape[0] // 2
        j0 = self.Xgrid.shape[1] // 2
        dXdx = (self.Xgrid[i0, j0 + 1] - self.Xgrid[i0, j0]).abs()
        dYdx = (self.Ygrid[i0, j0 + 1] - self.Ygrid[i0, j0]).abs()
        dXdy = (self.Xgrid[i0 + 1, j0] - self.Xgrid[i0, j0]).abs()
        dYdy = (self.Ygrid[i0 + 1, j0] - self.Ygrid[i0, j0]).abs()
        return dXdx, dYdx, dXdy, dYdy  # scalars (on device)

    def psf_patch_clean(self, xyzp_np):
        import torch
        import torch.nn.functional as F
        from math import pi

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

        xx = np.linspace(-0.5, 0.5, self.circ.shape[1]) * self.circ.shape[1]
        yy = np.linspace(-0.5, 0.5, self.circ.shape[0]) * self.circ.shape[0]
        XX, YY = np.meshgrid(xx, yy)
        circ_mask = torch.ones_like(self.circ)
        circ_mask[((XX) ** 2 + (YY) ** 2 > (5300 / self.ps_BFP) ** 2)] = 0
        self.circ_scaled = circ_mask
        xyzp = torch.from_numpy(xyzp_np).to(self.device)  # here it is um in object space (relative? probably  center removal is not required)

        # -----------------------------------
        # coordinates
        # -----------------------------------
        x_pix = xyzp[0:1] / self.ps_camera * self.M
        y_pix = xyzp[1:2] / self.ps_camera * self.M

        cx = float(self.centralBeadCoordinates_pixel[1])  # col
        cy = float(self.centralBeadCoordinates_pixel[0])  # row

        # central correction: the input center is now center of frame
        cx = cx-self.W / 2
        cy = cy-self.H / 2

        x_pix_rel = x_pix - cx
        y_pix_rel = y_pix - cy

        x = x_pix_rel * self.ps_camera / self.M
        y = y_pix_rel * self.ps_camera / self.M
        z = xyzp[2:3]
        photons = xyzp[3:4].unsqueeze(1)

        # -----------------------------------
        # coarse + sub-pixel split
        # -----------------------------------
        x_coarse = torch.round(x * self.M / self.ps_camera) * self.ps_camera / self.M
        y_coarse = torch.round(y * self.M / self.ps_camera) * self.ps_camera / self.M

        x_sub = x - x_coarse
        y_sub = y - y_coarse

        # -----------------------------------
        # BFP phase: axial + delicate sub-pixel lateral phase only
        # -----------------------------------
        NFPs_b = self.NFP

        phase_axial = self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFPs_b
        phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)

        circ_final_bfp = self.circ
        ef_bfp = torch.exp(1j * (phase_axial + phase_lateral_sub_pixel)).to(torch.complex64)
        ef_bfp = ef_bfp * circ_final_bfp
        ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)

        # optional debug field
        # ebfp_on_axis = torch.exp(1j * phase_axial).to(torch.complex64) * self.circ

        # -----------------------------------
        # propagate to mask plane
        # -----------------------------------
        #d = self.d_um()  # if callable(self.d_um) else self.d_um
        d = self.mask_offset_in_um  # self.d_um()  # torch scalar, has grad
        d_scalar = d  # float(d.detach().cpu().item()) if torch.is_tensor(d) else float(d)

        if abs(d_scalar) > 0:
            ef_mask = asm_propagate(ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, +d_scalar, n=1.0, bandlimit=True).to(torch.complex64)
        else:
            ef_mask = ef_bfp

        # -----------------------------------
        # convert coarse lateral position to mask-plane shift
        # NOTE:
        # This mapping is the part you should calibrate physically.
        # Current version uses a small-angle geometric approximation.
        #
        # x_coarse, y_coarse are in object-space um
        # f_4f is used here as a placeholder effective propagation geometry. for a microscopic system with no 4f: the 4f value should be tube lens/M (e.g. f_obj)
        # -----------------------------------
        theta_x = x_coarse / (self.f_4f / self.M)
        theta_y = y_coarse / (self.f_4f / self.M)

        dx_mask_um = d_scalar * theta_x
        dy_mask_um = d_scalar * theta_y

        dx_mask_px = (dx_mask_um / self.ps_BFP)#.squeeze(1)
        dy_mask_px = (dy_mask_um / self.ps_BFP)#.squeeze(1)

        dx_mask_px = torch.round(dx_mask_px).to(torch.int64)
        dy_mask_px = torch.round(dy_mask_px).to(torch.int64)
        # -----------------------------------
        # shift complex field at mask plane
        ef_mask_shifted = []
        #for i in range(ef_mask.shape[0]):
            #ef_mask_shifted.append(shift_complex_field_integer(ef_mask[i], int(dx_mask_px[i].item()), int(dy_mask_px[i].item())))
        ef_mask_shifted.append(shift_complex_field_integer(ef_mask, int(dx_mask_px.item()), int(dy_mask_px.item())))
        ef_mask_shifted = torch.stack(ef_mask_shifted, dim=0)

        # -----------------------------------
        # apply phase mask at mask plane
        phase = torch.exp(1j * self.phase_mask.to(ef_mask.device).to(torch.float32))
        circ_phase = (circ_mask > 0.5).unsqueeze(0)

        ef_mask_shifted = ef_mask_shifted * phase * circ_phase
        ef_mask_shifted = torch.where(circ_phase > 0.5, ef_mask_shifted, 0)

        # -----------------------------------
        # shift back
        ef_mask_unshifted = []
        #for i in range(ef_mask_shifted.shape[0]):
            #ef_mask_unshifted.append(shift_complex_field_integer(ef_mask_shifted[i], int(-dx_mask_px[i].item()), int(-dy_mask_px[i].item())))
        ef_mask_unshifted.append(shift_complex_field_integer(ef_mask_shifted[0], int(-dx_mask_px.item()), int(-dy_mask_px.item())))

        ef_mask_unshifted = torch.stack(ef_mask_unshifted, dim=0)

        # -----------------------------------
        # propagate back to BFP
        if abs(d_scalar) > 0:
            ef_bfp_after = asm_propagate(ef_mask_unshifted, self.lamda, self.ps_BFP, self.ps_BFP, -d_scalar, n=1.0, bandlimit=True).to(torch.complex64)
        else:
            ef_bfp_after = ef_mask_unshifted

        ef_bfp_after = torch.where(circ_final_bfp > 0.5, ef_bfp_after, 0)

        # -----------------------------------
        # image plane FFT
        psf_field = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ef_bfp_after, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
        psf = torch.abs(psf_field) ** 2
        psfs = psf / (torch.sum(psf, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # blur
        if len(self.g_sigma) == 1:
            g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
        else:
            g_sigma = self.g_sigma
        sigma = g_sigma[0] + torch.rand(1).to(self.device) * (g_sigma[1] - g_sigma[0])
        blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs), padding='same').squeeze(1)

        # renormalize after blur
        psfs = psfs / (torch.sum(psfs, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # crop
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        # debug

        return psfs[0].detach().cpu().numpy().astype(np.float32)
        '''
        asm_model = True

        if asm_model:
            import torch
            import torch.nn.functional as F
            from math import pi


            xx = np.linspace(-0.5,0.5,self.circ.shape[1]) * self.circ.shape[1]
            yy = np.linspace(-0.5,0.5,self.circ.shape[0]) * self.circ.shape[0]
            XX,YY = np.meshgrid(xx,yy)
            circ_mask = torch.ones_like(self.circ)
            circ_mask[((XX)**2+(YY)**2 > (5300/self.ps_BFP)**2)] = 0
            #circ_mask[((XX)**2+(YY)**2 > (6000/self.ps_BFP)**2)] = 0
            #circ_new[((XX+40)**2+(YY-40)**2 < (130/2)**2)] = 1
            #circ_new[((XX-40)**2+(YY+40)**2 < (130/2)**2)] = 1


            xyzp = torch.from_numpy(xyzp_np).to(self.device)  # here it is um in object space (relative? probably  center removal is not required)
            # pixel coordinates
            x_pix = xyzp[0:1]  * self.M / self.ps_camera #* self.ps_camera
            y_pix = xyzp[1:2] * self.M / self.ps_camera #* self.ps_camera

            # optical axis in pixels (col, row)
            cx = self.centralBeadCoordinates_pixel[1]
            cy = self.centralBeadCoordinates_pixel[0]

            # central correction: the input center is now center of frame
            cx = self.W/2 - cx
            cy = self.H/2 - cy

            # pixel offset from optical axis
            x_pix_rel = x_pix  - cx #
            y_pix_rel = y_pix  - cy

            # convert to sample-plane micrometers
            x = x_pix_rel * self.ps_camera / self.M  #
            y = y_pix_rel * self.ps_camera / self.M

            z = xyzp[2:3]

            photons = xyzp[3:4]

            # --- finite differences (center pixel) ---
            i0 = self.Xgrid.shape[0] // 2
            j0 = self.Xgrid.shape[1] // 2
            dX_col = (self.Xgrid[i0, j0 + 1] - self.Xgrid[i0, j0]).abs()  # phase coef step per +1 col
            dY_row = (self.Ygrid[i0 + 1, j0] - self.Ygrid[i0, j0]).abs()  # phase coef step per +1 row

            # true per-pixel phase increments
            dphi_col = dX_col * i0  # ~ |ΔX|*|x|
            dphi_row = dY_row * j0  # ~ |ΔY|*|y|

            # cycles/pixel
            ax = (dphi_col / (2 * torch.pi)).item()
            ay = (dphi_row / (2 * torch.pi)).item()

            ax = np.floor(np.abs(ax)) * 2 + 1
            ay = np.floor(np.abs(ay)) * 2 + 1
            s = int(np.sqrt(ax ** 2 + ay ** 2))  # ori's edit from 16/12/2025! to fix error in edges

            # --- scale lateral tilt & distance only ---
            x_s = x / s
            y_s = y / s
            z_s = xyzp[2:3] / s  # note -should be absolute value??
            d = self.mask_offset_in_um  #self.d_um()  # torch scalar, has grad
            d_eff = d * s
            #print('Debug!! d=' + str(d))

            # calculating round phase shift to keep sub pixel shift
            ''' '''
            phase_lateral_actual = self.Xgrid * x.unsqueeze(1) + self.Ygrid * y.unsqueeze(1)
            x_round = torch.floor(x * self.ps_camera) / self.ps_camera
            y_round = torch.floor(y * self.ps_camera) / self.ps_camera
            phase_lateral_round = self.Xgrid * x_round.unsqueeze(1) + self.Ygrid * y_round.unsqueeze(1)
            phase_lateral_sub_pixel = phase_lateral_actual - phase_lateral_round
            '''

            #
        '''
            # phases with scaled lateral tilt, original axial
            NFPs_b = torch.tensor(self.NFP).to(self.device).view(-1, 1, 1)
            NFPs_s = NFPs_b / s

            phase_axial = (self.Zgrid * z_s + self.NFPgrid * NFPs_s)
            actual_phase_axial = (self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFPs_b)
            phase_lateral = self.Xgrid * x_s.unsqueeze(1) + self.Ygrid * y_s.unsqueeze(1)

            # sub pixel phase
            #x_s_round = torch.floor(x_s * self.ps_camera) / self.ps_camera
            #y_s_round = torch.floor(y_s * self.ps_camera) / self.ps_camera
            #phase_lateral_round = self.Xgrid * x_s_round.unsqueeze(1) + self.Ygrid * y_s_round.unsqueeze(1)
            #phase_lateral_sub_pixel = phase_lateral - phase_lateral_round

            x_sub = x - torch.round(x * self.M / self.ps_camera) * self.ps_camera / self.M
            y_sub = y - torch.round(y * self.M / self.ps_camera) * self.ps_camera / self.M
            phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)
            phase_lateral_sub_pixel = phase_lateral_sub_pixel

            # adding amplitude of bfp 22/12/2025

            inx1 = torch.where(self.circ[np.shape(self.circ)[0] // 2, :] == 1)
            inx1 = inx1[0][0]
            inx1 = (np.shape(self.circ)[0] / 2 - inx1).to(self.device)

            #x_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
            #x_norm = x_phys / inx1
            #y_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
            #y_norm = y_phys / inx1

            # xx, yy = meshgrid(x_norm, y_norm); need to do meshgrid
            # rho2 = xx ** 2 + yy ** 2
            #rho2 = x_norm ** 2 + y_norm ** 2
            #amp = 1 / torch.abs(((1 - rho2 * (self.NA / self.n_immersion) ** 2 + 1e-16)) ** (1 / 4))
            ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)
            ef_bfp[torch.isnan(ef_bfp)] = 0
            ebfp_on_axis = torch.exp(1j * (actual_phase_axial)).to(torch.complex64)

            ef_bfp_nocirc = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)

            # forward propagate
            if d_eff != 0.0:
                ef_off = asm_propagate(ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0, bandlimit=True)
                ef_off_nocirc = asm_propagate(ef_bfp_nocirc, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0,bandlimit=True)
            else:
                ef_off = ef_bfp
                ef_off_nocirc = ef_bfp_nocirc

            # mask & circ
            phase = torch.exp(1j * self.phase_mask.to(ef_off.device).to(torch.float32))

            #circ_phase = (self.circ_scaled > 0.5).unsqueeze(0)
            circ_phase = (circ_mask > 0.5).unsqueeze(0)

            ef_off = ef_off * phase * circ_phase  #self.circ_scaled
            ef_off = torch.where(circ_phase, ef_off, 0)

            # back propagate
            if d_eff != 0.0:
                ef_bfp = asm_propagate(ef_off, self.lamda, self.ps_BFP, self.ps_BFP, -d_eff, n=1.0, bandlimit=True).to(torch.complex64)
                ef_bfp_nocirc = asm_propagate(ef_off_nocirc, self.lamda, self.ps_BFP, self.ps_BFP, -d_eff, n=1.0,bandlimit=True).to(torch.complex64)
            else:
                ef_bfp = ef_off
                ef_bfp_nocirc = ef_off_nocirc

            circ_final_bfp = self.circ  #self.circ or *self.circ_sample
            # remove only the (scaled) lateral phase before FFT (same as your original flow)
            ef_bfp = ef_bfp * torch.exp(1j * (-phase_lateral + phase_lateral_sub_pixel)).to(torch.complex64)# * self.circ
            #ef_bfp = ef_bfp * torch.exp(1j * (actual_phase_axial - phase_axial)).to(torch.complex64) * circ_final_bfp
            ef_bfp = ef_bfp * torch.exp(1j * (actual_phase_axial).to(torch.complex64)) * circ_final_bfp / ((ef_bfp_nocirc) * torch.exp(1j * (-phase_lateral).to(torch.complex64)))

            ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)

            psf_field = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))

            psf_on_axis = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ebfp_on_axis * circ_final_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))

            psf = torch.abs(psf_field) ** 2
            psfs = psf / torch.sum(torch.abs(psf), dim=(1, 2), keepdims=True) * photons


            MEAN = psfs.mean()
            STD = psfs.std()
            # MAX = psfs.max()
            psfs = torch.where(psfs>MEAN + 0*STD, psfs, 0)  # are we sure?

            # blur (batched) (to validate!

            #g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
            #sigma = g_sigma[0] + torch.rand(1).to(self.device) * (g_sigma[1] - g_sigma[0])
            #blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
            sigma = self.g_sigma[0] + torch.rand(1).to(self.device) * (self.g_sigma[1] - self.g_sigma[0])
            blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))

            #g_sigma = torch.tensor(self.g_sigma[0]).to(self.device)
            #blur_kernel = 1 / (2 * pi * g_sigma ** 2) * ( torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / g_sigma ** 2))
            psfs = F.conv2d(psfs.unsqueeze(1),blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs),padding='same').squeeze(1)

            # psfs = torch.where(psfs>MAX*0.5, psfs, 0)
            # psfs = psf / torch.sum(torch.abs(psf_on_axis) ** 2, dim=(1, 2), keepdims=True) * photons  # 22/12/2025 - normalize against on axis?

            #psfs = psfs / torch.sum(torch.abs(psfs), dim=(1, 2), keepdims=True) * photons
            psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
            #return psfs

            return psfs[0].detach().cpu().numpy().astype(np.float32)
    '''
    #end ori's edit
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.g_sigma = param_dict['g_sigma']
        self.baseline = param_dict['baseline']
        self.read_std = param_dict['read_std']
        self.non_uniform_noise_flag = param_dict['non_uniform_noise_flag']
        H, W = param_dict['H'], param_dict['W']
        self.H = H
        self.W = W
        self.non_uniform_noise = NonUniformBg(HW=(H, W), xy_offset=(W/5, H/5), angle_range=(-pi / 4, pi / 4))
        self.bitdepth = param_dict['bitdepth']
        # ori's edit
        #self.mask_offset_in_um = 40000*1  #38000 * 1 #float(param_dict.get('mask_offset_in_um', 0.0))  # +z from BFP to mask (um), air
        self.mask_offset_in_um = float(param_dict.get('mask_offset_in_um', 0.0))  # added on 2/01/2026 for mask displacement optimization
        self.centralBeadCoordinates_pixel = param_dict.get('centralBeadCoordinates_pixel', [self.H / 2, self.W / 2])  # added on 2/01/2026 for mask displacement optimization

        self.lamda = param_dict['lamda']
        self.ps_camera = param_dict['ps_camera']
        self.M = param_dict['M']
        self.n_sample = param_dict['n_sample']
        self.n_immersion = param_dict['n_immersion']
        self.ps_BFP = param_dict['ps_BFP']
        #self.centralBeadCoordinates_pixel = [849, 854] #test2 param_dict['centralBeadCoordinates_pixel']
        #self.centralBeadCoordinates_pixel = [965, 753] #fov6
        #self.centralBeadCoordinates_pixel = [562, 600]  # Mitochondria_flat_from_January2026
        #self.centralBeadCoordinates_pixel = [561, 647]  # Mitochondria_from_21January2026

        # adding amplitude of bfp 22/12/2025
        self.f_4f = param_dict['f_4f']
        self.NA = param_dict['NA']
        #self.circ_scaled = param_dict['circ_scale'] * param_dict['circ']
        #self.centralBeadCoordinates_pixel = [849, 545] #param_dict['centralBeadCoordinates_pixel'] flipped !
        # end ori's edit
    def blur_kernels(self, Nemitters):
        std_min, std_max = self.g_sigma
        stds = (std_min + (std_max - std_min) * torch.rand((Nemitters, 1))).to(self.device)
        gaussian_kernels = [torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / stds[i] ** 2) for i in range(Nemitters)]  
        gaussian_kernels = [kernel/kernel.sum() for kernel in gaussian_kernels] # normalization
        gaussian_kernels = torch.stack(gaussian_kernels)
        return gaussian_kernels


        '''
        def get_psfs(self, xyzps):  # each batch can only have the same number of particles
            # xyzps: tensor, rank 2 [x, 4]
            phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
            phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.NFPgrid * self.NFP
            ef_bfp = self.circ * torch.exp(1j * (phase_axial + phase_lateral + self.phase_mask))
            psf_field = fft.fftshift(fft.fftn(fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))  # FT
            psfs = torch.abs(psf_field) ** 2
            # photon normalization
            psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)
            # crop
            psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
            return psfs
        '''
        # ori's edit
        def get_psfs(self, xyzps):
            # Geometric phases from emitter (x,y,z)
            import torch
            import torch.nn.functional as F
            from math import pi

            xx = np.linspace(-0.5, 0.5, self.circ.shape[1]) * self.circ.shape[1]
            yy = np.linspace(-0.5, 0.5, self.circ.shape[0]) * self.circ.shape[0]
            XX, YY = np.meshgrid(xx, yy)
            circ_mask = torch.ones_like(self.circ)
            circ_mask[((XX) ** 2 + (YY) ** 2 > (5300 / self.ps_BFP) ** 2)] = 0
            # circ_mask[((XX)**2+(YY)**2 > (6000/self.ps_BFP)**2)] = 0
            # circ_new[((XX+40)**2+(YY-40)**2 < (130/2)**2)] = 1
            # circ_new[((XX-40)**2+(YY+40)**2 < (130/2)**2)] = 1

            xyzp = torch.from_numpy(xyzp_np).to(
                self.device)  # here it is um in object space (relative? probably  center removal is not required)
            # pixel coordinates
            x_pix = xyzp[0:1] * self.M / self.ps_camera  # * self.ps_camera
            y_pix = xyzp[1:2] * self.M / self.ps_camera  # * self.ps_camera

            # optical axis in pixels (col, row)
            cx = self.centralBeadCoordinates_pixel[1]
            cy = self.centralBeadCoordinates_pixel[0]

            # central correction: the input center is now center of frame rather than top-left
            cx = self.W/2 - cx
            cy = self.H/2 - cy

            # pixel offset from optical axis
            x_pix_rel = x_pix  - cx #
            y_pix_rel = y_pix  - cy

            # convert to sample-plane micrometers
            x = x_pix_rel * self.ps_camera / self.M  #
            y = y_pix_rel * self.ps_camera / self.M

            z = xyzp[2:3]

            photons = xyzp[3:4]

            # --- finite differences (center pixel) ---
            i0 = self.Xgrid.shape[0] // 2
            j0 = self.Xgrid.shape[1] // 2
            dX_col = (self.Xgrid[i0, j0 + 1] - self.Xgrid[i0, j0]).abs()  # phase coef step per +1 col
            dY_row = (self.Ygrid[i0 + 1, j0] - self.Ygrid[i0, j0]).abs()  # phase coef step per +1 row

            # true per-pixel phase increments
            dphi_col = dX_col * i0  # ~ |ΔX|*|x|
            dphi_row = dY_row * j0  # ~ |ΔY|*|y|

            # cycles/pixel
            ax = (dphi_col / (2 * torch.pi)).item()
            ay = (dphi_row / (2 * torch.pi)).item()

            ax = np.floor(np.abs(ax)) * 2 + 1
            ay = np.floor(np.abs(ay)) * 2 + 1
            s = int(np.sqrt(ax ** 2 + ay ** 2))  # ori's edit from 16/12/2025! to fix error in edges

            # --- scale lateral tilt & distance only ---
            x_s = x / s
            y_s = y / s
            z_s = xyzp[2:3] / s  # note -should be absolute value??
            d = self.mask_offset_in_um  # self.d_um()  # torch scalar, has grad
            d_eff = d * s
            # print('Debug!! d=' + str(d))

            # calculating round phase shift to keep sub pixel shift

            #

            # phases with scaled lateral tilt, original axial
            NFPs_b = torch.tensor(self.NFP).to(self.device).view(-1, 1, 1)
            NFPs_s = NFPs_b / s

            phase_axial = (self.Zgrid * z_s + self.NFPgrid * NFPs_s)
            actual_phase_axial = (self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFPs_b)
            phase_lateral = self.Xgrid * x_s.unsqueeze(1) + self.Ygrid * y_s.unsqueeze(1)

            # sub pixel phase
            #x_s_round = torch.floor(x_s * self.ps_camera) / self.ps_camera
            #y_s_round = torch.floor(y_s * self.ps_camera) / self.ps_camera
            #phase_lateral_round = self.Xgrid * x_s_round.unsqueeze(1) + self.Ygrid * y_s_round.unsqueeze(1)
            #phase_lateral_sub_pixel = phase_lateral - phase_lateral_round

            x_sub = x - torch.round(x * self.M / self.ps_camera) * self.ps_camera / self.M
            y_sub = y - torch.round(y * self.M / self.ps_camera) * self.ps_camera / self.M
            phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)
            phase_lateral_sub_pixel = phase_lateral_sub_pixel
            # adding amplitude of bfp 22/12/2025

            inx1 = torch.where(self.circ[np.shape(self.circ)[0] // 2, :] == 1)
            inx1 = inx1[0][0]
            inx1 = (np.shape(self.circ)[0] / 2 - inx1).to(self.device)

            x_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
            x_norm = x_phys / inx1
            y_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
            y_norm = y_phys / inx1

            # xx, yy = meshgrid(x_norm, y_norm); need to do meshgrid
            # rho2 = xx ** 2 + yy ** 2
            rho2 = x_norm ** 2 + y_norm ** 2
            amp = 1 / torch.abs(((1 - rho2 * (self.NA / self.n_immersion) ** 2 + 1e-16)) ** (1 / 4))
            ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)
            ef_bfp[torch.isnan(ef_bfp)] = 0
            ebfp_on_axis = torch.exp(1j * (actual_phase_axial)).to(torch.complex64)

            # forward propagate
            if d_eff != 0.0:
                ef_off = asm_propagate(ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0, bandlimit=True)
            else:
                ef_off = ef_bfp

            # mask & circ
            phase = torch.exp(1j * self.phase_mask.to(ef_off.device).to(torch.float32))

            # circ_phase = (self.circ_scaled > 0.5).unsqueeze(0)
            circ_phase = (circ_mask > 0.5).unsqueeze(0)

            ef_off = ef_off * phase * circ_phase  # self.circ_scaled
            ef_off = torch.where(circ_phase, ef_off, 0)

            # back propagate
            if d_eff != 0.0:
                ef_bfp = asm_propagate(ef_off, self.lamda, self.ps_BFP, self.ps_BFP, -d_eff, n=1.0, bandlimit=True).to(
                    torch.complex64)
            else:
                ef_bfp = ef_off

            circ_final_bfp = self.circ  # self.circ or *self.circ_sample
            # remove only the (scaled) lateral phase before FFT (same as your original flow)
            ef_bfp = ef_bfp * torch.exp(1j * (-phase_lateral + phase_lateral_sub_pixel)).to(torch.complex64)  # * self.circ
            ef_bfp = ef_bfp * torch.exp(1j * (actual_phase_axial - phase_axial)).to(torch.complex64) * circ_final_bfp
            ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)

            psf_field = torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2)
            )

            psf_on_axis = torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(ebfp_on_axis * circ_final_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2)
            )

            psf = torch.abs(psf_field) ** 2
            psfs = psf / torch.sum(torch.abs(psf), dim=(1, 2), keepdims=True) * photons
            # psfs = psf / torch.sum(torch.abs(psf_on_axis) ** 2, dim=(1, 2),
            #                       keepdims=True) * photons  # 22/12/2025 - normalize against on axis?

            # blur (batched) (to validate!

            # g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
            # sigma = g_sigma[0] + torch.rand(1).to(self.device) * (g_sigma[1] - g_sigma[0])

            # blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
            sigma = self.g_sigma[0] + torch.rand(1).to(self.device) * (self.g_sigma[1] - self.g_sigma[0])
            blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))

            # g_sigma = torch.tensor(self.g_sigma[0]).to(self.device)
            # blur_kernel = 1 / (2 * pi * g_sigma ** 2) * ( torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / g_sigma ** 2))
            psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs),
                            padding='same').squeeze(1)

            psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
            # return psfs

            return psfs[0].detach().cpu().numpy().astype(np.float32)
            '''
            phase_lateral = self.Xgrid * (xyzps[:, 0:1].unsqueeze(1)) + self.Ygrid * (xyzps[:, 1:2].unsqueeze(1))
            phase_axial = self.Zgrid * (xyzps[:, 2:3].unsqueeze(1)) + self.NFPgrid * self.NFP

            # 1) emitter field at the BFP: **no mask, no aperture**
            ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)

            # 2) propagate BFP -> mask plane (AIR)
            d = float(self.mask_offset_in_um)

            if d != 0.0:
                ef_off = asm_propagate(
                    ef_bfp,
                    wavelength_um=self.lamda,
                    dx_um=self.ps_BFP, dy_um=self.ps_BFP,
                    z_um=+d,
                    n=1.0,  # AIR
                    bandlimit=True
                )
            else:
                ef_off = ef_bfp

            # 3) multiply by phase mask (as-is) and aperture (circ) at that plane
            #    (no resizing or cropping of the mask)
            phase = torch.exp(1j * self.phase_mask.to(ef_off.device).to(torch.float32))
            ef_off = ef_off * phase * self.circ

            # 4) propagate mask plane -> back to BFP (AIR)
            if d != 0.0:
                ef_bfp = asm_propagate(
                    ef_off,
                    wavelength_um=self.lamda,
                    dx_um=self.ps_BFP, dy_um=self.ps_BFP,
                    z_um=-d,
                    n=1.0,  # AIR
                    bandlimit=True
                ).to(torch.complex64)
            else:
                ef_bfp = ef_off

            # 5) BFP -> image plane (Fourier transform)
            psf_field = torch.fft.fftshift(
                torch.fft.fftn(torch.fft.ifftshift(ef_bfp* self.circ, dim=(1, 2)), dim=(1, 2)),
                dim=(1, 2)
            )
            psfs = torch.abs(psf_field) ** 2

            # photon normalization
            psfs = psfs / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(1)

            # crop to requested FOV
            psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
            '''
            return psfs

        #
    def forward(self, xyzps):
        """
        image of point sources
        :param xyzps: spatial locations and photon counts, tensor, rank 2 [n 4]
        :return: tensor, image
        """
        psfs = self.get_psfs(xyzps)  # after emitter-wise normalization and cropping
        '''
        # blur, emitter-wise
        blur_kernels = self.blur_kernels(psfs.shape[0]).unsqueeze(0)
        im = F.conv2d(psfs.unsqueeze(0), blur_kernels, padding='same').squeeze()
        im = psfs
        # noise: background, shot, readout
        #im = torch.poisson(im + self.bg)  # rounded
        im = torch.poisson(im)  # rounded
        '''
        im = torch.sum(psfs, dim=0)


        read_baseline = self.baseline[0] + torch.rand(1, device=self.device) * (self.baseline[1] - self.baseline[0])
        read_std = self.read_std[0] + torch.rand(1, device=self.device) * (self.read_std[1] - self.read_std[0])

        if self.non_uniform_noise_flag:
            # choose a range in the preset read_std range to reshape the non-uniform distribution
            std_pv = torch.rand(1, device=self.device) * (self.read_std[1] - read_std)  # read_std--valley
            std = self.non_uniform_noise().to(self.device) * std_pv + read_std
            try:
                im = im + torch.round(read_baseline + torch.randn(im.shape, device=self.device) * std)
            except:  im = im + torch.round(read_baseline + torch.randn(im.shape, device=self.device) * read_std) # ori's edit

        else:
            im = im + torch.round(read_baseline + torch.randn(im.shape, device=self.device) * read_std)

        im[im < 0] = 0
        max_adu = 2 ** self.bitdepth - 1
        im[im > max_adu] = max_adu
        im = im.type(torch.int32) 

        return im


def calculate_cc(output, target):
    # output: rank 3, target: rank 3
    output_mean = np.mean(output, axis=(1, 2), keepdims=True)
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    ccs = (np.sum((output - output_mean) * (target - target_mean), axis=(1, 2)) /
           (np.sqrt(np.sum((output - output_mean) ** 2, axis=(1, 2)) * np.sum((target - target_mean) ** 2,
                                                                              axis=(1, 2))) + 1e-9))
    return ccs



class Sampling():
    def __init__(self, params):
        # define the reconstruction domain
        self.D = params['D']  # voxel number in z
        self.HH = params['HH']  # voxel number in y
        self.WW = params['WW']  # voxel number in x
        self.buffer_HH = params['buffer_HH']  # buffer in y, place Gaussian blobs and avoid PSF cropping
        self.buffer_WW = params['buffer_WW']  # buffer in x, place Gaussian blobs and avoid PSF cropping
        self.vs_xy, self.vs_z = params['vs_xy'], params['vs_z']
        self.zrange = params['zrange']

        self.Nsig_range = params['Nsig_range']  # photon count range
        self.num_particles_range = params['num_particles_range']  # emitter count range
        self.blob_maxv = params['blob_maxv']
        self.blob_r = params['blob_r']

        # define Gaussian blobs
        self.sigma = params['blob_sigma']
        pn = self.blob_r*2+1  # the number of pixels of the Gaussian blob
        xs = np.linspace(-self.blob_r, self.blob_r, pn)
        self.zz, self.yy, self.xx = np.meshgrid(xs, xs, xs, indexing='ij')
        self.normal_factor1 = 1 / (np.sqrt(2 * pi * self.sigma ** 2)) ** 3
        self.normal_factor2 = self.blob_maxv / self.Nsig_range[1]

    def xyzp_batch(self):  # one batch

        num_particles = np.random.randint(self.num_particles_range[0], self.num_particles_range[1]+1)

        # integers at center of voxels, starting from 0
        x_ids = np.random.randint(self.buffer_WW, self.WW - self.buffer_WW, num_particles)
        y_ids = np.random.randint(self.buffer_HH, self.HH - self.buffer_HH, num_particles)
        z_ids = np.random.randint(0, self.D, num_particles)
        xyz_ids = np.c_[x_ids, y_ids, z_ids]  # where to place 3D Gaussian blobs

        x_local = np.random.uniform(-0.49, 0.49, num_particles)
        y_local = np.random.uniform(-0.49, 0.49, num_particles)
        z_local = np.random.uniform(-0.49, 0.49, num_particles)
        xyz_local = np.c_[x_local, y_local, z_local]

        xyz = xyz_ids+xyz_local  # voxel

        xyz[:, 0] = (xyz[:, 0] - (self.WW-1) / 2) * self.vs_xy
        xyz[:, 1] = (xyz[:, 1] - (self.HH-1) / 2) * self.vs_xy
        xyz[:, 2] = (xyz[:, 2]+0.5) * self.vs_z + self.zrange[0]

        Nphotons = np.random.randint(self.Nsig_range[0], self.Nsig_range[1], num_particles)
        xyzps = np.c_[xyz, Nphotons]

        blob3d = np.exp(-0.5 * ((self.xx - xyz_local[:, 0, np.newaxis, np.newaxis, np.newaxis]) ** 2 +
                                (self.yy - xyz_local[:, 1, np.newaxis, np.newaxis, np.newaxis]) ** 2 +
                                (self.zz - xyz_local[:, 2, np.newaxis, np.newaxis, np.newaxis]) ** 2) / self.sigma ** 2)

        # blob3d = blob3d * self.normal_factor1 * xyzps[:, 3][:, np.newaxis, np.newaxis, np.newaxis]
        blob3d = blob3d * self.normal_factor2 * xyzps[:, 3][:, np.newaxis, np.newaxis, np.newaxis]

        return xyzps, xyz_ids, blob3d


    def show_volume(self, ):
        _, xyz_ids, blob3d = self.xyzp_batch()
        y = np.zeros((self.D, self.HH, self.WW))
        # assemble the representation of emitters
        y = np.pad(y, self.blob_r)
        for i in range(xyz_ids.shape[0]):
            xidx, yidx, zidx = xyz_ids[i, 0], xyz_ids[i, 1], xyz_ids[i, 2]
            y[zidx:zidx + 2 * self.blob_r + 1, yidx:yidx + 2 * self.blob_r + 1, xidx:xidx + 2 * self.blob_r + 1] += blob3d[i]
        y = y[self.blob_r:-self.blob_r, self.blob_r:-self.blob_r, self.blob_r:-self.blob_r]

        xy_proj = np.max(y, axis=0)
        xz_proj = np.max(y, axis=1)
        # yz_proj = np.max(y, axis=2)

        plt.figure(figsize=(4, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(xy_proj)
        plt.title('xy max projection')

        plt.subplot(2, 1, 2)
        plt.imshow(xz_proj)
        plt.title('xz max projection')

        # plt.show()
        plt.savefig('volume_projection.jpg', bbox_inches='tight', dpi=300)
        plt.clf()
        print('Volume (network output) example: volume_projection.jpg')

class Volume2XYZ(nn.Module):
    def __init__(self, params):
        super().__init__()
        # define the reconstruction volume
        self.blob_r = params['blob_r']  # buffer in z, place Gaussian blobs, radius of 3D gaussian blobs
        self.vs_xy = params['vs_xy']
        self.vs_z = params['vs_z']
        self.zrange = params['zrange']
        self.threshold = params['threshold']
        self.device = params['device']

        self.r = self.blob_r  # radius of the blob
        self.maxpool = MaxPool3d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)  # removed on 22/04/2026
        #self.maxpool_xy = MaxPool3d(kernel_size=(1, 2 * self.r + 1, 2 * self.r + 1), stride=1, padding=(0, self.r, self.r))  # for average weighting added on 22/04/2026

        #self.maxpool = MaxPool3d(kernel_size=1, stride=1, padding=1)  # try 1 next
        # added on 19/04/2026 to prevent z section preference
        '''
        rz = 4
        rxy = self.r
        self.maxpool = MaxPool3d(kernel_size=(2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), stride=1, padding=(rz, rxy, rxy))
        '''
        # end  19/04/2026

        self.pad = ConstantPad3d(self.r, 0.0)
        self.zero = torch.FloatTensor([0.0]).to(self.device)
        # ori's edit

        self.zrange = params['zrange']
        self.vs_z = params['vs_z']
        print("Volume2XYZ zrange, vs_z:", self.zrange, self.vs_z)
        #
        # construct the local average filters
        filt_vec = np.arange(-self.r, self.r + 1)
        yfilter, zfilter, xfilter = np.meshgrid(filt_vec, filt_vec, filt_vec)
        xfilter = torch.FloatTensor(xfilter).unsqueeze(0).unsqueeze(0)
        yfilter = torch.FloatTensor(yfilter).unsqueeze(0).unsqueeze(0)
        zfilter = torch.FloatTensor(zfilter).unsqueeze(0).unsqueeze(0)
        sfilter = torch.ones_like(xfilter)
        self.local_filter = torch.cat((sfilter, xfilter, yfilter, zfilter), 0).to(self.device)

        # blob catch
        offsets = torch.arange(0, self.r * 2 + 1, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
        self.grid_z = grid_z.flatten()
        self.grid_y = grid_y.flatten()
        self.grid_x = grid_x.flatten()

    def local_avg(self, xbool, ybool, zbool, pred_vol_pad):
        num_pts = len(zbool)
        all_z = zbool.unsqueeze(1) + self.grid_z
        all_y = ybool.unsqueeze(1) + self.grid_y
        all_x = xbool.unsqueeze(1) + self.grid_x
        pred_vol_all_ = pred_vol_pad[0][all_z, all_y, all_x].view(num_pts, self.r*2+1, self.r*2+1, self.r*2+1)

        conf_rec = torch.sum(pred_vol_all_, dim=(1, 2, 3))   # sum of the 3D sub-volume

        pred_vol_all = pred_vol_all_.unsqueeze(1)
        # convolve it using conv3d
        sums = conv3d(pred_vol_all, self.local_filter)
        # squeeze the sums and convert them to local perturbations
        xloc = torch.squeeze(sums[:, 1] / sums[:, 0])
        yloc = torch.squeeze(sums[:, 2] / sums[:, 0])
        zloc = torch.squeeze(sums[:, 3] / sums[:, 0])
        return xloc, yloc, zloc, conf_rec

    def forward(self, pred_vol, return_voxel_coords=False):
        # added on  19/04/2026 to prevent z section preference
        '''
        kz = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=pred_vol.device, dtype=pred_vol.dtype)  # make it more parameter based?
        kz = kz / kz.sum()
        kz = kz.view(1, 1, 5, 1, 1)
        pred_vol = F.conv3d(pred_vol.unsqueeze(1), kz, padding=(2, 0, 0)).squeeze(1)
        '''
        # end  19/04/2026
        # threshold
        # added on 22/04/2026 for average weightening instead of maxpooling:
        '''pred_thresh = torch.where(pred_vol > self.threshold, pred_vol, self.zero)

        # collapse z -> xy evidence map
        xy_map = torch.max(pred_thresh, dim=1, keepdim=True)[0]  # [B,1,H,W]

        # xy-only non-maximum suppression
        xy_map_3d = xy_map.unsqueeze(2)  # [B,1,1,H,W]
        xy_peaks = self.maxpool_xy(xy_map_3d)
        xy_peaks = torch.where(
            (xy_peaks > self.zero) & (xy_peaks == xy_map_3d),
            xy_peaks,
            self.zero
        )

        xy_peaks = torch.squeeze(xy_peaks)  # [H,W]
        batch_indices = torch.nonzero(xy_peaks, as_tuple=True)
        ybool, xbool = batch_indices[0], batch_indices[1]
        #end '''
        # 22/04/2026 for maxpooling:
        pred_thresh = torch.where(pred_vol > self.threshold, pred_vol, self.zero)

        # apply the 3D maxpooling to find local maxima
        conf_vol = self.maxpool(pred_thresh)  # removed on 20/04/2026. maxpool is suspected to cause the comb
        #conf_vol = pred_thresh
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)  # ~0.001s
        conf_vol = torch.squeeze(conf_vol)
        batch_indices = torch.nonzero(conf_vol, as_tuple=True)  # ~0.006s  indices of nonzero elements
        zbool, ybool, xbool = batch_indices[0], batch_indices[1], batch_indices[2]
        # end 22/04/2026 for maxpooling
        # if the prediction is empty return None otherwise convert to list of locations
        if len(zbool) == 0:  #  22/04/2026 for maxpooling
            #if len(ybool) == 0:  # added on 22/04/2026 for weightening instead of maxpooling
            xyz_rec = None
            conf_rec = None
            voxel_ids = None
            voxel_locals = None
        else:
            '''# added on 22/04/2026 for weightening instead of maxpooling:
            pred_vol_pad = self.pad(pred_vol) 

            D = pred_vol.shape[1]
            HH = pred_vol.shape[2]
            WW = pred_vol.shape[3]

            xloc_list = []
            yloc_list = []
            zrec_vox_list = []
            conf_rec_list = []

            for i in range(len(xbool)):
                xb = xbool[i]
                yb = ybool[i]

                # local cube across all z around this xy
                patch = pred_vol_pad[
                        0,  # batch
                        :,  # all z
                        yb:yb + 2 * self.r + 1,
                        xb:xb + 2 * self.r + 1
                        ]  # shape: [D+2r, 2r+1, 2r+1]

                # remove z padding
                patch = patch[:, :, :]
                z_profile = patch.sum(dim=(1, 2))  # [D+2r]
                z_profile = z_profile[self.r:self.r + D]

                z_profile = torch.clamp(z_profile, min=0.0)
                s = z_profile.sum()

                if s <= 0:
                    z_soft = torch.tensor(0.0, device=self.device, dtype=pred_vol.dtype)
                    conf = torch.tensor(0.0, device=self.device, dtype=pred_vol.dtype)
                else:
                    z_idx = torch.arange(D, device=self.device, dtype=pred_vol.dtype)
                    z_soft = (z_idx * z_profile).sum() / s
                    conf = z_profile.max()

                # optional xy local refinement using z-max projection in local patch
                xy_local = patch[self.r:self.r + D].sum(dim=0)  # [2r+1, 2r+1]
                xy_sum = xy_local.sum()

                if xy_sum <= 0:
                    xloc = torch.tensor(0.0, device=self.device, dtype=pred_vol.dtype)
                    yloc = torch.tensor(0.0, device=self.device, dtype=pred_vol.dtype)
                else:
                    offs = torch.arange(-self.r, self.r + 1, device=self.device, dtype=pred_vol.dtype)
                    yy, xx = torch.meshgrid(offs, offs, indexing="ij")
                    xloc = (xy_local * xx).sum() / xy_sum
                    yloc = (xy_local * yy).sum() / xy_sum

                xloc_list.append(xloc)
                yloc_list.append(yloc)
                zrec_vox_list.append(z_soft)
                conf_rec_list.append(conf)

            xloc = torch.stack(xloc_list)
            yloc = torch.stack(yloc_list)
            zrec_vox = torch.stack(zrec_vox_list)
            conf_rec = torch.stack(conf_rec_list)

            xrec = (xbool + xloc - ((WW - 1) / 2)) * self.vs_xy
            yrec = (ybool + yloc - ((HH - 1) / 2)) * self.vs_xy
            zrec = (zrec_vox + 0.5) * self.vs_z + self.zrange[0]

            xyz_rec = torch.stack((xrec, yrec, zrec), dim=1).cpu().numpy()
            conf_rec = conf_rec.cpu().numpy()
            #end '''
            # 22/04/2026  # pad the result with radius_px 0's for average calc. for maxpooling? not sure:
            pred_vol_pad = self.pad(pred_vol)
            # for each point calculate local weighted average
            xloc, yloc, zloc, conf_rec_sum = self.local_avg(xbool, ybool, zbool, pred_vol_pad)  # ~0.001

            D, HH, WW = conf_vol.size()
            # calculate the recovered positions assuming mid-voxel
            xrec = (xbool + xloc - ((WW-1) / 2)) * self.vs_xy  # shift the center
            yrec = (ybool + yloc - ((HH-1) / 2)) * self.vs_xy  # shift the center
            zrec = (zbool + zloc + 0.5) * self.vs_z + self.zrange[0]
            xyz_rec = torch.stack((xrec, yrec, zrec), dim=1).cpu().numpy()

            conf_rec = conf_vol[zbool, ybool, xbool]  # use the peak
            conf_rec = conf_rec.cpu().numpy()  # conf_rec is the sum of each 3D blob

            if return_voxel_coords:
                # zloc/yloc/xloc can collapse to 0-d scalars via torch.squeeze when there's
                # exactly one detection -- reshape(-1) normalizes them back to length-1 1-D
                # before stacking, which torch.stack(dim=1) requires.
                voxel_ids = torch.stack((zbool, ybool, xbool), dim=1).cpu().numpy()
                voxel_locals = torch.stack(
                    (zloc.reshape(-1), yloc.reshape(-1), xloc.reshape(-1)), dim=1
                ).cpu().numpy()
            else:
                voxel_ids = None
                voxel_locals = None

        if return_voxel_coords:
            return xyz_rec, conf_rec, voxel_ids, voxel_locals
        return xyz_rec, conf_rec


        # added on  19/04/2026 to prevent z section preference
        #
        #kz = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=pred_vol.device, dtype=pred_vol.dtype)  # make it more parameter based?
        #kz = kz / kz.sum()
        #kz = kz.view(1, 1, 5, 1, 1)
        #pred_vol = F.conv3d(pred_vol.unsqueeze(1), kz, padding=(2, 0, 0)).squeeze(1)
        #
        #end  19/04/2026
        ''' # original version?
        # removed on 22/04/2026 - replaced maxpool with weight averaging. for maxpooling:
        # threshold
        pred_thresh = torch.where(pred_vol > self.threshold, pred_vol, self.zero)

        # apply the 3D maxpooling to find local maxima
        #conf_vol = self.maxpool(pred_thresh)  # removed on 20/04/2026. maxpool is suspected to cause the comb
        conf_vol = pred_thresh
        conf_vol = torch.where((conf_vol > self.zero) & (conf_vol == pred_thresh), conf_vol, self.zero)  # ~0.001s
        conf_vol = torch.squeeze(conf_vol)
        batch_indices = torch.nonzero(conf_vol, as_tuple=True)  # ~0.006s  indices of nonzero elements
        zbool, ybool, xbool = batch_indices[0], batch_indices[1], batch_indices[2]

        # if the prediction is empty return None otherwise convert to list of locations
        if len(zbool) == 0:
            xyz_rec = None
            conf_rec = None
        else:
            # pad the result with radius_px 0's for average calc.
            pred_vol_pad = self.pad(pred_vol)
            # for each point calculate local weighted average
            xloc, yloc, zloc, conf_rec_sum = self.local_avg(xbool, ybool, zbool, pred_vol_pad)  # ~0.001

            D, HH, WW = conf_vol.size()
            # calculate the recovered positions assuming mid-voxel
            xrec = (xbool + xloc - ((WW-1) / 2)) * self.vs_xy  # shift the center
            yrec = (ybool + yloc - ((HH-1) / 2)) * self.vs_xy  # shift the center
            zrec = (zbool + zloc + 0.5) * self.vs_z + self.zrange[0]
            xyz_rec = torch.stack((xrec, yrec, zrec), dim=1).cpu().numpy()

            conf_rec = conf_vol[zbool, ybool, xbool]  # use the peak
            conf_rec = conf_rec.cpu().numpy()  # conf_rec is the sum of each 3D blob

        return xyz_rec, conf_rec '''

# added on 28/04/2026 to change training data image reading:
class MyDataset(Dataset):
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels, cache_info=None):

        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels

        self.r = labels['blob_r']
        self.maxv = labels['blob_maxv']
        self.volume_size = labels['volume_size']

        # tile-dependence info
        self.tile_grid = labels.get('tile_grid', (1, 1))
        self.camera_size_px = labels.get('camera_size_px', None)

        # memmap cache info
        self.cache_info = cache_info
        self.use_memmap = bool(cache_info is not None and cache_info.get('enabled', False))

        if self.use_memmap:
            self.memmap_path = cache_info['data_path']
            self.memmap_shape = tuple(cache_info['shape'])
            self.memmap_dtype = np.dtype(cache_info['dtype'])
            self.id_to_idx = {fname: i for i, fname in enumerate(cache_info['ids'])}
            self._x_memmap = None
        else:
            self.memmap_path = None
            self.memmap_shape = None
            self.memmap_dtype = None
            self.id_to_idx = {}
            self._x_memmap = None

        print("[TRAIN] camera_size_px in labels:", self.camera_size_px)
        print("[TRAIN] tile_grid:",  self.tile_grid)
        print("[TRAIN] use_memmap:", self.use_memmap)

    def _get_x_memmap(self):
        if self._x_memmap is None:
            self._x_memmap = np.memmap(
                self.memmap_path,
                mode='r',
                dtype=self.memmap_dtype,
                shape=self.memmap_shape
            )
        return self._x_memmap
    # end: 28/04/2026 to change training data image reading

    '''
#  removed on 28/04/2026 to change training data image reading
class MyDataset(Dataset):
    # initialization of the dataset
    def __init__(self, root_dir, list_IDs, labels):

        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.labels = labels

        self.r = labels['blob_r']
        self.maxv = labels['blob_maxv']
        self.volume_size = labels['volume_size']
        # NEW Ori's edit 12/08/2025 for tile dependence training:
        self.tile_grid = labels.get('tile_grid', (1, 1))
        self.camera_size_px = labels.get('camera_size_px', None)

        # 07/02/2026
        print("[TRAIN] camera_size_px in labels:", self.camera_size_px)
        print("[TRAIN] tile_grid:",  self.tile_grid)
        #print("[TRAIN] labels camera_size_px =", labels_dict.get("camera_size_px", None))

        # end 07/02/2026
    #  end: 28/04/2026 to change training data image reading
    
        # end Ori's edit
    # total number of samples in the dataset '''
    def __len__(self):
        return len(self.list_IDs)

    # sampling one example from the data
    def __getitem__(self, index):
        # added on 28/04/2026 to change training data image reading:
        # select sample
        ID = self.list_IDs[index]

        # load image from memmap cache if available; otherwise fall back to TIFF
        if self.use_memmap and ID in self.id_to_idx:
            x = np.asarray(self._get_x_memmap()[self.id_to_idx[ID]])
        else:
            x_file = os.path.join(self.root_dir, ID)
            x = io.imread(x_file)

        x = x[np.newaxis, :, :].astype(np.float32)
        # end: added on 28/04/2026 to change training data image reading:

        '''
        # removed on 28/04/2026 to change training data image reading:
        # select sample
        ID = self.list_IDs[index]
        # load tiff image
        #import time
        #t0 = time.perf_counter()
        x_file = os.path.join(self.root_dir, ID)
        x = io.imread(x_file)
        #t1 = time.perf_counter()

        x = x[np.newaxis, :, :].astype(np.float32)
        # end: removed 28/04/2026 to change training data image reading:
        '''
        y = np.zeros(self.volume_size)
        y = np.pad(y, self.r)
        xyz_ids, blob3d = self.labels[ID][0], self.labels[ID][1]

        for i in range(xyz_ids.shape[0]):
            #xidx, yidx, zidx = xyz_ids[i, 0], xyz_ids[i, 1], xyz_ids[i, 2]
            #y[zidx:zidx + 2 * self.r + 1, yidx:yidx + 2 * self.r + 1, xidx:xidx + 2 * self.r + 1] += blob3d[i]

            #ori's edit
            xidx, yidx, zidx = xyz_ids[i, 0], xyz_ids[i, 1], xyz_ids[i, 2]
            xidx = int(xidx)
            yidx = int(yidx)
            zidx = int(zidx)
            # keep the existing padding + add logic

            # bounds check and skip bad emitters:
            # after computing all (zidx, yidx, xidx) arrays for this patch:
            Z, Y, X = y.shape
            patch_size = 2 * self.r + 1

            # ... inside your loop over emitters ...
            z0, y0, x0 = zidx, yidx, xidx
            z1, y1, x1 = z0 + patch_size, y0 + patch_size, x0 + patch_size
            if z0 < 0 or y0 < 0 or x0 < 0 or z1 > Z or y1 > Y or x1 > X:
                continue
            # if this emitter’s 3D PSF cube would go out of bounds, skip it
            if (
                    z0 < 0 or y0 < 0 or x0 < 0 or
                    z1 > Z or y1 > Y or x1 > X
            ):
                # optionally: log once to see how often this happens
                # print("Skipping emitter at", z0, y0, x0, "for volume", Z, Y, X)
                continue

            y[z0:z1, y0:y1, x0:x1] += blob3d[i]

            # end bound check

            #y[zidx:zidx + 2 * self.blob_r + 1, yidx:yidx + 2 * self.blob_r + 1, xidx:xidx + 2 * self.blob_r + 1] += blob3d[i]
            #y[zidx:zidx + 2 * self.r + 1, yidx:yidx + 2 * self.r + 1, xidx:xidx + 2 * self.r + 1] += blob3d[i]  erased on 24/12/2025 - bug

            #end ori's edit

        y = (y[self.r:-self.r, self.r:-self.r, self.r:-self.r]).astype(np.float32)
        #return x, y

        #t2 = time.perf_counter()
        #print(f"imread={t1 - t0:.4f}s  ybuild={t2 - t1:.4f}s  total={t2 - t0:.4f}s")




        # --- NEW: Ori's edit 09/12/2025  ~~~~~~~~~
        # for tile dependence training add coord channels for global FOV position ---
        # base image as (1,H,W)
        #x = x[np.newaxis, :, :]  # (1,H,W)
        
        H_patch, W_patch = x.shape[1], x.shape[2]

        # default: no tiling info
        num_tiles_y, num_tiles_x = self.tile_grid
        Hc, Wc = self.camera_size_px if self.camera_size_px is not None else (H_patch, W_patch)

        # parse FOV index from filename, if present
        # e.g. "00000_FOV_00003.tif" -> fov_idx = 3
        if "_FOV_" in ID:
            base, rest = ID.split("_FOV_")
            fov_idx = int(rest.split(".")[0])
        else:
            fov_idx = 0

        # row-major indexing
        tile_row = fov_idx // num_tiles_x
        tile_col = fov_idx % num_tiles_x

        # tile size (should match what you used in training_data_func)
        H_new = Hc // num_tiles_y
        W_new = Wc // num_tiles_x

        # top-left in full camera coordinates
        y0 = tile_row * H_new
        x0 = tile_col * W_new

        # grid of local pixel coords in this patch
        yy_local, xx_local = np.mgrid[0:H_patch, 0:W_patch]  # removed on 15/01/2026 to increase speed
        #coord = self.coord_cache[fov_idx]  # added on 15/01/2026 to increase speed
        #x_full = np.concatenate([x, coord], axis=0) #added on 15/01/2026 to increase speed
        yy_global = yy_local + y0
        xx_global = xx_local + x0

        # normalize to [-1,1] using full camera size
        Xmap = (xx_global / (Wc - 1) * 2 - 1).astype(np.float32)
        Ymap = (yy_global / (Hc - 1) * 2 - 1).astype(np.float32)

        coord = np.stack([Xmap, Ymap], axis=0)  # (2,H,W)

        # final input: 3 channels
        x_full = np.concatenate([x, coord], axis=0)  # (3,H,W)


        # added on 07/02/2026
        # --- DEBUG: coordinate ranges (prints once every ~500 samples) ---
        '''
        if (index % 500) == 0:
            print(f"[TRAIN_GETITEM] Hc,Wc={Hc},{Wc}  tile y0={y0} x0={x0}  "
                  f"X[{Xmap.min():+.3f},{Xmap.max():+.3f}] mean={Xmap.mean():+.3f}  "
                  f"Y[{Ymap.min():+.3f},{Ymap.max():+.3f}] mean={Ymap.mean():+.3f}")

        if (index % 500) == 0:
            xm, xM = Xmap.min().item(), Xmap.max().item()
            ym, yM = Ymap.min().item(), Ymap.max().item()
            print(f"[TRAIN] idx={index} Xmap[{xm:+.3f},{xM:+.3f}]  Ymap[{ym:+.3f},{yM:+.3f}]") '''
        # end 07/02/2026
        return x_full, y
        # ~~~~~~~~~ end 09/12/2025 ~~~~~~~~~~~~~~~~


class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


class LON(nn.Module):
    def __init__(self, D, us_factor, maxv):

        # Ori's edit 09/12/2025 for tile depending training
        super(LON, self).__init__()
        C = 64
        self.us_factor = us_factor

        in_ch = 3  # image + X + Y
        #self.norm = nn.BatchNorm2d(num_features=in_ch, affine=True)  # removed on 06/02/2026 to improve fov dependance training
        self.norm_img = nn.BatchNorm2d(num_features=1, affine=True)  # added on 06/02/2026 to improve fov dependance training

        self.layer1 = Conv2DLeakyReLUBN(in_ch, C, 3, 1, 1, 0.2)

        # everywhere you had C+1, change to C+in_ch (C+3)
        self.layer2 = Conv2DLeakyReLUBN(C + in_ch, C, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(C + in_ch, C, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DLeakyReLUBN(C + in_ch, C, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DLeakyReLUBN(C + in_ch, C, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = Conv2DLeakyReLUBN(C + in_ch, C, 3, (16, 16), (16, 16), 0.2)

        # deconv1 also sees features = cat(out, im)
        self.deconv1 = Conv2DLeakyReLUBN(C + in_ch, C, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(C, C, 3, 1, 1, 0.2)

        self.layer7 = Conv2DLeakyReLUBN(C, D, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(D, D, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(D, D, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(D, D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=maxv)
        # end Ori's edit
        '''
        super(LON, self).__init__()
        C = 64
        self.us_factor = us_factor
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, C, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(C + 1, C, 3, 1, 1, 0.2)
        self.layer3 = Conv2DLeakyReLUBN(C + 1, C, 3, (2, 2), (2, 2), 0.2)
        self.layer4 = Conv2DLeakyReLUBN(C + 1, C, 3, (4, 4), (4, 4), 0.2)
        self.layer5 = Conv2DLeakyReLUBN(C + 1, C, 3, (8, 8), (8, 8), 0.2)
        self.layer6 = Conv2DLeakyReLUBN(C + 1, C, 3, (16, 16), (16, 16), 0.2)
        self.deconv1 = Conv2DLeakyReLUBN(C + 1, C, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(C, C, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(C, D, 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(D, D, 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(D, D, 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(D, D, kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=maxv)
        '''

    def forward(self, im):
        # extract multi-scale features

        #im = self.norm(im)  # removed on 06/02/2026 to improve fov dependance training
        # added on 06/02/2026 to improve fov dependance training
        #im = im.clone()
        #im[:, 0:1] = self.norm_img(im[:, 0:1])  # normalize image only
        im0 = self.norm_img(im[:, 0:1])  # normalize only image channel
        im = torch.cat((im0, im[:, 1:]), dim=1)  # keep X/Y as-is

        #print('test applied')
        #end

        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out
        features = torch.cat((out, im), 1)

        if self.us_factor == 1:
            out = self.deconv1(features)
            out = self.deconv2(out)
        elif self.us_factor == 2:
            out = interpolate(features, scale_factor=2)
            out = self.deconv1(out)
            out = self.deconv2(out)
        elif self.us_factor == 4:
            out = interpolate(features, scale_factor=2)
            out = self.deconv1(out)
            out = interpolate(out, scale_factor=2)
            out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out


def GaussianKernel(shape=(7, 7, 7), sigma=1.0, normfactor=1):
    """
    3D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]) in 3D
    """
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    y, x, z = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma ** 2))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    """
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
        h = h * normfactor
    """
    maxh = h.max()
    if maxh != 0:
        h /= maxh
        h = h * normfactor

    h = torch.from_numpy(h).type(torch.float32)
    h = h.unsqueeze(0)
    h = h.unsqueeze(1)

    return h


''' old version. replaced on 19/04/2026
class KDE_loss3D(nn.Module):
    def __init__(self, sigma, device):
        super(KDE_loss3D, self).__init__()
        self.kernel = GaussianKernel(sigma=sigma).to(device)
'''

class KDE_loss3D(nn.Module):
    def __init__(self, sigma, device, tv_z_weight=0.0):
        super(KDE_loss3D, self).__init__()
        self.kernel = GaussianKernel(sigma=sigma).to(device)
        self.tv_z_weight = tv_z_weight
    def forward(self, pred_bol, target_bol):
        # extract kernel dimensions
        _, _, D, _, _ = self.kernel.size()

        # extend prediction and target to have a single channel
        target_bol = target_bol.unsqueeze(1)
        pred_bol = pred_bol.unsqueeze(1)

        # KDE for both input and ground truth spikes
        Din = F.conv3d(pred_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))
        Dtar = F.conv3d(target_bol, self.kernel, padding=(int(np.round((D - 1) / 2)), 0, 0))

        #plt.figure()  # to try to plot here
        #plt.plot(Din[0], )

        # ori's edit 14/12/2025 for fove overlapping
        # ds3d_utils.py, in forward(), just before kde_loss = ...
        # Din and Dtar are the KDE densities from prediction and target

        # Make sure Din and Dtar have the same spatial size
        if Din.shape != Dtar.shape:
            # Expect 5D tensors: (B, C, Z, Y, X)
            # Compute common spatial size
            min_z = min(Din.size(-3), Dtar.size(-3))
            min_y = min(Din.size(-2), Dtar.size(-2))
            min_x = min(Din.size(-1), Dtar.size(-1))

            Din = Din[..., :min_z, :min_y, :min_x]
            Dtar = Dtar[..., :min_z, :min_y, :min_x]

        kde_loss = nn.functional.mse_loss(Din, Dtar)

        # end ori's edit

        # kde loss
        #kde_loss = nn.functional.mse_loss(Din, Dtar)
        '''# added on 19/04/2026
        # L2 smoothness along z
        #dz = pred_bol[:, :, 1:, :, :] - pred_bol[:, :, :-1, :, :]
        #tv_z_l1 = torch.mean(torch.abs(dz) ** 1)
        #tv_z_l2 = torch.mean(dz ** 2)

        #final_loss = kde_loss + self.tv_z_weight * tv_z_l1
        '''
        #print(" | kde_loss = " + str(kde_loss) + "tv_z_loss = " + str(self.tv_z_weight * tv_z_l1))
        # end
        # final loss
        final_loss = kde_loss  # removed on 19/04/2026

        return final_loss


def calc_jaccard_rmse(xyz_gt, xyz_rec, radius):
    # if the net didn't detect anything return None's
    if xyz_rec is None:
        print("Empty Prediction!")
        return 0.0, None, None, None

    else:

        # calculate the distance matrix for each GT to each prediction
        C = pairwise_distances(xyz_rec, xyz_gt, 'euclidean')

        # number of recovered points and GT sources
        num_rec = xyz_rec.shape[0]
        num_gt = xyz_gt.shape[0]

        # find the matching using the Hungarian algorithm
        rec_ind, gt_ind = linear_sum_assignment(C)

        # number of matched points
        num_matches = len(rec_ind)

        # run over matched points and filter points radius away from GT
        indicatorTP = [False] * num_matches
        for i in range(num_matches):

            # if the point is closer than radius then TP else it's FP
            if C[rec_ind[i], gt_ind[i]] < radius:
                indicatorTP[i] = True

        # resulting TP count
        TP = sum(indicatorTP)

        # resulting jaccard index
        jaccard_index = TP / (num_rec + num_gt - TP)

        # if there's TP
        if TP:

            # pairs of TP
            rec_ind_TP = (rec_ind[indicatorTP]).tolist()
            gt_ind_TP = (gt_ind[indicatorTP]).tolist()
            xyz_rec_TP = xyz_rec[rec_ind_TP, :]
            xyz_gt_TP = xyz_gt[gt_ind_TP, :]

            # calculate mean RMSE in xy, z, and xyz
            RMSE_xy = np.sqrt(np.mean(np.sum((xyz_rec_TP[:, :2] - xyz_gt_TP[:, :2]) ** 2, 1)))
            RMSE_z = np.sqrt(np.mean(np.sum((xyz_rec_TP[:, 2:] - xyz_gt_TP[:, 2:]) ** 2, 1)))
            RMSE_xyz = np.sqrt(np.mean(np.sum((xyz_rec_TP - xyz_gt_TP) ** 2, 1)))

            return jaccard_index, RMSE_xy, RMSE_z, RMSE_xyz
        else:
            return jaccard_index, None, None, None