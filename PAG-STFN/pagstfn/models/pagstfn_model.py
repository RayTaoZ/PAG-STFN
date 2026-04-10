import torch
import torch.nn.functional as F
from torch import nn

from pagstfn.models.simvp_model import SimVP_Model


class PAGSTFNSpatialAttentionGate(nn.Module):
    """Spatial attention gate used to modulate latent forcing features."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))


class PAGSTFNContextEncoder(nn.Module):
    """MSLP context encoder with configurable depth."""

    def __init__(self, in_channels, out_channels=16, num_layers=2, hidden_channels=32):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers = []
        current_in = in_channels
        for i in range(num_layers):
            is_last_layer = i == (num_layers - 1)
            current_out = out_channels if is_last_layer else hidden_channels
            layers.append(nn.Conv2d(current_in, current_out, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(current_out))
            layers.append(nn.Tanh() if is_last_layer else nn.LeakyReLU(0.1, inplace=True))
            current_in = current_out

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class PAGSTFNModel(SimVP_Model):
    """PAG-STFN architecture: wind encoder + MSLP fusion + temporal translator + decoder."""

    def __init__(
        self,
        in_shape_wind,
        in_shape_mslp,
        hid_S=64,
        hid_T=256,
        N_S=2,
        N_T=8,
        model_type="gSTA",
        mslp_encoder_layers=2,
        **kwargs,
    ):
        super().__init__(in_shape_wind, hid_S, hid_T, N_S, N_T, model_type, **kwargs)

        self.mslp_feat_channels = 16
        self.mslp_encoder = PAGSTFNContextEncoder(
            in_channels=in_shape_mslp[1],
            out_channels=self.mslp_feat_channels,
            num_layers=mslp_encoder_layers,
            hidden_channels=32,
        )

        self.mslp_align = nn.Sequential(
            nn.Conv2d(self.mslp_feat_channels, hid_S, kernel_size=1),
            nn.BatchNorm2d(hid_S),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.forget_gate = PAGSTFNSpatialAttentionGate(kernel_size=7)
        self.input_gate = PAGSTFNSpatialAttentionGate(kernel_size=7)
        self.gate_tanh = nn.Tanh()

    def forward(self, x_wind, x_mslp, return_gates=False, return_forcing_mag=False, **kwargs):
        B, T, C, H, W = x_wind.shape

        x_wind = x_wind.view(B * T, C, H, W)
        embed_wind, skip_wind = self.enc(x_wind)

        B_m, T_m, C_m, H_m, W_m = x_mslp.shape
        x_mslp = x_mslp.view(B_m * T_m, C_m, H_m, W_m)
        embed_mslp = self.mslp_encoder(x_mslp)

        if embed_mslp.shape[-2:] != embed_wind.shape[-2:]:
            embed_mslp = F.interpolate(
                embed_mslp,
                size=embed_wind.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        mslp_aligned = self.mslp_align(embed_mslp)

        f_gate = self.forget_gate(mslp_aligned)
        i_gate = self.input_gate(mslp_aligned)

        gate_tanh_out = self.gate_tanh(mslp_aligned)
        z = embed_wind * f_gate + i_gate * gate_tanh_out

        H_latent, W_latent = z.shape[-2], z.shape[-1]
        z = z.view(B, T, -1, H_latent, W_latent)
        z = self.hid(z)
        z = z.view(B * T, -1, H_latent, W_latent)

        y = self.dec(z, skip_wind)
        y = y.view(B, T, C, H, W)

        if return_gates:
            f_gate_out = f_gate.view(B, T, 1, H_latent, W_latent)
            i_gate_out = i_gate.view(B, T, 1, H_latent, W_latent)
            if return_forcing_mag:
                forcing_term = i_gate * gate_tanh_out
                forcing_mag = torch.norm(forcing_term, p=2, dim=1, keepdim=True)
                forcing_mag_out = forcing_mag.view(B, T, 1, H_latent, W_latent)
                return y, f_gate_out, i_gate_out, forcing_mag_out
            return y, f_gate_out, i_gate_out

        return y


__all__ = [
    "PAGSTFNModel",
    "PAGSTFNContextEncoder",
    "PAGSTFNSpatialAttentionGate",
]
