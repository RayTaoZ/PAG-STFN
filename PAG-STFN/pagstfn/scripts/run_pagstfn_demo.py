import argparse

import torch

from pagstfn import PAGSTFNModel
from pagstfn.models.simvp_modules import TAUSubBlock


def patch_tau_kernel_size():
    original_init = TAUSubBlock.__init__

    def patched_init(self, *args, **kwargs):
        if kwargs.get("kernel_size") == 21:
            kwargs["kernel_size"] = 11
        return original_init(self, *args, **kwargs)

    TAUSubBlock.__init__ = patched_init


def build_parser():
    parser = argparse.ArgumentParser(description="Run a PAG-STFN architecture forward-pass demo.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mslp-channels", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patch-tau-kernel", action="store_true", default=True)
    parser.add_argument("--no-patch-tau-kernel", action="store_false", dest="patch_tau_kernel")
    return parser


def main():
    args = build_parser().parse_args()

    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.patch_tau_kernel:
        patch_tau_kernel_size()

    model = PAGSTFNModel(
        in_shape_wind=(24, 2, 64, 80),
        in_shape_mslp=(24, args.mslp_channels, 120, 160),
        hid_S=64,
        hid_T=256,
        N_S=2,
        N_T=8,
        model_type="tau",
        mlp_ratio=8.0,
        drop=0.1,
        drop_path=0.1,
        mslp_encoder_layers=2,
    ).to(device)
    model.eval()

    x_wind = torch.randn(args.batch_size, 24, 2, 64, 80, device=device)
    x_mslp = torch.randn(args.batch_size, 24, args.mslp_channels, 120, 160, device=device)

    with torch.no_grad():
        y, f_gate, i_gate = model(x_wind, x_mslp, return_gates=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Model params: {param_count:,}")
    print(f"Input x_wind shape: {tuple(x_wind.shape)}")
    print(f"Input x_mslp shape: {tuple(x_mslp.shape)}")
    print(f"Output y shape: {tuple(y.shape)}")
    print(f"Forget gate shape: {tuple(f_gate.shape)}")
    print(f"Input gate shape: {tuple(i_gate.shape)}")


if __name__ == "__main__":
    main()
