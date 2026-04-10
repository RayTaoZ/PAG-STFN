# PAG-STFN

PAG-STFN is a vector wind speed prediction model with an encoder -> feature fusion -> temporal translator -> decoder architecture.

## Project Structure

- `pagstfn/models/`: includes `pagstfn_model.py` (the full PAG-STFN architecture) and the internal modules it depends on
- `pagstfn/scripts/run_pagstfn_demo.py`: runnable forward-pass demo

## Install

```bash
pip install -r requirements.txt
```

## Quick Run

```bash
python -m pagstfn.scripts.run_pagstfn_demo --batch-size 2 --mslp-channels 1
```

The demo prints model parameter count and output tensor shapes, including gate outputs when `return_gates=True`.

## Python Usage

```python
import torch
from pagstfn import PAGSTFNModel

model = PAGSTFNModel(
    in_shape_wind=(24, 2, 64, 80),
    in_shape_mslp=(24, 1, 120, 160),
    model_type="tau",
)

x_wind = torch.randn(2, 24, 2, 64, 80)
x_mslp = torch.randn(2, 24, 1, 120, 160)
y, f_gate, i_gate = model(x_wind, x_mslp, return_gates=True)
```

## Acknowledgement

This project includes adapted components derived from OpenSTL. Please keep original upstream licenses and attributions when redistributing.
