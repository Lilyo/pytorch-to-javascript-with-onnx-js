import torch
from torch import nn
import numpy as np

torch.manual_seed(0)

layer_count = 4

model = nn.LSTM(10, 20, num_layers=1, bidirectional=True)
model.eval()

with torch.no_grad():
    input = torch.randn(1, 3, 10)
    h0 = torch.randn(layer_count * 2, 3, 20)
    c0 = torch.randn(layer_count * 2, 3, 20)
    output, (hn, cn) = model(input, (h0, c0))

    # default export
    torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')
