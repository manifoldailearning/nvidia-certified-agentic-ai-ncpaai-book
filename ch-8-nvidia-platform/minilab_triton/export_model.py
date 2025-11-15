import torch
import torch.nn as nn

# 1. Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)

# 2. Instantiate model and dummy input
model = SimpleModel()
model.eval()

dummy_input = torch.randn(1, 4)

# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["INPUT__0"],
    output_names=["OUTPUT__0"],
    dynamic_axes={
        "INPUT__0": {0: "batch"},
        "OUTPUT__0": {0: "batch"}
    }
)

print("Exported model.onnx successfully.")
