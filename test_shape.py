# test_model_shape.py
import torch
from models.crossvit_deit import crossvit_9_deit_38

# mimic args from main
nb_classes = 38
model = crossvit_9_deit_38(pretrained=False, num_classes=nb_classes)

checkpoint = torch.load("output1/checkpoint.pth", map_location="cpu")
state_dict = checkpoint['model'] if "model" in checkpoint else checkpoint

# Bỏ head 1000 classes trong checkpoint
state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("[DEBUG] Missing keys:", missing)       # thường là ['head.weight', 'head.bias']
print("[DEBUG] Unexpected keys:", unexpected)
print("[DEBUG] model.head:", getattr(model, "head", None))

x = torch.randn(1, 3, 224, 224)
y = model(x)
print("forward output shape:", y.shape)   # phải là (1, 38)
