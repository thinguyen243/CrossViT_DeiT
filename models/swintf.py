import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F

# =============== Patch Embedding ===============
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)   # [B, C, H/4, W/4]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x

# =============== MLP ===============
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

# =============== Window Attention (simplified) ===============
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, C//heads]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# =============== Swin Block (simplified, no shift for brevity) ===============
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# =============== Swin Transformer (tiny version) ===============
class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.layers = nn.ModuleList()
        dim = embed_dim
        for i, depth in enumerate(depths):
            for _ in range(depth):
                self.layers.append(SwinBlock(dim, num_heads[i]))
            if i < len(depths) - 1:  # Patch merging
                self.layers.append(nn.Linear(dim, dim * 2))
                dim *= 2
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(1)  # Global average pooling
        return self.head(x)

# =============== Build Model ===============

@register_model
def swintf(pretrained = False, num_classes=38):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]
    )
    if pretrained:
        checkpoint = torch.load(
            "comparision/dataset2/with_swin/swintf.pth",
            map_location="cpu"
        )
        # Nếu checkpoint là dict chứa "model" hoặc "state_dict"
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
    return model

