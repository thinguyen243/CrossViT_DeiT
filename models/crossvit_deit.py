import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp, Block

# -----------------------------------------------------
# MLP
# -----------------------------------------------------
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# -----------------------------------------------------
# Attention
# -----------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# -----------------------------------------------------
# DropPath
# -----------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# -----------------------------------------------------
# Transformer Block
# -----------------------------------------------------
# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
#                               attn_drop=attn_drop, proj_drop=drop)
#         self.drop_path = nn.Identity() if drop_path==0. else DropPath(drop_path)
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim*mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
#                        drop=drop, act_layer=act_layer)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# -----------------------------------------------------
# CrossViT9Deit
# -----------------------------------------------------
class CrossViT9Deit(nn.Module):
    def __init__(self,
                 img_size=(224,224),
                 patch_size=(16,8),
                 in_chans=3,
                 num_classes=38,
                 embed_dims=(192,384),
                 depth=(4,6),
                 num_heads=(3,6),
                 mlp_ratio=(4.,4.),
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        # convert int → list nếu cần
        if isinstance(embed_dims,int):
            embed_dims = [embed_dims]*2
        if isinstance(depth,int):
            depth = [depth]*2
        if isinstance(num_heads,int):
            num_heads = [num_heads]*2
        if isinstance(mlp_ratio,(float,int)):
            mlp_ratio = [mlp_ratio]*2

        self.num_branches = len(embed_dims)

        # patch embedding + pos_embed
        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ParameterList()
        for i in range(self.num_branches):
            self.patch_embed.append(
                nn.Conv2d(in_chans, embed_dims[i],
                          kernel_size=patch_size[i], stride=patch_size[i])
            )
            num_patches = (img_size[0] // patch_size[i]) * (img_size[1] // patch_size[i])
            self.pos_embed.append(nn.Parameter(torch.zeros(1, num_patches+1, embed_dims[i])))

        # class tokens
        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1,1,d)) for d in embed_dims])
        self.pos_drop = nn.Dropout(drop_rate)

        # stochastic depth
        total_depth = sum(depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_ptr = 0

        # transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(self.num_branches):
            dpr_branch = dpr[dpr_ptr:dpr_ptr+depth[i]]
            dpr_ptr += depth[i]
            self.blocks.append(nn.ModuleList([
                Block(embed_dims[i], num_heads[i], mlp_ratio[i],
                      qkv_bias, drop_path=dpr_branch[j])
                for j in range(depth[i])
            ]))

        # norm
        self.norm = nn.ModuleList([norm_layer(embed_dims[i]) for i in range(self.num_branches)])

        # classifier head
        self.head = nn.Linear(sum(embed_dims), num_classes)

        # init
        for p in self.pos_embed:
            trunc_normal_(p,std=.02)
        for c in self.cls_token:
            trunc_normal_(c,std=.02)

    def forward_features(self, x):
        B, C, H, W = x.shape
        outs = []

        for i in range(self.num_branches):
            tmp = self.patch_embed[i](x)
            tmp = tmp.flatten(2).transpose(1, 2)  # [B, N, C]

            cls_tokens = self.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)

            # align pos_embed if shape mismatch
            if tmp.shape[1] != self.pos_embed[i].shape[1]:
                cls_pos = self.pos_embed[i][:, :1, :]
                patch_pos = self.pos_embed[i][:, 1:, :]
                N = patch_pos.shape[1]
                C = patch_pos.shape[2]
                H_old = W_old = int(N**0.5)
                patch_pos = patch_pos.transpose(1, 2).reshape(1, C, H_old, W_old)
                H_new = W_new = int((tmp.shape[1]-1)**0.5)
                patch_pos = F.interpolate(patch_pos, size=(H_new, W_new),
                                          mode='bicubic', align_corners=False)
                patch_pos = patch_pos.flatten(2).transpose(1, 2)
                pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
            else:
                pos_embed = self.pos_embed[i]

            tmp = tmp + pos_embed
            tmp = self.pos_drop(tmp)

            for blk in self.blocks[i]:
                tmp = blk(tmp)

            tmp = self.norm[i](tmp)
            outs.append(tmp[:, 0])  # class token

        x = torch.cat(outs, dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# -----------------------------------------------------
# Factory
# -----------------------------------------------------
def my_cfg():
    return {
        'input_size': (3, 224, 224),
        'mean': (0.485,0.456,0.406),
        'std': (0.229,0.224,0.225),
        'interpolation': 'bilinear',
        'crop_pct': 1.0,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
    }


@register_model
def crossvit_9_deit(pretrained=False, num_classes=38, checkpoint_dir=None, **kwargs):
    # num_classes = 38
    model = CrossViT9Deit(num_classes=num_classes, **kwargs)
    # model.default_cfg = my_cfg()
    if pretrained:
        checkpoint = torch.load("output1/checkpoint.pth", map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        model.load_state_dict(state_dict, strict =False)
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print('Missing keys:', missing)
    # print('Unexpected keys:', unexpected)

    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.head.parameters():
    #     param.requires_grad = True

    return model

@register_model
def crossvit_9_deit_38(pretrained=False, checkpoint_dir=None, **kwargs):
    # Ép số lớp về 38
    kwargs['num_classes'] = 38
    model = CrossViT9Deit(**kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load("output1/checkpoint.pth", map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # BỎ head đi để tránh load 1000 lớp
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        model.load_state_dict(state_dict, strict=False)
        print(state_dict)

    return model


@register_model
def crossvit_9_deit_4(pretrained=False, checkpoint_dir=None, **kwargs):
    kwargs['num_classes'] = 4
    model = CrossViT9Deit(**kwargs)
    if pretrained:
        checkpoint = torch.load("output1/model_best.pth", map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    return model



