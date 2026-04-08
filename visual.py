"""
visualize.py
============
A unified visualization toolkit for LesionAwareEnhancement ablation studies.
Includes:
  - Feature activation map visualization
  - Grad-CAM
  - Lesion–Background contrast map
  - Quantitative metrics (energy ratio, activation entropy)

Designed for CNN / Attention / Lesion-aware comparative analysis.
"""
from timm.models import create_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.mamba_vision import *
# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def normalize_map(x: np.ndarray):
    x = x - x.min()
    return x / (x.max() + 1e-6)


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


# -------------------------------------------------
# 1. Feature Activation Map
# -------------------------------------------------

def feature_activation_map(feat: torch.Tensor, index: int = 0):
    """
    Args:
        feat: Tensor [B, C, H, W]
    Returns:
        fmap: [H, W]
    """
    fmap = feat[index].mean(dim=0)
    fmap = tensor_to_numpy(fmap)
    return normalize_map(fmap)


# -------------------------------------------------
# 2. Grad-CAM
# -------------------------------------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.grad = None
        self.feat = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.feat = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.grad = grad_out[0]

    def generate(self, class_score: torch.Tensor):
        """
        Args:
            class_score: scalar tensor (e.g. logits[:, cls].sum())
        Returns:
            cam: [B, H, W]
        """
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feat).sum(dim=1)
        cam = F.relu(cam)

        cam = cam - cam.min(dim=-1, keepdim=True)[0]
        cam = cam / (cam.max(dim=-1, keepdim=True)[0] + 1e-6)
        return cam.detach()


# -------------------------------------------------
# 3. Lesion–Background Contrast Map
# -------------------------------------------------

def lesion_contrast_map(lesion_feat: torch.Tensor, bg_feat: torch.Tensor, index=0):
    """
    Args:
        lesion_feat, bg_feat: [B, C, H, W]
    Returns:
        contrast_map: [H, W]
    """
    contrast = lesion_feat[index] - bg_feat[index]
    contrast = contrast.mean(dim=0)
    contrast = F.relu(contrast)
    return normalize_map(tensor_to_numpy(contrast))


# -------------------------------------------------
# 4. Quantitative Metrics
# -------------------------------------------------

def activation_entropy(cam: torch.Tensor):
    """
    cam: [H, W]
    """
    cam = cam.flatten()
    cam = cam / (cam.sum() + 1e-6)
    entropy = -(cam * torch.log(cam + 1e-6)).sum()
    return entropy.item()


def energy_ratio(cam: torch.Tensor, topk: float = 0.2):
    """
    Ratio of energy within top-k activations
    """
    cam = cam.flatten()
    k = int(len(cam) * topk)
    topk_vals, _ = torch.topk(cam, k)
    return topk_vals.sum().item() / (cam.sum().item() + 1e-6)


# -------------------------------------------------
# 5. Visualization Helper
# -------------------------------------------------

def plot_maps(maps, titles, cmap='jet'):
    """
    maps: list of [H, W]
    """
    n = len(maps)
    plt.figure(figsize=(4 * n, 4))
    for i, (m, t) in enumerate(zip(maps, titles)):
        plt.subplot(1, n, i + 1)
        plt.imshow(m, cmap=cmap)
        plt.title(t)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
from PIL import Image
import torchvision.transforms as T

def load_image(img_path, img_size=224):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, H, W]

# -------------------------------------------------
# Example Usage (commented)
# -------------------------------------------------
"""
# Feature maps
fmap_base = feature_activation_map(feat_base)
fmap_sa   = feature_activation_map(feat_sa)
fmap_lafe = feature_activation_map(feat_lafe)

plot_maps([fmap_base, fmap_sa, fmap_lafe],
          ['Baseline', 'Spatial Attn', 'Lesion-aware'])

# Grad-CAM
cam = GradCAM(model, model.backbone.layer4)
cam_map = cam.generate(logits[:, target_class].sum())

# Contrast
contrast = lesion_contrast_map(lesion_feat, bg_feat)
"""
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_1 = create_model(
        'mamba_vision_T',
        # pretrained=args.pretrained,
        pretrained=False,
        num_classes=1000,
        global_pool='POOL',
        bn_momentum=None,
        bn_eps=None,
        scriptable=None,
        # checkpoint_path=args.initial_checkpoint,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0,
        conv_index=4,
        LAE_index=1,
        mamba_mode=0,
        transformer_blocks_mode=0,
        visual_mode=1,
    )
    initial_checkpoint = "/home/MambaVision/output/train/my_experiment/LAE/model_best.pth.tar"
    try:
        checkpoint = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # 获取当前模型的 state_dict
        model_dict = model_1.state_dict()

        # 过滤掉那些不匹配的参数（例如 classifier / head）
        filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

        # 更新模型权重（只加载匹配的部分!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!）
        model_dict.update(filtered_dict)
        model_1.load_state_dict(model_dict)

        # 冻结已经加载的层的参数
        # for name, param in model.named_parameters():
        #     if name in filtered_dict:
        #         param.requires_grad = False
    except FileNotFoundError:
        pass
    model_1 = model_1.to(device)
    # print(model)
    model_2 = create_model(
        'mamba_vision_T',
        # pretrained=args.pretrained,
        pretrained=False,
        num_classes=1000,
        global_pool='POOL',
        bn_momentum=None,
        bn_eps=None,
        scriptable=None,
        # checkpoint_path=args.initial_checkpoint,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0,
        conv_index=5,
        LAE_index=2,
        mamba_mode=1,
        transformer_blocks_mode=1,
        visual_mode=1,
    )
    # print(model_2)
    initial_checkpoint = "/home/MambaVision/output/train/my_experiment/S_A/model_best.pth.tar"
    try:
        checkpoint = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # 获取当前模型的 state_dict
        model_dict = model_2.state_dict()

        # 过滤掉那些不匹配的参数（例如 classifier / head）
        filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

        # 更新模型权重（只加载匹配的部分!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!）
        model_dict.update(filtered_dict)
        model_2.load_state_dict(model_dict)

        # 冻结已经加载的层的参数
        # for name, param in model.named_parameters():
        #     if name in filtered_dict:
        #         param.requires_grad = False
    except FileNotFoundError:
        pass
    model_2 = model_2.to(device)

    model_3 = create_model(
        'mamba_vision_T',
        # pretrained=args.pretrained,
        pretrained=False,
        num_classes=1000,
        global_pool='POOL',
        bn_momentum=None,
        bn_eps=None,
        scriptable=None,
        # checkpoint_path=args.initial_checkpoint,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0,
        conv_index=5,
        LAE_index=0,
        mamba_mode=1,
        transformer_blocks_mode=1,
        visual_mode=1,
    )
    initial_checkpoint = "/home/MambaVision/output/train/my_experiment/N_A/model_best.pth.tar"
    try:
        checkpoint = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # 获取当前模型的 state_dict
        model_dict = model_3.state_dict()

        # 过滤掉那些不匹配的参数（例如 classifier / head）
        filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}

        # 更新模型权重（只加载匹配的部分!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!）
        model_dict.update(filtered_dict)
        model_3.load_state_dict(model_dict)

        # 冻结已经加载的层的参数
        # for name, param in model.named_parameters():
        #     if name in filtered_dict:
        #         param.requires_grad = False
    except FileNotFoundError:
        pass
    model_3 = model_3.to(device)

    #input pic

    x = load_image("/home/soybean_dataset_spilt/train/bacterial_bligh/bacterial_blight_2.jpg").cuda()
    with torch.no_grad():
        x_lae, x_l_lae, l_lae, x_1_lae, x_pool_lae, x_norm_lae = model_1(x)
        x_sa, x_l_sa, l_sa, x_1_sa, x_pool_sa, x_norm_sa = model_2(x)
        x_na, x_l_na, l_na, x_1_na, x_pool_na, x_norm_na = model_3(x)

    # print(x.shape)
    # print(x_l.shape)
    # print(l.shape)
    # print(x_1.shape)
    # print(x_norm.shape)
    # print(x_pool.shape)
    # act_map = feature_activation_map(x_1)

    fmap_base = feature_activation_map(x_na)
    fmap_sa = feature_activation_map(x_sa)
    fmap_lafe = feature_activation_map(x_lae)

    plot_maps([fmap_base, fmap_sa, fmap_lafe],
              ['Baseline', 'Spatial Attn', 'Lesion-aware'])