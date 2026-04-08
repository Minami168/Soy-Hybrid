import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class TransformerGradCAM:
    def __init__(self, model, target_layer):
        """
        model: ViT 或 Mamba 模型
        target_layer: 一般是 encoder block 的输出层 (例如 model.blocks[-1].norm1)
        """
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()   # [B, N, C]，N=token数

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None, patch_shape=(14, 14)):
        """
        input_tensor: [1, C, H, W] 输入图像
        class_idx: 指定类别
        patch_shape: (H_p, W_p)，ViT 的 patch 网格大小
        """
        output = self.model(input_tensor)  # [1, num_classes]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # 权重 = GAP(梯度) -> [N, C] → [N]
        weights = self.gradients.mean(dim=-1)  # [B, N]
        cam = (weights.unsqueeze(-1) * self.activations).sum(-1)  # [B, N]

        cam = F.relu(cam)
        cam = cam[0, 1:]  # 去掉CLS token，只保留patch tokens

        # reshape 成 patch map
        cam = cam.reshape(patch_shape)
        cam = cam.unsqueeze(0).unsqueeze(0)  # [1,1,H_p,W_p]

        # resize 回输入图像大小
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize(self, input_image, mask, alpha=0.5, cmap="jet"):
        plt.imshow(input_image)
        plt.imshow(mask, cmap=cmap, alpha=alpha)
        plt.axis("off")
        plt.show()
