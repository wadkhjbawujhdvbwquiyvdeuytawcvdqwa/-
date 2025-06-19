import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# 定义 PatchEmbed 类
class PatchEmbed(nn.Module):
    """图像划分为 Patch 并嵌入"""

    def __init__(self, in_channels=1, patch_size=16, emb_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B x C x H x W -> B x Patch数 x emb_dim
        x = self.proj(x)  # B x E x H/p x W/p
        x = x.flatten(2).transpose(1, 2)  # B x N x E
        return x


# 定义 CrossAttentionFusion 类
class CrossAttentionFusion(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x_ir, x_vis):
        # Cross Attention
        attn_out, _ = self.attn(query=x_ir, key=x_vis, value=x_vis)
        x = self.norm1(x_ir + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


# 数据集类
class FusionDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, transform=None):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.transform = transform
        self.images = [f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        ir_path = os.path.join(self.ir_dir, img_name)
        vis_path = os.path.join(self.vis_dir, img_name)

        ir_image = Image.open(ir_path).convert("L")
        vis_image = Image.open(vis_path).convert("L")

        if self.transform:
            ir_image = self.transform(ir_image)
            vis_image = self.transform(vis_image)

        return ir_image, vis_image


# 改进的Transformer融合模型
class ImprovedFusionTransformer(nn.Module):
    def __init__(self, patch_size=16, emb_dim=256, img_size=256):
        super().__init__()
        self.embed_ir = PatchEmbed(1, patch_size, emb_dim)
        self.embed_vis = PatchEmbed(1, patch_size, emb_dim)

        # 双向交叉注意力
        self.cross_attn_ir = CrossAttentionFusion(emb_dim)
        self.cross_attn_vis = CrossAttentionFusion(emb_dim)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, patch_size * patch_size),
        )

        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, ir, vis):
        # 特征提取
        x_ir = self.embed_ir(ir)
        x_vis = self.embed_vis(vis)

        # 双向交叉注意力
        x_ir_fused = self.cross_attn_ir(x_ir, x_vis)
        x_vis_fused = self.cross_attn_vis(x_vis, x_ir)

        # 特征融合
        x_fused = torch.cat([x_ir_fused, x_vis_fused], dim=2)
        x_fused = self.fusion(x_fused)

        # 重建图像
        patches = self.decoder(x_fused)
        B, N, P2 = patches.shape
        H = W = int(self.img_size / self.patch_size)
        patches = patches.view(B, 1, H, W, self.patch_size, self.patch_size)
        fused = patches.permute(0, 1, 2, 4, 3, 5).reshape(B, 1, self.img_size, self.img_size)

        return fused


# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for ir_imgs, vis_imgs in dataloader:
            ir_imgs = ir_imgs.to(device)
            vis_imgs = vis_imgs.to(device)

            # 前向传播
            fused_imgs = model(ir_imgs, vis_imgs)

            # 计算损失 (这里使用简单的MSE损失，实际应用中可能需要更复杂的损失函数)
            loss = criterion(fused_imgs, (ir_imgs + vis_imgs) / 2)  # 简单示例，实际应使用更合适的目标

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * ir_imgs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model


# 可视化函数
def visualize_results(ir_img, vis_img, fused_img):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(ir_img.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Infrared')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(vis_img.squeeze().cpu().numpy(), cmap='gray')
    plt.title('Visible')
    plt.axis('off')

    plt.subplot(133)
    # 确保融合图像在[0,1]范围内
    fused_np = fused_img.squeeze().cpu().numpy()
    fused_np = (fused_np - fused_np.min()) / (fused_np.max() - fused_np.min() + 1e-8)
    plt.imshow(fused_np, cmap='gray')
    plt.title('Fused')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # 设置参数
    img_size = 256
    patch_size = 16
    batch_size = 4
    num_epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 创建数据集和数据加载器
    # 注意：这里需要替换为实际的数据集路径
    ir_dir = r'C:\Users\亿森\OneDrive\桌面\pys\shuju\ir'
    vis_dir = r'C:\Users\亿森\OneDrive\桌面\pys\shuju\vi'
    dataset = FusionDataset(ir_dir, vis_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = ImprovedFusionTransformer(patch_size=patch_size, img_size=img_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    print("开始训练...")
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs, device)

    # 保存模型
    torch.save(trained_model.state_dict(), 'fusion_transformer.pth')
    print("模型已保存")

    # 测试模型
    print("测试模型...")
    trained_model.eval()

    # 加载测试图像
    ir_test = Image.open('ir.png').convert("L")
    vis_test = Image.open('vis.png').convert("L")

    ir_test = transform(ir_test).unsqueeze(0).to(device)
    vis_test = transform(vis_test).unsqueeze(0).to(device)

    # 生成融合图像
    with torch.no_grad():
        fused_test = trained_model(ir_test, vis_test)

    # 可视化结果
    visualize_results(ir_test, vis_test, fused_test)


if __name__ == "__main__":
    main()
