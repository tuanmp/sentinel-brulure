"""
Prithvi burn scar inference - pure PyTorch implementation.
Loads Prithvi-EO-1.0-100M checkpoint without mmseg/mmcv dependency.
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar"
CHECKPOINT_NAME = "burn_scars_Prithvi_100M.pth"
PATCH_SIZE = 512
PATCH_BANDS = 6
OVERLAP = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_dir():
    return Path.home() / ".cache" / "sentinel-sat" / "models"


def download_checkpoint():
    from huggingface_hub import hf_hub_download
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CHECKPOINT_NAME,
        cache_dir=str(model_dir),
    )


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=512, patch_size=16, in_chans=6, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, num_patches, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_patches, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ value).transpose(1, 2).reshape(batch_size, num_patches, channels)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PrithviViT(nn.Module):
    """Prithvi ViT encoder matching the checkpoint structure."""
    def __init__(self, img_size=512, patch_size=16, in_chans=6, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        feat = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        feat = torch.cat((cls_tokens, feat), dim=1)
        feat = feat + self.pos_embed[:, :feat.shape[1], :]
        for blk in self.blocks:
            feat = blk(feat)
        feat = self.norm(feat)
        feat = feat[:, 1:, :]  # Remove cls token, (B, num_patches, embed_dim)
        return feat


class PrithviSegmentation(nn.Module):
    """Prithvi with segmentation decoder."""
    def __init__(self, img_size=512, patch_size=16, num_classes=2, **kwargs):
        super().__init__()
        self.encoder = PrithviViT(img_size, patch_size, **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size

        # Decoder: upsample from patch embeddings to full resolution
        # num_patches = (img_size/patch_size)^2, spatial dims = img_size/patch_size
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder.embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # Final upsample to img_size
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        feat = self.encoder(x)  # (B, num_patches, embed_dim)
        num_patches = self.encoder.num_patches
        feat_h = int(num_patches ** 0.5)
        feat = feat.transpose(1, 2).reshape(batch_size, -1, feat_h, feat_h)
        feat = self.decoder(feat)
        # Resize to match input spatial dims
        if feat.shape[2:] != (height, width):
            feat = torch.nn.functional.interpolate(feat, size=(height, width), mode='bilinear', align_corners=False)
        return feat


def load_model(checkpoint_path=None, device=DEVICE):
    """Load Prithvi burn scar model from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = download_checkpoint()

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    model = PrithviSegmentation(
        img_size=512, patch_size=16, in_chans=6, embed_dim=768,
        depth=12, num_heads=12, num_classes=2
    )

    filtered = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model


def prepare_input(geotiff_path, device=DEVICE):
    """Read 6-band GeoTIFF, normalize to [0,1], return (1,6,H,W) tensor."""
    with rasterio.open(geotiff_path) as src:
        data = src.read()  # (C, H, W)
    if data.shape[0] > 6:
        data = data[:6]
    if data.max() > 1:
        data = data.astype(np.float32) / 10000.0
    tensor = torch.from_numpy(data).unsqueeze(0).to(device)
    return tensor


def create_patches(tensor, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """Create overlapping patches from (1,6,H,W) tensor."""
    _, _, height, width = tensor.shape
    stride = int(patch_size * (1 - overlap))
    patches = []
    y_positions = list(range(0, max(1, height - patch_size + 1), stride))
    if not y_positions or y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    x_positions = list(range(0, max(1, width - patch_size + 1), stride))
    if not x_positions or x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)
    for y in y_positions:
        for x in x_positions:
            patch = tensor[:, :, y:y+patch_size, x:x+patch_size]
            patches.append((patch, (y, x)))
    return patches


def recombine_patches(patch_preds, output_shape, patch_size=PATCH_SIZE, overlap=OVERLAP):
    """Weighted recombine patches into full prediction mask."""
    height, width = output_shape
    full_pred = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    ramp = np.linspace(0, 1, int(patch_size * overlap)) ** 2
    weight_mask = np.ones(patch_size, dtype=np.float32)
    weight_mask[:len(ramp)] *= ramp[::-1]
    weight_mask[-len(ramp):] *= ramp
    weight_2d = np.outer(weight_mask, weight_mask)
    for pred, (y, x) in patch_preds:
        full_pred[y:y+patch_size, x:x+patch_size] += pred * weight_2d
        weight_sum[y:y+patch_size, x:x+patch_size] += weight_2d
    return full_pred / np.maximum(weight_sum, 1e-6)


@torch.no_grad()
def run_inference(geotiff_path, model=None, checkpoint_path=None, device=DEVICE):
    """
    Run burn scar inference with automatic patching.
    Returns: numpy array (H, W) with probabilities [0, 1] for burn scar class.
    """
    if model is None:
        model = load_model(checkpoint_path, device)
    tensor = prepare_input(geotiff_path, device)
    _, _, height, width = tensor.shape

    if height <= PATCH_SIZE and width <= PATCH_SIZE:
        if height < PATCH_SIZE or width < PATCH_SIZE:
            padded = torch.zeros((1, 6, PATCH_SIZE, PATCH_SIZE), device=device)
            padded[:, :, :height, :width] = tensor
            pred = model(padded)
            return torch.softmax(pred, dim=1)[0, 1, :height, :width].cpu().numpy()
        pred = model(tensor)
        return torch.softmax(pred, dim=1)[0, 1].cpu().numpy()

    patches = create_patches(tensor)
    patch_preds = []
    for patch, (y, x) in patches:
        _, _, p_h, p_w = patch.shape
        if p_h < PATCH_SIZE or p_w < PATCH_SIZE:
            padded = torch.zeros((1, 6, PATCH_SIZE, PATCH_SIZE), device=device)
            padded[:, :, :p_h, :p_w] = patch
            pred = model(padded)
            patch_preds.append((torch.softmax(pred, dim=1)[0, 1, :p_h, :p_w].cpu().numpy(), (y, x)))
        else:
            pred = model(patch)
            patch_preds.append((torch.softmax(pred, dim=1)[0, 1].cpu().numpy(), (y, x)))
    return recombine_patches(patch_preds, (height, width))


def save_mask(mask, output_path, reference_geotiff):
    """Save burn scar prediction as GeoTIFF."""
    with rasterio.open(reference_geotiff) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.float32)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask.astype(rasterio.float32), 1)
