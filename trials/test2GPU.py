import os
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Detect GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running pipeline on device: {device}")

# =====================================================================
# GPU ALGORITHM IMPLEMENTATIONS (PyTorch Tensors)
# =====================================================================

def gpu_morphology_tophat(gray_tensor, ksize=5):
    """
    GPU Top-Hat + Black-Hat transform using MaxPool2d (Dilation/Erosion).
    Input shape: (1, 1, H, W) float tensor [0, 1] on GPU.
    """
    padding = ksize // 2
    # Dilation via MaxPool2d
    dilation = F.max_pool2d(gray_tensor, kernel_size=ksize, stride=1, padding=padding)
    # Erosion via -MaxPool2d(-x)
    erosion = -F.max_pool2d(-gray_tensor, kernel_size=ksize, stride=1, padding=padding)
    
    white_hat = torch.clamp(gray_tensor - erosion, 0.0, 1.0)
    black_hat = torch.clamp(dilation - gray_tensor, 0.0, 1.0)
    
    combined = torch.clamp(white_hat + black_hat, 0.0, 1.0)
    
    # Gaussian Blur on GPU
    blurred = TF.gaussian_blur(combined, kernel_size=[3, 3], sigma=[1.0, 1.0])
    
    # Simple Sobel-based Canny surrogate on GPU
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    
    gx = F.conv2d(blurred, sobel_x, padding=1)
    gy = F.conv2d(blurred, sobel_y, padding=1)
    mag = torch.sqrt(gx**2 + gy**2)
    
    # Thresholding on GPU
    edges = (mag > 0.15).float()
    return edges.squeeze().cpu().numpy()


def gpu_hsv_edges(bgr_tensor):
    """
    GPU HSV channel split & thresholding.
    Input shape: (1, 3, H, W) float tensor [0, 1] on GPU.
    """
    # RGB Conversion
    rgb = bgr_tensor[:, [2, 1, 0], :, :]
    
    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
    max_c, _ = torch.max(rgb, dim=1, keepdim=True)
    min_c, _ = torch.min(rgb, dim=1, keepdim=True)
    delta = max_c - min_c + 1e-7
    
    # Saturation (S) and Value (V) on GPU
    s = delta / (max_c + 1e-7)
    v = max_c
    
    # Sobel Gradients on GPU
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    
    mag_s = torch.sqrt(F.conv2d(s, sobel_x, padding=1)**2 + F.conv2d(s, sobel_y, padding=1)**2)
    mag_v = torch.sqrt(F.conv2d(v, sobel_x, padding=1)**2 + F.conv2d(v, sobel_y, padding=1)**2)
    
    edges_s = (mag_s > 0.2).float()
    edges_v = (mag_v > 0.2).float()
    
    combined = torch.clamp(edges_s + edges_v, 0.0, 1.0)
    return combined.squeeze().cpu().numpy()


def gpu_ridge_hessian(gray_tensor):
    """
    GPU Second-Order Derivative (Hessian Matrix) Ridge Filter.
    Input shape: (1, 1, H, W) float tensor [0, 1] on GPU.
    """
    # 2nd derivative kernels
    sobel_xx = torch.tensor([[1, -2, 1], [2, -4, 2], [1, -2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_yy = torch.tensor([[1, 2, 1], [-2, -4, -2], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    
    dxx = F.conv2d(gray_tensor, sobel_xx, padding=1)
    dyy = F.conv2d(gray_tensor, sobel_yy, padding=1)
    
    hessian_mag = torch.sqrt(dxx**2 + dyy**2)
    ridges = (hessian_mag > 0.25).float()
    
    return ridges.squeeze().cpu().numpy()

# =====================================================================
# BATCH PROCESSOR
# =====================================================================

def process_batch_gpu(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Filter only JPG files (ignores XMLs)
    jpg_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                glob.glob(os.path.join(input_folder, "*.jpeg")) + \
                glob.glob(os.path.join(input_folder, "*.JPG"))
    
    if not jpg_files:
        print(f"No JPG images found in '{input_folder}'.")
        return

    print(f"Found {len(jpg_files)} JPG files. Processing on GPU ({device})...\n")

    for index, img_path in enumerate(jpg_files, start=1):
        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        print(f"[{index}/{len(jpg_files)}] GPU Processing: {filename}")

        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            continue
            
        # Convert NumPy image to GPU PyTorch Tensor [0, 1]
        bgr_tensor = torch.from_numpy(bgr_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        gray_tensor = 0.299 * bgr_tensor[:, 2:3, :, :] + 0.587 * bgr_tensor[:, 1:2, :, :] + 0.114 * bgr_tensor[:, 0:1, :, :]

        # 1. Execute GPU Parallel Pipeline
        with torch.no_grad():
            sol1_edges = gpu_morphology_tophat(gray_tensor)
            sol2_edges = gpu_hsv_edges(bgr_tensor)
            sol3_edges = gpu_ridge_hessian(gray_tensor)

        # 2. Build 2x2 comparison grid plot
        gray_cpu = gray_tensor.squeeze().cpu().numpy()
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axes[0, 0].imshow(gray_cpu, cmap='gray')
        axes[0, 0].set_title("Original Grayscale Input", fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(sol1_edges, cmap='gray')
        axes[0, 1].set_title("Sol 1: GPU Morphological Top-Hat", fontsize=12)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(sol2_edges, cmap='gray')
        axes[1, 0].set_title("Sol 2: GPU HSV (S + V Channels)", fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(sol3_edges, cmap='gray')
        axes[1, 1].set_title("Sol 3: GPU Ridge / Hessian Filter", fontsize=12, color='darkgreen', fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()

        # 3. Save plot to output directory
        save_path = os.path.join(output_folder, f"{filename_no_ext}_comparison.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\nGPU Processing Complete! Results saved in: '{output_folder}'")

# --- EXECUTION ---
input_dir = "test"
output_dir = "output"

process_batch_gpu(input_dir, output_dir)