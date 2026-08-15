import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_solution_1_tophat(gray_img):
    """Solution 1: Morphological Top-Hat + Black-Hat Dual Filter"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    white_hat = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    
    combined_hat = cv2.add(white_hat, black_hat)
    blurred = cv2.GaussianBlur(combined_hat, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=90)
    return edges

def run_solution_2_hsv(bgr_img):
    """Solution 2: HSV Color Space (Saturation + Value Channels)"""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    edges_v = cv2.Canny(v, threshold1=40, threshold2=120)
    edges_s = cv2.Canny(s, threshold1=40, threshold2=120)
    
    return cv2.bitwise_or(edges_v, edges_s)

def run_solution_3_ridge(gray_img):
    """Solution 3: Ridge Detection Filter (Hessian Matrix)"""
    try:
        ridge_detector = cv2.ximgproc.createRidgeDetectionFilter(
            ddepth=cv2.CV_32F, dx=1, dy=1, ksize=3, out_dtype=cv2.CV_8U, scale=1, delta=0
        )
        ridges = ridge_detector.getRidgeFilteredImage(gray_img)
        _, binary_ridges = cv2.threshold(ridges, 30, 255, cv2.THRESH_BINARY)
        return binary_ridges
    except AttributeError:
        # Fallback to second-order Sobel Hessian magnitude if ximgproc is missing
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 2, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 2, ksize=3)
        mag = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
        _, binary_ridges = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)
        return binary_ridges

def process_batch(input_folder, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Grab only .jpg (and .jpeg) files while ignoring .xml and other extensions
    jpg_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                glob.glob(os.path.join(input_folder, "*.jpeg")) + \
                glob.glob(os.path.join(input_folder, "*.JPG"))
    
    if not jpg_files:
        print(f"No JPG images found in '{input_folder}'. Check the path!")
        return

    print(f"Found {len(jpg_files)} JPG files. Processing batch...")

    for index, img_path in enumerate(jpg_files, start=1):
        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        print(f"[{index}/{len(jpg_files)}] Processing: {filename}")

        # Read image
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            print(f"  --> Warning: Could not read {filename}. Skipping.")
            continue
            
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # 1. Run all 3 solutions
        sol1_edges = run_solution_1_tophat(gray_img)
        sol2_edges = run_solution_2_hsv(bgr_img)
        sol3_edges = run_solution_3_ridge(gray_img)

        # 2. Build 2x2 comparison grid plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axes[0, 0].imshow(gray_img, cmap='gray')
        axes[0, 0].set_title("Original Grayscale Input", fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(sol1_edges, cmap='gray')
        axes[0, 1].set_title("Sol 1: Morphological Top-Hat", fontsize=12)
        axes[0, 1].axis('off')

        axes[1, 0].imshow(sol2_edges, cmap='gray')
        axes[1, 0].set_title("Sol 2: HSV (S + V Channels)", fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(sol3_edges, cmap='gray')
        axes[1, 1].set_title("Sol 3: Ridge / Hessian Filter", fontsize=12, color='darkgreen', fontweight='bold')
        axes[1, 1].axis('off')

        plt.tight_layout()

        # 3. Save plot to output directory
        save_path = os.path.join(output_folder, f"{filename_no_ext}_comparison.jpg")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close memory buffer

    print(f"\nProcessing complete! All comparison grids saved to: '{output_folder}'")

# --- EXECUTION ---
input_dir = "test"
output_dir = "output"

process_batch(input_dir, output_dir)