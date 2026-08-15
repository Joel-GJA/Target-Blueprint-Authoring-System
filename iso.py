import cv2
import numpy as np
import matplotlib.pyplot as plt

def isolate_target_features_from_ridge(image_path, roi):
    # 1. Load Original Image
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    h, w, _ = bgr_img.shape
    x, y, box_w, box_h = roi

    # 2. Extract Clean Silhouette Mask via GrabCut using ROI
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(bgr_img, mask, (x, y, box_w, box_h), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    silhouette_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

    # 3. Calculate Solution 3 (Hessian/Sobel Second Derivative Ridge Filter)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    sobel_xx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
    sobel_yy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
    mag = np.uint8(np.clip(cv2.magnitude(sobel_xx, sobel_yy), 0, 255))
    _, binary_ridges = cv2.threshold(mag, 35, 255, cv2.THRESH_BINARY)

    # 4. Mask the ridges strictly inside the silhouette (Erode slightly to strip outer paper edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    inner_mask = cv2.erode(silhouette_mask, kernel, iterations=3)
    
    clean_inner_rings = cv2.bitwise_and(binary_ridges, binary_ridges, mask=inner_mask)

    # ---------------------------------------------------------------------
    # 5. VISUALIZE IN A 2x2 GRID & SAVE PLOT
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Top-Left: GrabCut Silhouette
    axes[0, 0].imshow(silhouette_mask, cmap='gray')
    axes[0, 0].set_title("1. GrabCut Silhouette Mask", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Top-Right: Raw Hessian/Sobel Ridge Filter
    axes[0, 1].imshow(binary_ridges, cmap='gray')
    axes[0, 1].set_title("2. Raw Hessian Ridge Filter", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Bottom-Left: Eroded Inner Mask
    axes[1, 0].imshow(inner_mask, cmap='gray')
    axes[1, 0].set_title("3. Eroded Inner Target Mask", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Bottom-Right: Final Isolated Rings
    axes[1, 1].imshow(clean_inner_rings, cmap='gray')
    axes[1, 1].set_title("4. Isolated Target Inner Rings", fontsize=12, fontweight='bold', color='darkgreen')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save the 2x2 plot as an image file
    plt.savefig("Target_Analysis_2x2_Grid.png", dpi=300, bbox_inches='tight')
    print("Successfully saved 2x2 visualization grid as 'Target_Analysis_2x2_Grid.png'")

    # Display the 2x2 plot
    plt.show()

    return silhouette_mask, clean_inner_rings

# --- EXECUTION ---
user_roi = (88, 0, 663, 1584)
silhouette, inner_rings = isolate_target_features_from_ridge('outdoor target.jpeg', user_roi)
