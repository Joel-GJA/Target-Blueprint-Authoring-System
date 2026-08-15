import cv2
import numpy as np

def extract_target_grabcut(image_path, roi):
    # 1. Load Original Image
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    h, w, _ = bgr_img.shape
    x, y, box_w, box_h = roi

    # 2. Setup GrabCut Mask and Models
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 3. Run GrabCut using your exact ROI
    # rect format: (x, y, width, height)
    rect = (x, y, box_w, box_h)
    cv2.grabCut(bgr_img, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # 4. Extract target mask (GrabCut values 1 & 3 are foreground)
    silhouette_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

    # Smooth the silhouette slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    silhouette_mask = cv2.morphologyEx(silhouette_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Get the outer boundary contour
    contours, _ = cv2.findContours(silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette_edge = np.zeros((h, w), np.uint8)
    if contours:
        target_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(silhouette_edge, [target_contour], -1, 255, thickness=2)

    return silhouette_mask, silhouette_edge

# --- EXECUTION ---
my_roi = (588, 85, 1677, 3811)  # (x, y, width, height)
target_mask, target_edge = extract_target_grabcut('dataset\\20260701_163736.jpg', my_roi)

tm = cv2.resize(target_mask, (167, 381))
te = cv2.resize(target_edge, (167, 381))
# Display in OpenCV popup windows
cv2.imshow("Target Mask", tm)
cv2.imshow("Target Edge", te)

cv2.waitKey(0)  # Press any key to close
cv2.destroyAllWindows()