import cv2
import numpy as np
import matplotlib.pyplot as plt

class ScaledROISelector:
    def __init__(self, image, window_name="Select Target (Aspect Ratio Preserved)"):
        self.orig_img = image
        self.orig_h, self.orig_w = image.shape[:2]
        self.aspect_ratio = self.orig_w / float(self.orig_h)
        self.window_name = window_name
        
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.roi_selected = False

    def mouse_callback(self, event, x, y, flags, param):
        # 1. Fetch current canvas size inside the window
        try:
            win_rect = cv2.getWindowImageRect(self.window_name)
            current_win_w = win_rect[2]
            current_win_h = win_rect[3]
        except Exception:
            current_win_w, current_win_h = self.orig_w, self.orig_h

        if current_win_w <= 0 or current_win_h <= 0:
            return

        # 2. Calculate the aspect-preserved scale factor & padding offsets
        window_aspect = current_win_w / float(current_win_h)

        if window_aspect > self.aspect_ratio:
            # Letterboxed horizontally (black bars on left/right)
            display_h = current_win_h
            display_w = int(display_h * self.aspect_ratio)
            offset_x = (current_win_w - display_w) // 2
            offset_y = 0
        else:
            # Letterboxed vertically (black bars on top/bottom)
            display_w = current_win_w
            display_h = int(display_w / self.aspect_ratio)
            offset_x = 0
            offset_y = (current_win_h - display_h) // 2

        # Uniform scale ratio (Original / Rendered Image Dimensions)
        scale = self.orig_w / float(display_w) if display_w > 0 else 1.0

        # 3. Adjust mouse coordinates by subtracting window padding offsets
        adj_x = x - offset_x
        adj_y = y - offset_y

        # Map directly to full-resolution original image coordinates
        orig_x = int(adj_x * scale)
        orig_y = int(adj_y * scale)

        # Clamp points strictly to valid image coordinates
        orig_x = max(0, min(orig_x, self.orig_w - 1))
        orig_y = max(0, min(orig_y, self.orig_h - 1))

        # 4. Handle mouse drag events
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (orig_x, orig_y)
            self.end_point = (orig_x, orig_y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (orig_x, orig_y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (orig_x, orig_y)
            self.roi_selected = True

    def select_roi(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Enforce aspect ratio locking property on window resize
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        
        # Set initial display window size maintaining original image aspect ratio
        initial_win_width = 800
        initial_win_height = int(initial_win_width / self.aspect_ratio)
        cv2.resizeWindow(self.window_name, initial_win_width, initial_win_height)
        
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n[INSTRUCTIONS]:")
        print(" 1. Resize or scale the window freely (Aspect ratio is preserved).")
        print(" 2. Click and drag a box around the target.")
        print(" 3. Press ENTER or SPACE to confirm, or 'r' to reset box.")

        while True:
            display_copy = self.orig_img.copy()

            if self.start_point and self.end_point:
                cv2.rectangle(display_copy, self.start_point, self.end_point, (0, 255, 0), 3)

            cv2.imshow(self.window_name, display_copy)
            key = cv2.waitKey(20) & 0xFF

            if key in [13, 32] and self.roi_selected:  # ENTER or SPACE
                break
            elif key == ord('r'):
                self.start_point = None
                self.end_point = None
                self.roi_selected = False
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None

        cv2.destroyWindow(self.window_name)

        # Output bounding box relative to original high-res image
        x1, x2 = sorted([self.start_point[0], self.end_point[0]])
        y1, y2 = sorted([self.start_point[1], self.end_point[1]])
        box_w = x2 - x1
        box_h = y2 - y1

        return (x1, y1, box_w, box_h)


def process_target_scaled(image_path):
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print("Error: Could not load image.")
        return

    h, w, _ = bgr_img.shape

    # 1. Select ROI with aspect-preserved scale tracking
    selector = ScaledROISelector(bgr_img)
    roi = selector.select_roi()

    if roi is None or roi[2] == 0 or roi[3] == 0:
        print("No valid ROI selected.")
        return

    x, y, box_w, box_h = roi
    print(f"\nCaptured Bounding Box on Original Image Resolution ({w}x{h}):")
    print(f" -> X: {x}, Y: {y}, Width: {box_w}, Height: {box_h}")

    # 2. GrabCut Silhouette Extraction (Inside User Box)
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (x, y, box_w, box_h)
    cv2.grabCut(bgr_img, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    silhouette_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    silhouette_mask = cv2.morphologyEx(silhouette_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(silhouette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette_edge = np.zeros((h, w), np.uint8)
    if contours:
        target_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(silhouette_edge, [target_contour], -1, 255, thickness=2)

    # 3. Solution 3: Ridge Filter (Masked)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    sobel_xx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
    sobel_yy = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)
    mag = np.uint8(np.clip(cv2.magnitude(sobel_xx, sobel_yy), 0, 255))
    _, binary_ridges = cv2.threshold(mag, 35, 255, cv2.THRESH_BINARY)

    # Erode mask slightly to leave only inner rings
    inner_mask = cv2.erode(silhouette_mask, kernel, iterations=3)
    inner_rings_clean = cv2.bitwise_and(binary_ridges, binary_ridges, mask=inner_mask)

    # 4. Display Results
    user_selection_display = bgr_img.copy()
    cv2.rectangle(user_selection_display, (x, y), (x + box_w, y + box_h), (0, 255, 0), 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv2.cvtColor(user_selection_display, cv2.COLOR_BGR2RGB))
    axes[0].set_title("1. Aspect-Locked ROI Box")
    axes[0].axis('off')

    axes[1].imshow(silhouette_edge, cmap='gray')
    axes[1].set_title("2. Clean Silhouette")
    axes[1].axis('off')

    axes[2].imshow(inner_rings_clean, cmap='gray')
    axes[2].set_title("3. Pure Solution 3 Inner Rings")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
process_target_scaled('outdoor target.jpeg')