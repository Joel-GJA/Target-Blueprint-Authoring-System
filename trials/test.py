import cv2

def nothing(x):
    pass

# Load original image in grayscale
img = cv2.imread('outdoor target.jpeg', cv2.IMREAD_GRAYSCALE)

# Create a window for trackbars
cv2.namedWindow('CLAHE + Canny Fine-Tuning', cv2.WINDOW_NORMAL)
cv2.resizeWindow('CLAHE + Canny Fine-Tuning', 1000, 700)

# Create sliders for fine-tuning
# Clip limit is multiplied by 10 in trackbar because trackbars only use integers
cv2.createTrackbar('Clip Limit x10', 'CLAHE + Canny Fine-Tuning', 20, 100, nothing)  # Default 2.0
cv2.createTrackbar('Grid Size', 'CLAHE + Canny Fine-Tuning', 8, 32, nothing)         # Default 8x8
cv2.createTrackbar('Blur Kernel', 'CLAHE + Canny Fine-Tuning', 1, 5, nothing)         # 1=3x3, 2=5x5, etc.
cv2.createTrackbar('Canny Low', 'CLAHE + Canny Fine-Tuning', 50, 255, nothing)
cv2.createTrackbar('Canny High', 'CLAHE + Canny Fine-Tuning', 150, 255, nothing)

while True:
    # 1. Read slider values
    clip_val = max(1, cv2.getTrackbarPos('Clip Limit x10', 'CLAHE + Canny Fine-Tuning')) / 10.0
    grid_val = max(2, cv2.getTrackbarPos('Grid Size', 'CLAHE + Canny Fine-Tuning'))
    blur_val = cv2.getTrackbarPos('Blur Kernel', 'CLAHE + Canny Fine-Tuning')
    canny_low = cv2.getTrackbarPos('Canny Low', 'CLAHE + Canny Fine-Tuning')
    canny_high = cv2.getTrackbarPos('Canny High', 'CLAHE + Canny Fine-Tuning')

    # 2. Apply CLAHE with current slider values
    clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(grid_val, grid_val))
    enhanced = clahe.apply(img)

    # 3. Apply Gaussian Blur to suppress CLAHE noise amplification
    if blur_val > 0:
        k_size = blur_val * 2 + 1  # Ensures odd kernel size (3, 5, 7, etc.)
        processed = cv2.GaussianBlur(enhanced, (k_size, k_size), 0)
    else:
        processed = enhanced

    # 4. Apply Canny Edge Detection
    edges = cv2.Canny(processed, canny_low, canny_high)

    # Combine enhanced grayscale and edges side-by-side for comparison
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([enhanced_bgr, edges_bgr])

    # Show live result
    cv2.imshow('CLAHE + Canny Fine-Tuning', combined)

    # Press 'q' or 'ESC' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()