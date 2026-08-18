import cv2
import numpy as np
import matplotlib.pyplot as plt

from core.calibration.image_loader import load_image

image_data = load_image(
    "test_images/outdoor target.jpeg"
)

hsv = cv2.cvtColor(
    image_data.image,
    cv2.COLOR_BGR2HSV,
)

mask = cv2.inRange(
    hsv,
    np.array([0, 0, 140], dtype=np.uint8),
    np.array([180, 90, 255], dtype=np.uint8),
)

plt.figure(figsize=(10, 12))
plt.imshow(mask, cmap="gray")
plt.title("Paper Candidate Mask")
plt.axis("off")
plt.show()