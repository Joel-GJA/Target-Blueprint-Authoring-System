from core.image_loader import load_image
from core.apriltag_detector import AprilTagDetector


image_data = load_image(
    "test_images/target.jpg"
)

detector = AprilTagDetector(
    families="tag36h11",
    nthreads=4,
)

result = detector.detect(image_data)


print("\n==============================")
print("APRILTAG DETECTION")
print("==============================")

print(f"Tags detected: {result.count}")
print(f"Tag IDs:       {result.tag_ids}")

for tag in result.detections:

    print()
    print(f"Tag ID:          {tag.tag_id}")
    print(f"Family:          {tag.tag_family}")
    print(f"Center:          {tag.center}")
    print(f"Decision margin: {tag.decision_margin:.2f}")
    print(f"Hamming:         {tag.hamming}")
    print("Corners:")
    print(tag.corners)

import cv2
import numpy as np
import matplotlib.pyplot as plt


display = image_data.image.copy()


for tag in result.detections:

    corners = np.round(
        tag.corners
    ).astype(np.int32)

    cv2.polylines(
        display,
        [corners],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    center = tuple(
        np.round(tag.center).astype(int)
    )

    cv2.circle(
        display,
        center,
        6,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        display,
        f"ID {tag.tag_id}",
        center,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )


display_rgb = cv2.cvtColor(
    display,
    cv2.COLOR_BGR2RGB,
)

plt.figure(figsize=(12, 10))
plt.imshow(display_rgb)
plt.title(
    f"AprilTag Detection — {result.count} tags"
)
plt.axis("off")
plt.show()