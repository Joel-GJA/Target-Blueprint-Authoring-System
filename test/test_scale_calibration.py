from core.image_loader import load_image
from core.apriltag_detector import AprilTagDetector
from core.scale_calibrator import calibrate_scale


image_data = load_image(
    "test_images\\outdoor target.jpeg"
)

detector = AprilTagDetector(
    families="tag36h11",
    nthreads=4,
)

tag_result = detector.detect(
    image_data
)

scale_result = calibrate_scale(
    tag_result,
    tag_size_mm=50.0,   # <-- replace with actual tag size
)


print()
print("=" * 60)
print("PHYSICAL SCALE CALIBRATION")
print("=" * 60)

print(
    f"Pixels / mm : "
    f"{scale_result.pixels_per_mm:.6f}"
)

print(
    f"mm / pixel  : "
    f"{scale_result.millimeters_per_pixel:.6f}"
)

print(
    f"Tag size    : "
    f"{scale_result.reference_tag_size_mm:.2f} mm"
)

print(
    f"Tags used   : "
    f"{scale_result.contributing_tag_ids}"
)

print()
print("Individual measurements:")

for measurement in scale_result.measurements:

    print(
        f"  Tag {measurement.tag_id}: "
        f"{measurement.pixels_per_mm:.6f} px/mm "
        f"| quality={measurement.quality:.3f}"
    )