from core.calibration.image_loader import load_image, ImageLoadError


try:
    data = load_image("test_images/outdoor target.jpeg")

    print(data.image.shape)

    print(data.metadata.filename)
    print(data.metadata.width)
    print(data.metadata.height)
    print(data.metadata.channels)
    print(data.metadata.dtype)
    print(data.metadata.file_size_bytes)

except ImageLoadError as exc:
    print(f"Failed to load image: {exc}")