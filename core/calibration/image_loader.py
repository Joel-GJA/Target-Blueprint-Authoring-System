from pathlib import Path

import cv2

from core.models import ImageData
from core.models import ImageMetadata


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or validated."""


def load_image(file_path: str | Path) -> ImageData:
    """
    Load and validate an image from disk.

    Parameters
    ----------
    file_path:
        Path to the image file.

    Returns
    -------
    ImageData
        Validated image and its associated metadata.

    Raises
    ------
    ImageLoadError
        If the file does not exist, has an unsupported extension,
        cannot be decoded, or produces an invalid image.
    """

    path = Path(file_path).expanduser().resolve()

    # ---------------------------------------------------------
    # 1. Validate path
    # ---------------------------------------------------------

    if not path.exists():
        raise ImageLoadError(
            f"Image file does not exist: {path}"
        )

    if not path.is_file():
        raise ImageLoadError(
            f"Path is not a file: {path}"
        )

    # ---------------------------------------------------------
    # 2. Validate extension
    # ---------------------------------------------------------

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image format '{extension}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # ---------------------------------------------------------
    # 3. Decode image
    # ---------------------------------------------------------

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ImageLoadError(
            f"OpenCV could not decode image: {path}"
        )

    # ---------------------------------------------------------
    # 4. Validate image dimensions
    # ---------------------------------------------------------

    if image.size == 0:
        raise ImageLoadError(
            f"Decoded image contains no pixel data: {path}"
        )

    if image.ndim != 3:
        raise ImageLoadError(
            f"Expected a 3-channel image, got shape {image.shape}"
        )

    height, width, channels = image.shape

    if width <= 0 or height <= 0:
        raise ImageLoadError(
            f"Invalid image dimensions: {width}x{height}"
        )

    if channels != 3:
        raise ImageLoadError(
            f"Expected 3 channels, got {channels}"
        )

    # ---------------------------------------------------------
    # 5. Construct metadata
    # ---------------------------------------------------------

    metadata = ImageMetadata(
        path=path,
        filename=path.name,
        width=width,
        height=height,
        channels=channels,
        dtype=str(image.dtype),
        file_size_bytes=path.stat().st_size,
    )

    # ---------------------------------------------------------
    # 6. Construct ImageData
    # ---------------------------------------------------------

    return ImageData(
        image=image,
        metadata=metadata,
    )
