from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF


@dataclass
class ImageDisplayTransform:
    """
    Maps coordinates between original image space and
    the displayed image space.

    The image aspect ratio is always preserved.
    """

    image_width: int
    image_height: int

    display_width: int = 0
    display_height: int = 0

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def update(
        self,
        display_width: int,
        display_height: int,
    ) -> None:

        self.display_width = display_width
        self.display_height = display_height

        scale_x = (
            display_width / self.image_width
        )

        scale_y = (
            display_height / self.image_height
        )

        # Preserve aspect ratio.
        self.scale = min(
            scale_x,
            scale_y,
        )

        rendered_width = (
            self.image_width * self.scale
        )

        rendered_height = (
            self.image_height * self.scale
        )

        # Center image in display area.
        self.offset_x = (
            display_width - rendered_width
        ) / 2.0

        self.offset_y = (
            display_height - rendered_height
        ) / 2.0

    def image_to_display(
        self,
        x: float,
        y: float,
    ) -> QPointF:

        return QPointF(
            x * self.scale + self.offset_x,
            y * self.scale + self.offset_y,
        )

    def display_to_image(
        self,
        point: QPointF,
    ) -> tuple[int, int]:

        x = (
            point.x() - self.offset_x
        ) / self.scale

        y = (
            point.y() - self.offset_y
        ) / self.scale

        x = int(round(x))
        y = int(round(y))

        x = max(
            0,
            min(x, self.image_width - 1),
        )

        y = max(
            0,
            min(y, self.image_height - 1),
        )

        return x, y