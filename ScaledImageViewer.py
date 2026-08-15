import cv2
import numpy as np


class ScaledImageViewer:
    """
    High-performance aspect-ratio locked image viewer.

    Features
    --------
    • Freely resizable window
    • Aspect ratio always preserved
    • Cached resized image (only regenerated when window changes)
    • Pixel-perfect coordinate mapping
    • Smooth ROI drawing
    • Original image never modified
    """

    def __init__(self, image,
                 window_name="Select Target"):

        self.window_name = window_name

        # ------------------------------------------------------------------
        # Original Image
        # ------------------------------------------------------------------

        self.original = image
        self.orig_h, self.orig_w = image.shape[:2]
        self.aspect = self.orig_w / self.orig_h

        # ------------------------------------------------------------------
        # Cached Display
        # ------------------------------------------------------------------

        self.cached_display = None

        self.display_width = self.orig_w
        self.display_height = self.orig_h

        self.scale = 1.0

        self.offset_x = 0
        self.offset_y = 0

        self.window_width = self.orig_w
        self.window_height = self.orig_h

        self.previous_window_size = (-1, -1)

        # ------------------------------------------------------------------
        # ROI State
        # ------------------------------------------------------------------

        self.dragging = False
        self.roi_selected = False

        self.start_point = None
        self.end_point = None

    # ======================================================================
    # Window Utilities
    # ======================================================================

    def _get_window_size(self):

        try:
            _, _, w, h = cv2.getWindowImageRect(self.window_name)

            if w <= 0 or h <= 0:
                raise Exception()

            return w, h

        except Exception:
            return self.window_width, self.window_height

    # ======================================================================

    def _update_display_cache(self):
        """
        Regenerates the cached display image only if the window size changed.
        """

        win_w, win_h = self._get_window_size()

        if (win_w, win_h) == self.previous_window_size:
            return

        self.previous_window_size = (win_w, win_h)

        self.window_width = win_w
        self.window_height = win_h

        window_aspect = win_w / float(win_h)

        # --------------------------------------------------------------
        # Determine display size
        # --------------------------------------------------------------

        if window_aspect > self.aspect:

            self.display_height = win_h
            self.display_width = int(self.display_height * self.aspect)

            self.offset_x = (win_w - self.display_width) // 2
            self.offset_y = 0

        else:

            self.display_width = win_w
            self.display_height = int(self.display_width / self.aspect)

            self.offset_x = 0
            self.offset_y = (win_h - self.display_height) // 2

        # --------------------------------------------------------------
        # Scale
        # --------------------------------------------------------------

        self.scale = self.display_width / float(self.orig_w)

        # --------------------------------------------------------------
        # Resize ONCE
        # --------------------------------------------------------------

        interpolation = (
            cv2.INTER_AREA
            if self.scale < 1.0
            else cv2.INTER_LINEAR
        )

        self.cached_display = cv2.resize(
            self.original,
            (self.display_width,
             self.display_height),
            interpolation=interpolation
        )

    # ======================================================================
    # Coordinate Conversion
    # ======================================================================

    def display_to_image(self, x, y):

        x -= self.offset_x
        y -= self.offset_y

        x = np.clip(x, 0, self.display_width - 1)
        y = np.clip(y, 0, self.display_height - 1)

        img_x = int(round(x / self.scale))
        img_y = int(round(y / self.scale))

        img_x = np.clip(img_x, 0, self.orig_w - 1)
        img_y = np.clip(img_y, 0, self.orig_h - 1)

        return img_x, img_y

    # ======================================================================

    def image_to_display(self, x, y):

        disp_x = int(round(x * self.scale))
        disp_y = int(round(y * self.scale))

        return disp_x, disp_y

    # ======================================================================
    # Mouse Callback
    # ======================================================================

    def mouse_callback(self, event, x, y, flags, param):

        self._update_display_cache()

        img_x, img_y = self.display_to_image(x, y)

        # --------------------------------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:

            self.dragging = True

            self.start_point = (img_x, img_y)
            self.end_point = (img_x, img_y)

        # --------------------------------------------------------------

        elif event == cv2.EVENT_MOUSEMOVE:

            if self.dragging:

                self.end_point = (img_x, img_y)

        # --------------------------------------------------------------

        elif event == cv2.EVENT_LBUTTONUP:

            self.dragging = False

            self.end_point = (img_x, img_y)

            self.roi_selected = True

    # ======================================================================
    # Rendering
    # ======================================================================

    def _build_frame(self):

        self._update_display_cache()

        canvas = np.zeros(
            (
                self.window_height,
                self.window_width,
                3
            ),
            dtype=np.uint8
        )

        canvas[
            self.offset_y:
            self.offset_y + self.display_height,

            self.offset_x:
            self.offset_x + self.display_width

        ] = self.cached_display.copy()

        # --------------------------------------------------------------
        # ROI Overlay
        # --------------------------------------------------------------

        if self.start_point and self.end_point:

            x1, y1 = self.image_to_display(*self.start_point)
            x2, y2 = self.image_to_display(*self.end_point)

            x1 += self.offset_x
            x2 += self.offset_x

            y1 += self.offset_y
            y2 += self.offset_y

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return canvas

    # ======================================================================
    # Public Viewer
    # ======================================================================

    def show(self):

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_ASPECT_RATIO,
            cv2.WINDOW_KEEPRATIO
        )

        # --------------------------------------------------------------
        # Fit image to approximately 90% of a Full-HD screen.
        # User can resize afterwards.
        # --------------------------------------------------------------

        max_w = 1700
        max_h = 900

        scale = min(
            max_w / self.orig_w,
            max_h / self.orig_h
        )

        cv2.resizeWindow(
            self.window_name,
            int(self.orig_w * scale),
            int(self.orig_h * scale)
        )

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback
        )