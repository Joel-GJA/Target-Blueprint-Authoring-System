import cv2
import numpy as np

class ScaledImageViewer:
    """
    Aspect-ratio preserving ROI selector with cached rendering.
    """

    def __init__(self, image, window_name="Select Target"):
        self.window_name = window_name
        self.original = image
        self.orig_h, self.orig_w = image.shape[:2]
        self.aspect = self.orig_w / self.orig_h

        self.cached_display = None
        self.display_width = self.orig_w
        self.display_height = self.orig_h
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.window_width = self.orig_w
        self.window_height = self.orig_h
        self.previous_window_size = (-1, -1)

        self.dragging = False
        self.roi_selected = False
        self.start_point = None
        self.end_point = None

    def _get_window_size(self):
        try:
            _, _, w, h = cv2.getWindowImageRect(self.window_name)
            if w <= 0 or h <= 0:
                raise RuntimeError
            return w, h
        except Exception:
            return self.window_width, self.window_height

    def _update_display_cache(self):
        win_w, win_h = self._get_window_size()

        if (win_w, win_h) == self.previous_window_size:
            return

        self.previous_window_size = (win_w, win_h)
        self.window_width = win_w
        self.window_height = win_h

        if (win_w / float(win_h)) > self.aspect:
            self.display_height = win_h
            self.display_width = int(self.display_height * self.aspect)
            self.offset_x = (win_w - self.display_width) // 2
            self.offset_y = 0
        else:
            self.display_width = win_w
            self.display_height = int(self.display_width / self.aspect)
            self.offset_x = 0
            self.offset_y = (win_h - self.display_height) // 2

        self.scale = self.display_width / float(self.orig_w)

        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        self.cached_display = cv2.resize(
            self.original,
            (self.display_width, self.display_height),
            interpolation=interp
        )

    def display_to_image(self, x, y):
        x = np.clip(x - self.offset_x, 0, self.display_width - 1)
        y = np.clip(y - self.offset_y, 0, self.display_height - 1)
        ix = int(round(x / self.scale))
        iy = int(round(y / self.scale))
        return (
            int(np.clip(ix, 0, self.orig_w - 1)),
            int(np.clip(iy, 0, self.orig_h - 1))
        )

    def image_to_display(self, x, y):
        return (
            int(round(x * self.scale)),
            int(round(y * self.scale))
        )

    def mouse_callback(self, event, x, y, flags, param):
        self._update_display_cache()
        ix, iy = self.display_to_image(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start_point = (ix, iy)
            self.end_point = (ix, iy)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_point = (ix, iy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.end_point = (ix, iy)
            self.roi_selected = True

    def _build_frame(self):
        self._update_display_cache()

        canvas = np.zeros(
            (self.window_height, self.window_width, 3),
            dtype=np.uint8
        )

        h, w = self.cached_display.shape[:2]
        canvas[
            self.offset_y:self.offset_y + h,
            self.offset_x:self.offset_x + w
        ] = self.cached_display

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

    def show(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_ASPECT_RATIO,
            cv2.WINDOW_KEEPRATIO
        )

        max_w = 1700
        max_h = 900
        s = min(max_w / self.orig_w, max_h / self.orig_h)

        cv2.resizeWindow(
            self.window_name,
            int(self.orig_w * s),
            int(self.orig_h * s)
        )

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback
        )

    def select_roi(self):
        self.show()

        print("Resize window, drag ROI, ENTER/SPACE confirm, R reset, ESC cancel")

        while True:
            cv2.imshow(self.window_name, self._build_frame())
            key = cv2.waitKey(16) & 0xFF

            if key in (13, 32):
                if self.roi_selected:
                    break

            elif key == ord("r"):
                self.start_point = None
                self.end_point = None
                self.dragging = False
                self.roi_selected = False

            elif key == 27:
                cv2.destroyWindow(self.window_name)
                return None

        cv2.destroyWindow(self.window_name)

        x1 = min(self.start_point[0], self.end_point[0])
        y1 = min(self.start_point[1], self.end_point[1])
        x2 = max(self.start_point[0], self.end_point[0])
        y2 = max(self.start_point[1], self.end_point[1])

        return (x1, y1, x2 - x1, y2 - y1)

# Usage:
image = cv2.imread('dataset\\20260701_163736.jpg')
selector = ScaledImageViewer(image)
roi = selector.select_roi()
print("Selected ROI:", roi)  # roi returns (x, y, width, height) in ORIGINAL image coordinates.
