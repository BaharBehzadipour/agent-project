"""Sketch helpers (black lines on white background, like the paper's conditioning images)."""
import numpy as np
from PIL import Image, ImageFilter


def resize_short_side(img: Image.Image, size: int, resample=Image.BICUBIC) -> Image.Image:
    w, h = img.size
    s = size / min(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return img.resize((nw, nh), resample)


def resize_sketch(sk: Image.Image, size: int) -> Image.Image:
    """Down-size a thin-line sketch WITHOUT losing the lines.

    A 1px line on a 2700px image would vanish at 512px; thicken first by the
    down-scale factor (min filter on dark lines), then area-average.
    """
    sk = sk.convert("L")
    w, h = sk.size
    scale = size / min(w, h)
    if scale < 0.5:
        k = int(round(1 / scale))
        k = k + 1 if k % 2 == 0 else k
        sk = sk.filter(ImageFilter.MinFilter(min(k, 9)))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return sk.resize((nw, nh), Image.BOX)


def adaptive_canny(img: Image.Image, blur=2.0, pct=92, min_len=40, dilate=1) -> Image.Image:
    """Contrast-adaptive Canny, tuned by eye on the dataset's photos.

    * thresholds come from a percentile of the gradient magnitude (NOT the image
      brightness - median-based "auto canny" returns almost nothing on bright
      photos, which is why the original sketches were so sparse),
    * small speckle components are removed,
    * lines are thickened to ~3px so they survive down-scaling.
    Needs opencv (pip install opencv-python-headless).
    """
    import cv2
    g = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    hi = float(np.percentile(np.sqrt(gx ** 2 + gy ** 2), pct))
    e = cv2.Canny(g, 0.4 * hi, hi, L2gradient=True)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_len
    e = (keep[lab] * 255).astype(np.uint8)
    if dilate:
        e = cv2.dilate(e, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    return Image.fromarray(255 - e).convert("L")


def make_sketch(img: Image.Image, mode: str = "adaptive_canny") -> Image.Image:
    if mode == "adaptive_canny":
        return adaptive_canny(img)
    raise ValueError(mode)


def thicken(sk: Image.Image, k: int = 3) -> Image.Image:
    return sk.filter(ImageFilter.MinFilter(k))


def random_erase_lines(sk: Image.Image, rng, n_max=4, size_frac=(0.05, 0.2)) -> Image.Image:
    """Whiten a few random rectangles (simulates incomplete hand sketches)."""
    a = np.array(sk.convert("L"))
    h, w = a.shape
    for _ in range(rng.randint(1, n_max)):
        bh = int(h * rng.uniform(*size_frac))
        bw = int(w * rng.uniform(*size_frac))
        y, x = rng.randint(0, max(0, h - bh)), rng.randint(0, max(0, w - bw))
        a[y:y + bh, x:x + bw] = 255
    return Image.fromarray(a)
