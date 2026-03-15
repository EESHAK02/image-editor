import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False
import mediapipe as mp
from io import BytesIO
import streamlit as st


# image conversion helpers

def pil_to_cv2_bgr(img: Image.Image) -> np.ndarray:

    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv2_bgr_to_pil(arr: np.ndarray) -> Image.Image:

    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# color and tone

def tool_apply_sepia(img: Image.Image, intensity: float = 1.0, **_) -> Image.Image:

    intensity = max(0.0, min(1.0, float(intensity)))
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    nr = np.clip(r*(1 - 0.607*intensity) + g*0.769*intensity + b*0.189*intensity, 0, 255)
    ng = np.clip(r*0.349*intensity + g*(1 - 0.314*intensity) + b*0.168*intensity, 0, 255)
    nb = np.clip(r*0.272*intensity + g*0.534*intensity + b*(1 - 0.869*intensity), 0, 255)
    return Image.fromarray(np.stack([nr, ng, nb], axis=2).astype(np.uint8))

def tool_apply_grayscale(img: Image.Image, **_) -> Image.Image:

    return img.convert("L").convert("RGB")

def tool_adjust_brightness(img: Image.Image, value: int = 40, **_) -> Image.Image:

    factor = max(0.1, 1.0 + float(value) / 100.0)
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)

def tool_adjust_contrast(img: Image.Image, value: int = 50, **_) -> Image.Image:

    factor = max(0.1, 1.0 + float(value) / 100.0)
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)

def tool_adjust_saturation(img: Image.Image, value: int = 50, **_) -> Image.Image:

    factor = max(0.0, 1.0 + float(value) / 100.0)
    return ImageEnhance.Color(img.convert("RGB")).enhance(factor)

def tool_adjust_sharpness(img: Image.Image, value: int = 50, **_) -> Image.Image:

    factor = max(0.0, 1.0 + float(value) / 100.0)
    return ImageEnhance.Sharpness(img.convert("RGB")).enhance(factor)

def tool_apply_warmth(img: Image.Image, value: int = 30, **_) -> Image.Image:

    arr = np.array(img.convert("RGB"), dtype=np.int16)
    v = int(value)
    arr[:, :, 0] = np.clip(arr[:, :, 0] - v, 0, 255)  # red channel
    arr[:, :, 2] = np.clip(arr[:, :, 2] + v, 0, 255)  # blue channel
    return Image.fromarray(arr.astype(np.uint8))

# filter and effect tools

def tool_apply_blur(img: Image.Image, radius: int = 5, **_) -> Image.Image:
    # PIL GaussianBlur. No 'must be odd' constraint unlike cv2
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(1, int(radius))))

def tool_apply_sharpen_filter(img: Image.Image, **_) -> Image.Image:
    # Industry-standard UnsharpMask: more controllable than a Laplacian kernel
    return img.convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

def tool_apply_edge_detection(img: Image.Image, threshold1: int = 100, threshold2: int = 200, **_) -> Image.Image:
    # cv2.Canny on BGR array. Returns 3-channel RGB for display consistency
    edges = cv2.Canny(pil_to_cv2_bgr(img), int(threshold1), int(threshold2))
    return Image.fromarray(edges).convert("RGB")

def tool_apply_cartoon(img: Image.Image, **_) -> Image.Image:

    cv_img = pil_to_cv2_bgr(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 5), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(cv_img, d=9, sigmaColor=200, sigmaSpace=200)
    return cv2_bgr_to_pil(cv2.bitwise_and(color, color, mask=edges))

def tool_apply_vignette(img: Image.Image, strength: float = 0.5, **_) -> Image.Image:

    rgb = img.convert("RGB")
    w, h = rgb.size
    strength = max(0.1, min(1.0, float(strength)))
    X = cv2.getGaussianKernel(w, int(w * 0.6))
    Y = cv2.getGaussianKernel(h, int(h * 0.6))
    kernel = (Y * X.T)
    kernel = kernel / kernel.max()
    blend = 1.0 - strength * (1.0 - kernel)
    arr = np.array(rgb, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] * blend, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# presets (combinations of tools)

def tool_apply_vintage(img: Image.Image, **_) -> Image.Image:
    # Sepia (60%) + slight desaturate + vignette + warm tones
    img = tool_apply_sepia(img, intensity=0.6)
    img = tool_adjust_saturation(img, value=-20)
    img = tool_apply_vignette(img, strength=0.45)
    img = tool_apply_warmth(img, value=-15)
    return img

def tool_apply_cinematic(img: Image.Image, **_) -> Image.Image:
    # High contrast + slight desaturate + cool tone + vignette + slight darken
    img = tool_adjust_contrast(img, value=40)
    img = tool_adjust_saturation(img, value=-15)
    img = tool_apply_warmth(img, value=10)
    img = tool_apply_vignette(img, strength=0.5)
    img = tool_adjust_brightness(img, value=-8)
    return img

def tool_apply_hdr(img: Image.Image, **_) -> Image.Image:
    # High contrast + high saturation + sharpen
    img = tool_adjust_contrast(img, value=55)
    img = tool_adjust_saturation(img, value=60)
    img = tool_apply_sharpen_filter(img)
    return img


# geometric transformations

def tool_rotate_image(img: Image.Image, angle: int = 90, **_) -> Image.Image:
    return img.convert("RGB").rotate(int(angle), expand=True)

def tool_flip_horizontal(img: Image.Image, **_) -> Image.Image:
    return img.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)

def tool_flip_vertical(img: Image.Image, **_) -> Image.Image:
    return img.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)

def tool_crop_center(img: Image.Image, pct: int = 80, **_) -> Image.Image:
    """Crop to the centre pct% of the image, preserving aspect ratio."""
    pct = max(10, min(100, int(pct))) / 100.0
    w, h = img.size
    nw, nh = int(w * pct), int(h * pct)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


# AI features

def tool_remove_background(img: Image.Image, **_) -> Image.Image:

    if not REMBG_AVAILABLE:
        st.error('rembg not installed. Run: pip install "rembg[cpu]"')
        return img
    raw = pil_to_bytes(img.convert("RGB"), "PNG")
    out = Image.open(BytesIO(rembg_remove(raw))).convert("RGBA")
    bg = Image.new("RGBA", out.size, (18, 18, 24, 255))
    bg.paste(out, mask=out.split()[3])
    return bg.convert("RGB")

def tool_detect_faces(img: Image.Image, **_) -> Image.Image:

    rgb = np.array(img.convert("RGB"))
    annotated = rgb.copy()
    mp_fd = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        results = fd.process(rgb)
        if results.detections:
            for det in results.detections:
                mp_draw.draw_detection(annotated, det)
    return Image.fromarray(annotated)

def tool_reset(img: Image.Image, **_) -> Image.Image:
    """Restore from original_image in session state."""
    return st.session_state.original_image.copy() if st.session_state.original_image else img


# Tool catalog and execution
# Structure: { "tool_name": (function, "Human Label", {"param": "description"}) }
# The param schema is used to build the agent system prompts dynamically.

TOOL_REGISTRY: dict[str, tuple] = {
    "apply_sepia":          (tool_apply_sepia,          "Sepia",              {"intensity": "float 0.0-1.0"}),
    "apply_grayscale":      (tool_apply_grayscale,      "Grayscale",          {}),
    "adjust_brightness":    (tool_adjust_brightness,    "Brightness",         {"value": "int -100 to 100"}),
    "adjust_contrast":      (tool_adjust_contrast,      "Contrast",           {"value": "int -100 to 100"}),
    "adjust_saturation":    (tool_adjust_saturation,    "Saturation",         {"value": "int -100 to 100"}),
    "adjust_sharpness":     (tool_adjust_sharpness,     "Sharpness",          {"value": "int -100 to 100"}),
    "apply_warmth":         (tool_apply_warmth,         "Warmth/Cool",        {"value": "int, negative=warmer, positive=cooler"}),
    "apply_blur":           (tool_apply_blur,           "Blur",               {"radius": "int 1-30"}),
    "apply_sharpen_filter": (tool_apply_sharpen_filter, "Sharpen",            {}),
    "apply_edge_detection": (tool_apply_edge_detection, "Edge Detection",     {"threshold1": "int 50-200", "threshold2": "int 100-300"}),
    "apply_cartoon":        (tool_apply_cartoon,        "Cartoon",            {}),
    "apply_vignette":       (tool_apply_vignette,       "Vignette",           {"strength": "float 0.1-1.0"}),
    "apply_vintage":        (tool_apply_vintage,        "Vintage preset",     {}),
    "apply_cinematic":      (tool_apply_cinematic,      "Cinematic preset",   {}),
    "apply_hdr":            (tool_apply_hdr,            "HDR preset",         {}),
    "rotate_image":         (tool_rotate_image,         "Rotate",             {"angle": "int degrees"}),
    "flip_horizontal":      (tool_flip_horizontal,      "Flip horizontal",    {}),
    "flip_vertical":        (tool_flip_vertical,        "Flip vertical",      {}),
    "crop_center":          (tool_crop_center,          "Crop centre",        {"pct": "int 10-100, percent of image to keep"}),
    "remove_background":    (tool_remove_background,    "Remove background",  {}),
    "detect_faces":         (tool_detect_faces,         "Detect faces",       {}),
    "reset":                (tool_reset,                "Reset to original",  {}),
}


def build_tool_list_str() -> str:

    lines = []
    for name, (_, label, params) in TOOL_REGISTRY.items():
        ps = ", ".join(f"{k}: {v}" for k, v in params.items()) if params else "no params"
        lines.append(f"  - {name}: {label} ({ps})")
    return "\n".join(lines)


def execute_steps(steps: list[dict]) -> tuple[Image.Image, list[str]]:

    img = st.session_state.current_image.copy()
    applied = []

    for step in steps:
        name = step.get("tool", "")
        params = step.get("params", {})

        if name == "reset":
            img = tool_reset(img)
            applied.append("↺ Reset")
            continue

        if name not in TOOL_REGISTRY:
            st.warning(f"Unknown tool: `{name}` — skipped.")
            continue

        fn, label, _ = TOOL_REGISTRY[name]
        try:
            img = fn(img, **params)
            ps = ", ".join(f"{k}={v}" for k, v in params.items())
            applied.append(label + (f" ({ps})" if ps else ""))
        except Exception as e:
            st.warning(f"Tool `{name}` failed: {e}")

    return img, applied


# Collage maker

COLLAGE_LAYOUTS = {
    "2 side-by-side": {"cols": 2, "rows": 1},
    "2 stacked":      {"cols": 1, "rows": 2},
    "3 horizontal":   {"cols": 3, "rows": 1},
    "4 grid (2×2)":   {"cols": 2, "rows": 2},
    "6 grid (3×2)":   {"cols": 3, "rows": 2},
}

def make_collage(
    images: list[Image.Image],
    cols: int,
    rows: int,
    cell_w: int = 400,
    cell_h: int = 300,
    gap: int = 6,
    bg: tuple = (14, 14, 18),
) -> Image.Image:

    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap),
        bg,
    )
    for i, img in enumerate(images[: cols * rows]):
        col_i, row_i = i % cols, i // cols
        x = gap + col_i * (cell_w + gap)
        y = gap + row_i * (cell_h + gap)
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((cell_w, cell_h), Image.LANCZOS)
        canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
    return canvas
