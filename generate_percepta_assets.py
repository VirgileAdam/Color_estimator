from __future__ import annotations

import io
import math
import urllib.request
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Percepta_Documentation" / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

DARK = (20, 27, 42)
INK = (27, 34, 52)
MUTED = (102, 112, 133)
PANEL = (244, 246, 250)
RULE = (215, 221, 232)
BLUE = (47, 111, 237)
RED = (229, 62, 77)
GREEN = (22, 166, 106)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def load_source(size: int = 720) -> Image.Image:
    urls = [
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Girl_with_a_Pearl_Earring.jpg?width=1000",
        "https://commons.wikimedia.org/wiki/Special:Redirect/file/Johannes_Vermeer_-_Girl_with_a_Pearl_Earring_-_WGA24666.jpg?width=1000",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PerceptaDocs/1.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                im = Image.open(io.BytesIO(response.read())).convert("RGB")
            return ImageOps.fit(im, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.43))
        except Exception:
            pass
    # Deterministic fallback portrait-like source.
    yy, xx = np.mgrid[0:size, 0:size]
    arr = np.zeros((size, size, 3), dtype=np.float32)
    arr[..., 0] = 12 + 20 * (1 - yy / size)
    arr[..., 1] = 18 + 22 * (1 - yy / size)
    arr[..., 2] = 30 + 35 * (1 - yy / size)
    face = ((xx - 0.53 * size) / (0.22 * size)) ** 2 + ((yy - 0.42 * size) / (0.29 * size)) ** 2 < 1
    arr[face] = [184, 127, 96]
    turban = ((xx - 0.46 * size) / (0.28 * size)) ** 2 + ((yy - 0.22 * size) / (0.16 * size)) ** 2 < 1
    arr[turban] = [32, 78, 120]
    cloth = (yy > 0.58 * size) & (xx > 0.25 * size) & (xx < 0.78 * size)
    arr[cloth] = [171, 126, 45]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def source_array(im: Image.Image, n: int) -> np.ndarray:
    prepared = ImageEnhance.Contrast(ImageOps.fit(im, (n, n), Image.Resampling.LANCZOS)).enhance(1.55)
    return np.asarray(prepared, dtype=np.float32) / 255.0


def smooth(v: np.ndarray, radius: int = 6) -> np.ndarray:
    if radius <= 0:
        return v
    k = np.ones(2 * radius + 1, dtype=np.float32) / (2 * radius + 1)
    return np.convolve(np.pad(v, (radius, radius), mode="edge"), k, mode="valid")


def render_vertical(im: Image.Image, n: int = 14, size: int = 520, strength: float = 0.8, sep: int = 10) -> Image.Image:
    a = source_array(im, size)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    period = size / max(4, n)
    wmax = 0.66 * period
    wmin = max(1.0, 0.018 * period)
    shifts = (-sep, 0, sep)
    for c, shift in enumerate(shifts):
        for k in range(max(4, n)):
            center = (k + 0.5) * period
            lo = max(0, int(center - 0.30 * period))
            hi = min(size, int(center + 0.30 * period) + 1)
            if hi <= lo:
                continue
            profile = smooth(a[:, lo:hi, c].mean(axis=1), 6)
            widths = wmin + (wmax - wmin) * np.clip(strength * profile ** 0.82, 0, 1)
            cx = center + shift
            for y in range(size):
                x0 = max(0, int(round(cx - widths[y] / 2)))
                x1 = min(size, int(round(cx + widths[y] / 2)) + 1)
                if x1 > x0:
                    out[y, x0:x1, c] = 255
    return Image.fromarray(out)


def render_horizontal(im: Image.Image, n: int = 14, size: int = 520) -> Image.Image:
    rotated = im.transpose(Image.Transpose.ROTATE_90)
    result = render_vertical(rotated, n=n, size=size)
    return result.transpose(Image.Transpose.ROTATE_270)


def render_diagonal(im: Image.Image, n: int = 18, size: int = 520) -> Image.Image:
    side = int(math.ceil(size * math.sqrt(2)))
    src = ImageOps.fit(im, (side, side), Image.Resampling.LANCZOS)
    src = src.rotate(45, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    pat = render_vertical(src, n=max(4, round(n * 1.35)), size=side)
    pat = pat.rotate(-45, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    return ImageOps.fit(pat, (size, size), Image.Resampling.LANCZOS)


def render_halftone(im: Image.Image, spacing: int = 12, size: int = 520, hexagonal: bool = False, strength: float = 0.8, sep: int = 10) -> Image.Image:
    a = source_array(im, size)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    p = max(6, int(spacing))
    ystep = p * math.sqrt(3) / 2 if hexagonal else p
    j = 0
    y = p / 2
    while y < size:
        xoff = p / 2 if hexagonal and j % 2 else 0
        x = p / 2 + xoff
        while x < size:
            sx = int(np.clip(round(x), 0, size - 1))
            sy = int(np.clip(round(y), 0, size - 1))
            rr = max(1, int(0.34 * p))
            x0, x1 = max(0, sx - rr), min(size, sx + rr + 1)
            y0, y1 = max(0, sy - rr), min(size, sy + rr + 1)
            vals = a[y0:y1, x0:x1].mean(axis=(0, 1))
            for c, shift in enumerate((-sep * 0.25, 0, sep * 0.25)):
                v = float(np.clip(strength * vals[c], 0, 1))
                r = max(1.0, 0.47 * p * math.sqrt(v))
                cx = x + shift
                box = (cx - r, y - r, cx + r, y + r)
                colour = [0, 0, 0]
                colour[c] = 255
                draw.ellipse(box, fill=tuple(colour))
            x += p
        y += ystep
        j += 1
    return canvas


def render_spiral(im: Image.Image, spacing: int = 19, size: int = 520, strength: float = 0.8, sep: int = 10) -> Image.Image:
    a = source_array(im, size)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    layers = [Image.new("L", (size, size), 0) for _ in range(3)]
    draws = [ImageDraw.Draw(layer) for layer in layers]
    x0 = y0 = (size - 1) / 2
    b = spacing / (2 * math.pi)
    corner = math.hypot(x0, y0) * 1.02
    theta_max = corner / b
    samples = int(np.clip(theta_max * max(18, spacing), 9000, 60000))
    theta = np.linspace(0, theta_max, samples, dtype=np.float32)
    r = b * theta
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    dx = b * np.cos(theta) - r * np.sin(theta)
    dy = b * np.sin(theta) + r * np.cos(theta)
    norm = np.maximum(np.hypot(dx, dy), 1e-6)
    nx, ny = -dy / norm, dx / norm
    sx = np.clip(np.rint(x).astype(int), 0, size - 1)
    sy = np.clip(np.rint(y).astype(int), 0, size - 1)
    values = a[sy, sx]
    for c in range(3):
        values[:, c] = smooth(values[:, c], 6)
    wmin = max(1.0, 0.018 * spacing)
    wmax = max(wmin + 1.0, 0.88 * spacing)
    offsets = (-min(sep, 0.95 * spacing), 0, min(sep, 0.95 * spacing))
    stride = max(2, samples // 9000)
    for c in range(3):
        widths = wmin + (wmax - wmin) * np.clip(strength * values[:, c] ** 0.82, 0, 1)
        xx = x + offsets[c] * nx
        yy = y + offsets[c] * ny
        for i in range(0, samples - stride, stride):
            p0 = (float(xx[i]), float(yy[i]))
            p1 = (float(xx[i + stride]), float(yy[i + stride]))
            draws[c].line([p0, p1], fill=255, width=max(1, int(round(widths[i]))))
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for c, layer in enumerate(layers):
        arr[..., c] = np.asarray(layer)
    return Image.fromarray(arr)


def rounded(draw: ImageDraw.ImageDraw, box, radius=10, fill=(255, 255, 255), outline=RULE, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_preview(im: Image.Image, box: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(im, box, Image.Resampling.LANCZOS)


def make_logo(width=560, height=220) -> Image.Image:
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    d.text((width / 2, 56), "PERCEPTA", font=font(48, True), fill=INK, anchor="mm")
    x0, y0 = width / 2 - 130, 104
    for x in range(0, 260, 18):
        d.line((x0 + x, y0, x0 + x, y0 + 54), fill=RED, width=5)
        d.line((x0 + x + 5, y0, x0 + x + 5, y0 + 54), fill=GREEN, width=5)
        d.line((x0 + x + 10, y0, x0 + x + 10, y0 + 54), fill=BLUE, width=5)
    d.text((width / 2, 186), "Perceptual pattern image generator", font=font(18), fill=MUTED, anchor="mm")
    return im


def make_about() -> Image.Image:
    im = Image.new("RGB", (700, 430), PANEL)
    d = ImageDraw.Draw(im)
    rounded(d, (70, 30, 630, 400), 18, "white", RULE, 2)
    logo = make_logo(450, 180)
    im.paste(logo, (125, 48))
    d.text((350, 239), "Percepta", font=font(25, True), fill=INK, anchor="mm")
    d.text((350, 273), "Version 0.31.0", font=font(16), fill=MUTED, anchor="mm")
    d.text((350, 307), "© 2026 Virgile Adam", font=font(16), fill=INK, anchor="mm")
    d.text((350, 338), "IBS — CNRS — Université Grenoble Alpes", font=font(15), fill=MUTED, anchor="mm")
    d.text((350, 368), "virgile.adam@ibs.fr   ·   GitHub / Percepta", font=font(14), fill=BLUE, anchor="mm")
    return im


def field(draw, y, label, value, unit=""):
    draw.text((28, y), label, font=font(16), fill=INK)
    rounded(draw, (178, y - 7, 315, y + 26), 6, "white", RULE, 1)
    draw.text((300, y + 9), f"{value}{unit}", font=font(15), fill=INK, anchor="rm")


def make_ui(source: Image.Image, result: Image.Image, pattern: str, density_label: str, density_value: str, anim_start: str, anim_end: str, filename: str, dropdown: bool = False):
    W, H = 1440, 850
    im = Image.new("RGB", (W, H), (238, 241, 246))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 42), fill=DARK)
    d.text((22, 21), "PERCEPTA", font=font(19, True), fill="white", anchor="lm")
    d.text((W - 24, 21), "—   □   ×", font=font(18), fill=(220, 225, 235), anchor="rm")
    d.rectangle((0, 42, W, 75), fill="white")
    d.text((22, 59), "File", font=font(15), fill=INK, anchor="lm")
    d.text((70, 59), "Help", font=font(15), fill=INK, anchor="lm")

    rounded(d, (14, 88, 338, 672), 12, "white", RULE, 1)
    d.text((28, 112), "Image", font=font(18, True), fill=INK)
    rounded(d, (28, 132, 315, 170), 7, PANEL, RULE, 1)
    d.text((45, 151), "Open image…", font=font(15, True), fill=INK, anchor="lm")
    d.text((28, 190), "Girl with a Pearl Earring.jpg", font=font(12), fill=MUTED)
    d.line((28, 212, 315, 212), fill=RULE, width=1)
    d.text((28, 238), "Rendering", font=font(18, True), fill=INK)
    field(d, 272, "Pattern", pattern)
    field(d, 316, density_label, density_value)
    field(d, 360, "Strength", "0.80")
    field(d, 404, "Colour separation", "10", " px")
    field(d, 448, "Source contrast", "1.55")
    field(d, 492, "Output size", "1600", " px")
    rounded(d, (28, 535, 164, 580), 8, BLUE, BLUE, 1)
    d.text((96, 558), "Generate", font=font(16, True), fill="white", anchor="mm")
    rounded(d, (179, 535, 315, 580), 8, "white", RULE, 1)
    d.text((247, 558), "Export…", font=font(16, True), fill=INK, anchor="mm")
    d.text((28, 614), "Framing", font=font(15, True), fill=BLUE)
    d.text((150, 614), "Pattern options", font=font(15), fill=MUTED)
    d.line((28, 633, 315, 633), fill=RULE, width=1)
    d.text((28, 652), "Crop: Square   Zoom: 1.00   Rotation: 0°", font=font(12), fill=MUTED)

    for x, title, preview in [(360, "SOURCE", source), (892, "GENERATED", result)]:
        rounded(d, (x, 88, x + 514, 638), 12, "white", RULE, 1)
        d.text((x + 22, 114), title, font=font(14, True), fill=MUTED)
        p = fit_preview(preview, (472, 472))
        im.paste(p, (x + 21, 142))

    rounded(d, (14, 690, 1426, 828), 12, "white", RULE, 1)
    d.text((32, 716), "Density animation", font=font(17, True), fill=INK)
    labels = [(32, "Parameter"), (164, anim_start), (254, "→"), (300, anim_end), (428, "Duration"), (525, "4.00 s"), (650, "FPS"), (708, "12"), (790, "Linear"), (910, "Ping-pong")]
    for x, txt in labels:
        d.text((x, 765), txt, font=font(15, txt in {"Parameter", "Duration", "FPS"}), fill=INK, anchor="lm")
    rounded(d, (1120, 738, 1250, 790), 8, PANEL, RULE, 1)
    d.text((1185, 764), "Preview", font=font(15, True), fill=INK, anchor="mm")
    rounded(d, (1264, 738, 1398, 790), 8, BLUE, BLUE, 1)
    d.text((1331, 764), "Create", font=font(15, True), fill="white", anchor="mm")
    d.text((360, 663), f"Pattern: {pattern} — 1600 × 1600 px", font=font(13), fill=MUTED)

    if dropdown:
        rounded(d, (178, 265, 330, 472), 6, "white", RULE, 1)
        items = ["Vertical stripes", "Horizontal stripes", "Diagonal stripes", "Halftone", "Hexagonal halftone", "Spiral stripes"]
        for i, item in enumerate(items):
            y = 287 + i * 30
            if item == "Spiral stripes":
                d.rectangle((181, y - 14, 327, y + 14), fill=(231, 238, 254))
            d.text((190, y), item, font=font(13), fill=INK, anchor="lm")
        rounded(d, (20, 584, 330, 670), 7, PANEL, RULE, 1)
        d.text((32, 605), "Spiral", font=font(15, True), fill=INK)
        d.text((32, 632), "Centre X  50%    Centre Y  50%", font=font(12), fill=MUTED)
        d.text((32, 655), "Clockwise    Path smoothing  6", font=font(12), fill=MUTED)

    im.save(OUT / filename, quality=95)


def main():
    source = load_source(720)
    source.save(OUT / "source_reference.jpg", quality=92)
    vertical = render_vertical(source, 14)
    horizontal = render_horizontal(source, 14)
    diagonal = render_diagonal(source, 18)
    halftone = render_halftone(source, 12, hexagonal=False)
    hexagonal = render_halftone(source, 12, hexagonal=True)
    spiral = render_spiral(source, 19)

    make_about().save(OUT / "image(1924).png")
    make_ui(source, vertical, "Vertical stripes", "Number of stripes", "14", "8", "24", "image(1925).png")
    make_ui(source, horizontal, "Horizontal stripes", "Number of stripes", "14", "8", "24", "image(1926).png")
    make_ui(source, diagonal, "Diagonal stripes", "Number of stripes", "18", "10", "28", "image(1927).png")
    make_ui(source, halftone, "Halftone", "Dot spacing", "12 px", "24 px", "7 px", "image(1928).png")
    make_ui(source, hexagonal, "Hexagonal halftone", "Dot spacing", "12 px", "24 px", "7 px", "image(1929).png")
    make_ui(source, spiral, "Spiral stripes", "Turn spacing", "19 px", "24 px", "6 px", "image(1930).png")
    make_ui(source, spiral, "Spiral stripes", "Turn spacing", "19 px", "24 px", "6 px", "image(1931).png", dropdown=True)

    frames = []
    strip_frames = []
    values = np.linspace(24, 7, 22)
    for idx, value in enumerate(values):
        frame = render_halftone(source, int(round(value)), size=420, hexagonal=True)
        frames.append(np.asarray(frame))
        if idx in {0, 5, 10, 16, 21}:
            strip_frames.append(frame.resize((230, 230), Image.Resampling.LANCZOS))
    gif_name = "Johannes_Vermeer_(1632-1675)_-_The_Girl_With_The_Pearl_Earring_(1665)_density_animation.gif"
    imageio.mimsave(OUT / gif_name, frames, duration=0.11, loop=0)
    strip = Image.new("RGB", (len(strip_frames) * 230, 270), "white")
    sd = ImageDraw.Draw(strip)
    for i, frame in enumerate(strip_frames):
        strip.paste(frame, (i * 230, 0))
        val = [24, 20, 16, 11, 7][i]
        sd.text((i * 230 + 115, 250), f"{val} px", font=font(14, True), fill=INK, anchor="mm")
    strip.save(OUT / "density_animation_sequence.png")


if __name__ == "__main__":
    main()
