from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "Percepta_Documentation" / "docs"
IMG = DOCS / "images"
BASE = DOCS / "Percepta_Technical_Manual_base.pdf"
OUT = DOCS / "Percepta_Technical_Manual.pdf"

A4 = fitz.paper_rect("a4")
INK = (0.09, 0.13, 0.20)
MUTED = (0.40, 0.44, 0.52)
BLUE = (0.18, 0.44, 0.93)
RULE = (0.84, 0.87, 0.91)


def fit_rect(container: fitz.Rect, w: float, h: float) -> fitz.Rect:
    scale = min(container.width / w, container.height / h)
    nw, nh = w * scale, h * scale
    x = container.x0 + (container.width - nw) / 2
    y = container.y0 + (container.height - nh) / 2
    return fitz.Rect(x, y, x + nw, y + nh)


def title(page: fitz.Page, text: str, subtitle: str = ""):
    page.draw_rect(fitz.Rect(0, 0, A4.width, 6), color=None, fill=BLUE)
    page.insert_text((48, 58), text, fontsize=23, fontname="helvB", color=INK)
    if subtitle:
        page.insert_textbox(fitz.Rect(48, 68, A4.width - 48, 105), subtitle, fontsize=10.5, fontname="helv", color=MUTED, lineheight=1.25)
    page.draw_line((48, 112), (A4.width - 48, 112), color=RULE, width=0.8)


def footer(page: fitz.Page, number: str):
    page.draw_line((48, A4.height - 38), (A4.width - 48, A4.height - 38), color=RULE, width=0.6)
    page.insert_text((48, A4.height - 22), "PERCEPTA / ILLUSTRATED TECHNICAL MANUAL", fontsize=7.5, color=MUTED)
    page.insert_text((A4.width - 62, A4.height - 22), number, fontsize=7.5, color=MUTED)


def add_image(page: fitz.Page, path: Path, rect: fitz.Rect):
    with Image.open(path) as im:
        target = fit_rect(rect, im.width, im.height)
    page.insert_image(target, filename=str(path), keep_proportion=True)
    page.draw_rect(target, color=RULE, width=0.6)
    return target


def crop_patterns():
    crop_dir = IMG / "crops"
    crop_dir.mkdir(exist_ok=True)
    names = [
        ("image(1925).png", "vertical.png"),
        ("image(1926).png", "horizontal.png"),
        ("image(1927).png", "diagonal.png"),
        ("image(1928).png", "halftone.png"),
        ("image(1929).png", "hexagonal.png"),
        ("image(1930).png", "spiral.png"),
    ]
    for src_name, dst_name in names:
        with Image.open(IMG / src_name).convert("RGB") as im:
            # Generated interface: result image occupies x=913..1385, y=142..614.
            crop = im.crop((913, 142, 1385, 614))
            crop.save(crop_dir / dst_name, quality=94)
    return crop_dir


def build():
    base = fitz.open(BASE)
    out = fitz.open()
    # Cover, about and contents from the typeset manual.
    first_block = min(3, base.page_count)
    out.insert_pdf(base, from_page=0, to_page=first_block - 1)

    # Full interface page.
    p = out.new_page(width=A4.width, height=A4.height)
    title(p, "Illustrated interface", "The screenshots below reproduce the final compact layout and the control values supplied with Percepta 0.31.0.")
    add_image(p, IMG / "image(1925).png", fitz.Rect(44, 128, A4.width - 44, 510))
    p.insert_textbox(fitz.Rect(52, 532, A4.width - 52, 640),
        "The left panel contains image loading, rendering parameters and framing. The centre and right panes show the framed source and generated pattern. Density animation controls remain visible along the bottom edge, allowing still and animated workflows to use the same parameter model.",
        fontsize=10.5, fontname="helv", color=INK, lineheight=1.35)
    p.insert_textbox(fitz.Rect(52, 650, A4.width - 52, 735),
        "Reference values shown: 14 vertical stripes, strength 0.80, colour separation 10 px, source contrast 1.55 and a 1600 px output.",
        fontsize=10, fontname="helvB", color=BLUE, lineheight=1.3)
    footer(p, "FIG. 1")

    # Pattern gallery page.
    crop_dir = crop_patterns()
    p = out.new_page(width=A4.width, height=A4.height)
    title(p, "Six geometric encodings", "The same framed source is redistributed into six different RGB geometries. Compare the near-field structure before judging the reduced image.")
    labels = [
        ("vertical.png", "Vertical stripes"),
        ("horizontal.png", "Horizontal stripes"),
        ("diagonal.png", "Diagonal stripes"),
        ("halftone.png", "Halftone"),
        ("hexagonal.png", "Hexagonal halftone"),
        ("spiral.png", "Spiral stripes"),
    ]
    margin, gap = 45, 18
    cell_w = (A4.width - 2 * margin - gap) / 2
    cell_h = 196
    for i, (name, label) in enumerate(labels):
        col, row = i % 2, i // 2
        x0 = margin + col * (cell_w + gap)
        y0 = 128 + row * 220
        rect = fitz.Rect(x0, y0, x0 + cell_w, y0 + cell_h)
        add_image(p, crop_dir / name, rect)
        p.insert_text((x0, y0 + cell_h + 17), label, fontsize=10, fontname="helvB", color=INK)
    footer(p, "FIG. 2")

    # Pattern options page.
    p = out.new_page(width=A4.width, height=A4.height)
    title(p, "Pattern-specific options", "The interface changes terminology and options so the displayed density corresponds to the quantity that the renderer actually controls.")
    add_image(p, IMG / "image(1931).png", fitz.Rect(44, 128, A4.width - 44, 515))
    p.insert_textbox(fitz.Rect(52, 534, A4.width - 52, 688),
        "For spiral rendering, Turn spacing is expressed directly in output pixels. Centre X and Centre Y position the Archimedean path, Clockwise changes its orientation, and Path smoothing averages sampled intensity along the curve. Line and halftone families expose their own relevant options rather than a generic list.",
        fontsize=10.5, fontname="helv", color=INK, lineheight=1.35)
    p.insert_textbox(fitz.Rect(52, 700, A4.width - 52, 756),
        "This pattern-aware vocabulary is central to Percepta: 14 stripes, 12 px dot spacing and 19 px turn spacing are meaningful user parameters, not internal renderer constants.",
        fontsize=9.8, fontname="helvB", color=BLUE, lineheight=1.3)
    footer(p, "FIG. 3")

    # Animation sequence page.
    p = out.new_page(width=A4.width, height=A4.height)
    title(p, "Density as motion", "A PDF cannot play the supplied GIF, so five representative frames are shown. The README retains the animated version.")
    add_image(p, IMG / "density_animation_sequence.png", fitz.Rect(40, 150, A4.width - 40, 350))
    p.insert_textbox(fitz.Rect(52, 382, A4.width - 52, 520),
        "In this example, hexagonal dot spacing decreases from 24 px to 7 px. Large, sparse dots foreground the construction; smaller spacing increases spatial sampling and progressively strengthens the reconstructed image. Ping-pong playback reverses the same trajectory without a hard reset.",
        fontsize=10.5, fontname="helv", color=INK, lineheight=1.38)
    p.insert_textbox(fitz.Rect(52, 548, A4.width - 52, 660),
        "Reference animation settings\n\nStart 24 px   ·   End 7 px   ·   Duration 4.00 s   ·   12 FPS   ·   Linear easing",
        fontsize=11, fontname="helvB", color=BLUE, lineheight=1.5)
    footer(p, "FIG. 4")

    # Remaining technical pages.
    if base.page_count > first_block:
        out.insert_pdf(base, from_page=first_block, to_page=base.page_count - 1)
    out.set_metadata({
        "title": "Percepta Technical Manual",
        "author": "Virgile Adam",
        "subject": "Perceptual pattern image generator — algorithms and user guide",
        "keywords": "Percepta, RGB, halftone, stripes, Archimedean spiral, visual perception",
    })
    out.save(OUT, garbage=4, deflate=True)
    out.close()
    base.close()


if __name__ == "__main__":
    build()
