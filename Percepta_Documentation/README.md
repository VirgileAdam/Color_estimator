<p align="center">
  <img src="docs/images/image(1924).png" alt="Percepta logo and About dialog" width="460">
</p>

<h1 align="center">Percepta</h1>

<p align="center"><strong>Turn photographs into RGB perceptual patterns that reveal the source image at a distance.</strong></p>

<p align="center">
  Vertical stripes · Horizontal stripes · Diagonal stripes · Halftone · Hexagonal halftone · Spiral stripes
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#patterns">Patterns</a> ·
  <a href="#controls">Controls</a> ·
  <a href="#animation">Animation</a> ·
  <a href="docs/Percepta_Technical_Manual.pdf">Technical manual</a>
</p>

---

![Percepta interface — vertical stripes](docs/images/image%281925%29.png)

## What is Percepta?

Percepta is a desktop image-processing application that converts a conventional RGB image into geometric coloured structures. At close range, the output reads as stripes, dots or a spiral. When the image is reduced or viewed from farther away, spatial averaging recombines the red, green and blue information and the source picture emerges.

The image is not simply covered with a pattern. Percepta measures local red, green and blue intensities independently and turns those values into geometry:

- stripe **width** for the line renderers;
- dot **area** for the halftone renderers;
- path **width** along one continuous Archimedean spiral.

Overlapping RGB structures produce white; pairwise overlaps produce cyan, magenta and yellow fringes. This makes the close-up pattern visually explicit while preserving the large-scale colour information needed for reconstruction.

## Highlights

- Six complementary perceptual renderers.
- Independent red, green and blue geometric encoding.
- Interactive crop, zoom, pan and rotation.
- Pattern-specific density units that describe the quantity actually controlled.
- Adjustable strength, source contrast and RGB separation.
- Adaptive defaults referenced to a 1600 px output.
- Density animation with duration, frame rate, easing and ping-pong playback.
- High-resolution still export for screen, layout and print workflows.
- Compact PyQt6 interface with source and result previews.

## Animated example

<p align="center">
  <img src="docs/images/Johannes_Vermeer_(1632-1675)_-_The_Girl_With_The_Pearl_Earring_(1665)_density_animation.gif" alt="Percepta density animation" width="620">
</p>

The animation changes the density parameter while preserving the same source and rendering model. Coarse geometry makes the pattern dominant; finer geometry makes the reconstructed image easier to perceive.

## Quick start

1. Launch Percepta and select **Open image…**.
2. In **Framing**, choose the crop ratio and adjust zoom, pan or rotation.
3. Choose one of the six patterns.
4. Start with the proposed density and tune **Strength**, **Colour separation** and **Source contrast**.
5. Select the output size and press **Generate**.
6. Inspect the result at several preview sizes.
7. Use **Export** for a still image, or the lower animation bar to create a density animation.

> **Viewing tip**  
> A result should be judged both close up and reduced. The near view reveals the RGB construction; the reduced view tests whether the source image reconstructs clearly.

## Patterns

### Vertical stripes

![Vertical stripes](docs/images/image%281925%29.png)

Local RGB intensity changes the width of vertical channel stripes. This is a strong starting point for portraits and centred compositions.

- **Density unit:** number of stripes
- **Reference default at 1600 px:** 14
- **Reference animation:** 8 → 24

### Horizontal stripes

![Horizontal stripes](docs/images/image%281926%29.png)

The same width-modulation principle is applied horizontally. It works naturally with landscapes, horizons and layered scenes.

- **Density unit:** number of stripes
- **Reference default at 1600 px:** 14
- **Reference animation:** 8 → 24

### Diagonal stripes

![Diagonal stripes](docs/images/image%281927%29.png)

The source is encoded in a diagonal coordinate system, producing a more dynamic visual rhythm while retaining the same independent RGB logic.

- **Density unit:** number of stripes
- **Reference default at 1600 px:** 18
- **Reference animation:** 10 → 28

### Halftone

![Halftone](docs/images/image%281928%29.png)

Each grid cell becomes three locally sized colour dots. Because dot area is proportional to source intensity, the image reconstructs when neighbouring dots are visually averaged.

- **Density unit:** distance between neighbouring dots
- **Reference default at 1600 px:** 12 px
- **Reference animation:** 24 px → 7 px

### Hexagonal halftone

![Hexagonal halftone](docs/images/image%281929%29.png)

Dots are placed on a hexagonal lattice with six equidistant neighbours. The arrangement is less directionally biased than a square grid and is well suited to detailed images.

- **Density unit:** nearest-neighbour dot spacing
- **Reference default at 1600 px:** 12 px
- **Reference animation:** 24 px → 7 px

### Spiral stripes

![Spiral stripes](docs/images/image%281930%29.png)

One continuous Archimedean spiral carries the image. Red, green and blue paths are displaced along the local normal and independently modulated in width.

- **Density unit:** distance between successive turns
- **Reference default at 1600 px:** 19 px
- **Reference animation:** 24 px → 6 px

Spiral-specific options include centre X/Y, direction and path smoothing.

![Pattern selector and spiral options](docs/images/image%281931%29.png)

## Controls

### Common rendering controls

| Control | Reference default | What it changes |
|---|---:|---|
| **Pattern** | Vertical stripes | Selects the geometric renderer. |
| **Density** | Pattern dependent | Stripe count, dot spacing or spiral turn spacing. |
| **Strength** | 0.80 | Amount of geometric width/area modulation. |
| **Colour separation** | 10 px | Offset between red, green and blue structures. |
| **Source contrast** | 1.55 | Contrast applied before local sampling. |
| **Output size** | 1600 px | Final square render resolution. |

### Framing

The **Framing** tab modifies the composition without changing the source file:

- crop ratio, including square, portrait and landscape compositions;
- zoom;
- pan X and pan Y;
- rotation;
- reset to the neutral framing state.

The source is rotated before cropping and then fitted to the square render canvas without anisotropic stretching.

### Pattern options

The available options follow the selected family:

- **Lines:** profile smoothing.
- **Halftone:** dot shape, sampling angle and minimum dot size.
- **Spiral:** centre X/Y, clockwise or counter-clockwise direction, and path smoothing.

## How it works

For each output neighbourhood, Percepta samples the source channels

\[
\mathbf{I}(x,y)=\bigl(I_R(x,y),I_G(x,y),I_B(x,y)\bigr)
\]

and creates three masks

\[
\mathbf{M}(x,y)=\bigl(M_R(x,y),M_G(x,y),M_B(x,y)\bigr).
\]

The geometry is chosen so that local coloured area approximates local source intensity. A simplified reconstruction model is a spatial convolution:

\[
P_c(x,y)=\iint M_c(u,v)\,h(x-u,y-v)\,\mathrm{d}u\,\mathrm{d}v,
\]

where \(h\) represents the combined blur of scale reduction, display/print and visual observation. The complete equations for stripes, halftone, the spiral path, animation and viewing geometry are developed in the [technical manual](docs/Percepta_Technical_Manual.pdf).

## Animation

The lower control bar varies the pattern density between a start and end value.

Available controls include:

- start and end parameter;
- duration;
- frames per second;
- easing;
- ping-pong playback;
- creation and preview controls.

For linear easing,

\[
q(t)=q_0+\frac{t}{T}(q_1-q_0).
\]

Use a low-resolution preview to validate motion, then create the final animation at the required output size.

## Export and print

Percepta is designed for both digital viewing and high-resolution output. The current distribution supports still-image export including PNG, TIFF, SVG and PDF. The most faithful raster master is generally PNG or TIFF; document-oriented formats are convenient for layout and print workflows.

For a print of width \(W\) mm made from an \(N\)-pixel output, a feature pitch of \(p_{px}\) pixels becomes

\[
p_{mm}=W\frac{p_{px}}{N}.
\]

Always test a crop at the final physical size: feature visibility and fusion distance depend on print size, medium, lighting and the observer.

## Installation and launch

Percepta is distributed as a compact Python desktop application and can also be packaged as a standalone executable.

For a source distribution containing `main.py` and `assets/`:

```bash
python main.py
```

The application checks required components at startup and installs missing runtime dependencies according to the distributed build configuration. A typical development environment uses Python 3.10 or newer with PyQt6, NumPy, Pillow and the animation/export dependencies included by the project.

## Recommended starting points

| Source | Pattern | First adjustment |
|---|---|---|
| Portrait | Vertical stripes or halftone | Moderate colour separation |
| Landscape | Horizontal or hexagonal | Reduce contrast if sky clips |
| Architecture | Diagonal or vertical | Increase output resolution |
| Detailed illustration | Hexagonal halftone | Decrease dot spacing |
| Abstract composition | Spiral | Move the spiral centre |

## Troubleshooting

**The source cannot be recognised at any scale.**  
Increase strength, use finer density, reduce excessive source contrast, or try halftone for highly detailed material.

**The result is almost solid.**  
Reduce strength, reduce stripe count, or increase dot/turn spacing.

**The RGB fringes dominate the image.**  
Reduce colour separation. The same pixel offset is more aggressive when pattern spacing is small.

**Dark regions disappear.**  
Increase strength or halftone minimum size, and reduce source contrast if shadows are clipped.

**The spiral or lines look noisy.**  
Increase the relevant smoothing option.

## Documentation package

```text
Percepta_Documentation/
├── README.md
└── docs/
    ├── Percepta_Technical_Manual.pdf
    ├── Percepta_Technical_Manual.tex
    └── images/
        ├── image(1924).png
        ├── image(1925).png
        ├── image(1926).png
        ├── image(1927).png
        ├── image(1928).png
        ├── image(1929).png
        ├── image(1930).png
        ├── image(1931).png
        └── Johannes_Vermeer_..._density_animation.gif
```

## Author

**Virgile Adam**  
IBS — CNRS — Université Grenoble Alpes  
[virgile.adam@ibs.fr](mailto:virgile.adam@ibs.fr)  
[www.virgile-adam.com](https://www.virgile-adam.com)

Project page: [github.com/VirgileAdam/Percepta](https://github.com/VirgileAdam/Percepta)

## Copyright and licence

© 2026 Virgile Adam.

No open-source licence is asserted by this README. Add a `LICENSE` file before public distribution to define the permissions granted to users and contributors.
