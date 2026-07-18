"""Re-render fiber4_mesh.png from saved PLYs: per-panel render + autocrop +
PIL montage (1x4 labeled strip). No re-simulation."""
import os
import numpy as np
import meshio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageChops, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
# generated artifacts live in <repo>/output/bicep, not next to the script
OUT = os.path.join(HERE, "..", "..", "output", "bicep")
os.makedirs(OUT, exist_ok=True)
MODELS = ["nofiber", "plain", "aniso_arap", "millard"]
LABELS = {"nofiber": "no fiber", "plain": "plain  C=|Fd|",
          "aniso_arap": "aniso ARAP (ours)", "millard": "Hill-Millard"}

pos = {m: meshio.read(os.path.join(OUT, f"fiber4_{m}.ply")).points for m in MODELS}
faces = meshio.read(os.path.join(OUT, "fiber4_nofiber.ply")).cells[0].data
base = pos["nofiber"]
disp = {m: np.linalg.norm(pos[m] - base, axis=1) for m in MODELS}
alld = np.concatenate([disp[m] for m in MODELS])
vmax = float(np.percentile(alld[alld > 0], 92)) or 1.0

mn, mx = base.min(0), base.max(0)
c = (mn + mx) / 2
span = mx - mn
pad = 1.03


def autocrop(im, b=6):
    im = im.convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if not bbox:
        return im
    return im.crop((max(0, bbox[0]-b), max(0, bbox[1]-b),
                    min(im.width, bbox[2]+b), min(im.height, bbox[3]+b)))


def render_panel(m):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111, projection="3d")
    fc = plt.cm.turbo(np.clip(disp[m][faces].mean(1) / vmax, 0, 1))
    ax.add_collection3d(Poly3DCollection(pos[m][faces], facecolors=fc,
                        edgecolors="k", linewidths=0.03, alpha=1.0))
    for lim, ci, si in ((ax.set_xlim, c[0], span[0]), (ax.set_ylim, c[1], span[1]),
                        (ax.set_zlim, c[2], span[2])):
        lim(ci - pad*si/2, ci + pad*si/2)
    ax.set_box_aspect(tuple(span / span.max()))
    ax.set_axis_off()
    ax.view_init(elev=8, azim=-90)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    tmp = os.path.join(OUT, f"_tmp_{m}.png")
    fig.savefig(tmp, dpi=200, transparent=False)
    plt.close(fig)
    im = autocrop(Image.open(tmp))
    os.remove(tmp)
    return im


panels = {m: render_panel(m) for m in MODELS}
PW = max(p.width for p in panels.values())
PH = max(p.height for p in panels.values())

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    tfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 27)
except Exception:
    font = tfont = ImageFont.load_default()

lab_h = 40
title_h = 8   # no title, small top margin
gap = 18
cols = 4
cell_w = PW + gap
canvas = Image.new("RGB", (cols * cell_w + gap, title_h + lab_h + PH + gap), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

for i, m in enumerate(MODELS):
    x0 = gap + i * cell_w
    draw.text((x0 + PW // 2, title_h + lab_h // 2), LABELS[m], fill=(0, 0, 0), font=font, anchor="mm")
    p = panels[m]
    canvas.paste(p, (x0 + (PW - p.width) // 2, title_h + lab_h + (PH - p.height) // 2))

out = os.path.join(OUT, "fiber4_mesh.png")
canvas.save(out)
print("wrote", out, canvas.size)
