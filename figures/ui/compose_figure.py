"""Crop raw captures into final panels and compose Extended Data Fig. 3."""
from PIL import Image, ImageDraw, ImageFont

OUT = "figures/ui"
def font(sz, kr=False):
    names = (["malgun.ttf", "malgunbd.ttf"] if kr else []) + \
            ["arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            pass
    return ImageFont.load_default()

# --- final panel crops (all source pixels at 2x scale) ---
a = Image.open(f"{OUT}/panel_a_input.png").crop((430, 170, 2690, 1340))
b = Image.open(f"{OUT}/panel_b_graph.png").crop((250, 0, 2700, 1880))
c = Image.open(f"{OUT}/panel_c_glucose_sheet.png")

a.save(f"{OUT}/panel_a.png")
b.save(f"{OUT}/panel_b.png")
c.save(f"{OUT}/panel_c.png")

# --- composite ---
W = 2880; GUT = 46; GAP = 40; LABEL = 92; TITLE = 150
a_c = a.resize((W - 2*GUT, int(a.height * (W - 2*GUT) / a.width)), Image.LANCZOS)

bh = 1500
b_c = b.resize((int(b.width * bh / b.height), bh), Image.LANCZOS)
c_c = c.resize((int(c.width * bh / c.height), bh), Image.LANCZOS)
row_w = b_c.width + GAP + c_c.width
if row_w < W - 2*GUT:
    k = (W - 2*GUT) / row_w
    b_c = b_c.resize((int(b_c.width * k), int(b_c.height * k)), Image.LANCZOS)
    c_c = c_c.resize((int(c_c.width * k), int(c_c.height * k)), Image.LANCZOS)
    bh = b_c.height

H = TITLE + LABEL + a_c.height + GAP + LABEL + bh + GUT
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)

f_title = font(54); f_lab = font(66); f_sub = font(30, kr=True)
d.text((GUT, 42), "Extended Data Fig. 3  |  Human13 web interface and example state-tracing workflow",
       fill="#111", font=f_title)
d.text((GUT, 108), "Case: tirzepatide injection (Mounjaro) — saved run a6b1629ace8346f4, "
                   "illustrative interface example (not a benchmark panel)", fill="#556", font=f_sub)

def panel(img, x, y, label):
    d.text((x, y), label, fill="#111", font=f_lab)
    y0 = y + LABEL
    canvas.paste(img, (x, y0))
    d.rectangle([x, y0, x + img.width - 1, y0 + img.height - 1], outline="#c9d2dc", width=2)
    return y0 + img.height

ya = panel(a_c, GUT, TITLE, "a")
yb = ya + GAP
panel(b_c, GUT, yb, "b")
panel(c_c, GUT + b_c.width + GAP, yb, "c")

canvas.save(f"{OUT}/extended_data_fig3.png")
canvas.save(f"{OUT}/extended_data_fig3.pdf")
print("saved", canvas.size)
