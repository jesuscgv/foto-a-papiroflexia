"""Cuadernillo de regalo estilo revista: portada, materiales y leyenda, mapa de piezas, paso a paso,
hojas de piezas y contraportada con dedicatoria.

python kit.py carpeta --nombre "Mi figura" [--alto 18] [--compacto 0.6] [--credito "..."]
Necesita en la carpeta: low.obj + texturas (lo de siempre) y render.png (render.py, fondo transparente).
"""
import argparse, json, math, subprocess, sys
from pathlib import Path

import cairo
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from unfold import layout, affine_to, A4, MARGIN

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
INK = (0.16, 0.14, 0.12)
MUTED = (0.52, 0.48, 0.44)
ACCENT = (0.84, 0.43, 0.20)      # naranja de acento
CREAM = (0.955, 0.932, 0.898)
RULE = (0.84, 0.80, 0.75)
TABFILL = (0.965, 0.950, 0.930)
DIDOT, SANS, ITAL = "Didot", "Avenir Next", "Avenir Next"  # cursivas con serifa se descuadran en cairo
PALETTE = [(236, 170, 124), (156, 198, 178), (244, 210, 138), (168, 186, 216), (224, 158, 170),
           (196, 214, 148), (248, 192, 150), (186, 168, 212), (148, 202, 212), (232, 202, 168),
           (212, 148, 128), (178, 212, 188), (240, 222, 176), (204, 180, 156), (158, 178, 204),
           (222, 190, 212), (190, 202, 138), (250, 172, 140), (170, 200, 160), (210, 170, 200)]
HEAD, FOOT = 14.0, 17.0


# ---------------------------------------------------------------- tipografía
def font(ctx, fam, size, bold=False, italic=False):
    ctx.select_font_face(fam, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def width(ctx, s, tracking=0.0):
    return ctx.text_extents(s).x_advance + tracking * max(0, len(s) - 1)


def put(ctx, x, y, s, rgb=INK, align="left", tracking=0.0, alpha=1.0):
    w = width(ctx, s, tracking)
    x0 = x - (w / 2 if align == "center" else w if align == "right" else 0)
    ctx.set_source_rgba(*rgb, alpha)
    if tracking:
        for ch in s:
            ctx.move_to(x0, y); ctx.show_text(ch); x0 += ctx.text_extents(ch).x_advance + tracking
    else:
        ctx.move_to(x0, y); ctx.show_text(s)
    return w


def wrap(ctx, s, maxw):
    lines, cur = [], ""
    for wd in s.split():
        t = (cur + " " + wd).strip()
        if width(ctx, t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    return lines


def para(ctx, x, y, w, s, fam=SANS, size=3.6, lead=1.5, rgb=INK, italic=False, bold=False, align="left"):
    font(ctx, fam, size, bold, italic)
    for ln in wrap(ctx, s, w):
        put(ctx, x + (w / 2 if align == "center" else 0), y, ln, rgb, align=align)
        y += size * lead
    return y


def kicker(ctx, x, y, s, rgb=ACCENT, align="left", size=2.7):
    font(ctx, SANS, size, bold=True)
    return put(ctx, x, y, s.upper(), rgb, align=align, tracking=size * 0.32)


def rule(ctx, x0, y, x1, rgb=RULE, w=0.25):
    ctx.set_source_rgb(*rgb); ctx.set_line_width(w); ctx.move_to(x0, y); ctx.line_to(x1, y); ctx.stroke()


def circle_num(ctx, x, y, r, s, fill=ACCENT, txt=(1, 1, 1), size=None, stroke=None):
    ctx.new_path()
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.set_source_rgb(*fill); ctx.fill_preserve()
    if stroke:
        ctx.set_source_rgb(*stroke); ctx.set_line_width(0.35); ctx.stroke()
    else:
        ctx.new_path()
    font(ctx, SANS, size or r * 1.15, bold=True)
    e = ctx.text_extents(s)
    ctx.set_source_rgb(*txt); ctx.move_to(x - e.width / 2 - e.x_bearing, y - e.height / 2 - e.y_bearing); ctx.show_text(s)


def png(ctx, path, x, y, w=None, h=None, box=None):
    """Pega un PNG. Con box=(x, y, w, h) lo encaja centrado."""
    im = cairo.ImageSurface.create_from_png(str(path))
    iw, ih = im.get_width(), im.get_height()
    if box:
        bx, by, bw, bh = box
        s = min(bw / iw, bh / ih)
        x, y = bx + (bw - iw * s) / 2, by + (bh - ih * s) / 2
    else:
        s = w / iw if w else h / ih
    ctx.save(); ctx.translate(x, y); ctx.scale(s, s)
    ctx.set_source_surface(im, 0, 0); ctx.get_source().set_filter(cairo.FILTER_GOOD); ctx.paint(); ctx.restore()
    return x, y, iw * s, ih * s, s


def crop_alpha(src, dst, pad=0.04, thr=8, feather=0.0):
    """Recorta al contenido (alfa > thr). feather: funde a transparente los bordes (fracción del lado)."""
    im = Image.open(src).convert("RGBA")
    a = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(a > thr)
    if not len(xs):
        im.save(dst); return (0, 0, im.width, im.height)
    p = int(pad * max(im.size))
    box = (max(0, xs.min() - p), max(0, ys.min() - p), min(im.width, xs.max() + p), min(im.height, ys.max() + p))
    im = im.crop(box)
    if feather:
        arr = np.asarray(im).astype(np.float32)
        h, w = arr.shape[:2]
        fy = np.minimum(np.arange(h), np.arange(h)[::-1]) / (feather * h)
        fx = np.minimum(np.arange(w), np.arange(w)[::-1]) / (feather * w)
        f = np.clip(np.minimum.outer(fy, fx), 0, 1)
        arr[..., 3] *= f * f * (3 - 2 * f)
        im = Image.fromarray(arr.astype(np.uint8))
    im.save(dst)
    return box


# ---------------------------------------------------------------- orden de montaje
def assembly(L):
    V, F, edges, cut, isl_of, pn = L["V"], L["F"], L["edges"], L["cut"], L["isl_of"], L["piece_name"]
    n = len(pn)
    face_piece = [pn[isl_of[fi]] for fi in range(len(F))]
    links = {}       # (a, b) -> [números de pestaña]
    for num, e in enumerate(cut, 1):
        a, b = sorted((face_piece[edges[e][0]], face_piece[edges[e][1]]))
        links.setdefault((a, b), []).append(num)
    area = {pn[k]: L["islands"][k]["poly"].area for k in pn}
    # empieza por la pieza del hocico: la cara más adelantada de la mitad superior
    cz = V[F].mean(1)
    up = cz[:, 1] > np.median(cz[:, 1])
    start = face_piece[int(np.argmax(np.where(up, cz[:, 2], -1e9)))]
    order, done = [start], {start}
    while len(done) < n:
        best = max((p for p in range(1, n + 1) if p not in done),
                   key=lambda p: (sum(len(links.get(tuple(sorted((p, q))), [])) for q in done), area[p]))
        order.append(best); done.add(best)
    # pasos: las piezas pequeñas se juntan con la anterior si están unidas a ella
    med = np.median(list(area.values()))
    steps = []
    for p in order:
        if steps and area[p] < 0.3 * med and len(steps[-1]) < 3 and any(
                tuple(sorted((p, q))) in links for q in steps[-1]):
            steps[-1].append(p)
        else:
            steps.append([p])
    return face_piece, links, steps


def step_text(i, st, links, done):
    who = f"Pieza {st[0]}" if len(st) == 1 else "Piezas " + ", ".join(map(str, st[:-1])) + f" y {st[-1]}"
    own = sorted(n for (a, b), ns in links.items() for n in ns if a in st and b in st)
    to = sorted(n for (a, b), ns in links.items() for n in ns if (a in st) != (b in st) and (a in done or b in done))
    mates = sorted({b if a in st else a for (a, b) in links if (a in st) != (b in st) and (a in done or b in done)})
    fmt = lambda ns: ", ".join(map(str, ns[:7])) + (f" (+{len(ns) - 7})" if len(ns) > 7 else "")
    lines = []
    if i == 1:
        lines.append("Empieza por aquí.")
    if own:
        lines.append(f"Cierra sus pestañas: {fmt(own)}.")
    if to:
        lines.append(f"Únela a la {', '.join(map(str, mates))}: pestañas {fmt(to)}.")
    return who, lines


# ---------------------------------------------------------------- páginas
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--nombre", default="Mi figura")
    ap.add_argument("--alto", type=float, default=18.0)
    ap.add_argument("--compacto", type=float, default=0.6)
    ap.add_argument("--separacion", type=float, default=5.0, help="mm entre piezas en las hojas (sitio para los números)")
    ap.add_argument("--horas", default="3–4")
    ap.add_argument("--credito", default="foto → papiroflexia · herramienta de @jesuscgv · modelo 3D con MakerWorld Image-to-3D")
    ap.add_argument("--sin-render", action="store_true", help="reutiliza las ilustraciones ya generadas")
    a = ap.parse_args()
    d = Path(a.dir)
    L = layout(d, a.alto, a.compacto, head=HEAD, foot=FOOT, gap=a.separacion)
    V, F, src, texfiles = L["V"], L["F"], L["src"], L["texfiles"]
    edges, info, joined, islands = L["edges"], L["info"], L["joined"], L["islands"]
    tabs, labels, pages, pn, cut = L["tabs"], L["labels"], L["pages"], L["piece_name"], L["cut"]
    face_piece, links, steps = assembly(L)
    npcs = len(pn)

    # modelo.obj con el mismo orden de caras (lo escribe unfold.py; aquí lo regeneramos por si acaso)
    subprocess.run([sys.executable, str(Path(__file__).parent / "unfold.py"), str(d), "--alto", str(a.alto),
                    "--compacto", str(a.compacto), "--dpi", "20"], check=True, capture_output=True)
    cents = (V[F].mean(1) * L["scale"]).round(4).tolist()
    json.dump({"face_piece": face_piece, "centroids": cents, "steps": steps, "palette": PALETTE[:max(npcs, 1)]},
              open(d / "kit_piezas.json", "w"))
    kd = d / "kit"
    if not a.sin_render:
        r = subprocess.run([BLENDER, "-b", "-P", str(Path(__file__).parent / "piecemap.py"), "--", str(d)],
                           capture_output=True, text=True)
        if "PIECEMAP OK" not in r.stdout:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit("falló piecemap")
    mapa = json.load(open(kd / "mapa.json"))
    hero = crop_alpha(d / "render.png", kd / "hero.png", pad=0.06, thr=120, feather=0.12)
    boxes = {nm: crop_alpha(kd / f"mapa_{nm}.png", kd / f"mapa_{nm}_c.png") for nm in ("frente", "atras")}
    for i in range(1, len(steps) + 1):
        crop_alpha(kd / f"paso_{i:02d}.png", kd / f"paso_{i:02d}_c.png")
    sheet_of = {}
    for pi, items in enumerate(pages, 1):
        for k, *_ in items:
            sheet_of[pn[k]] = pi

    W, H = A4
    M = 18.0  # margen de las páginas editoriales

    # ------------------------------------------------ 1. portada
    def cover(ctx):
        ctx.set_source_rgb(*CREAM); ctx.paint()
        kicker(ctx, W / 2, 24, "Papercraft", align="center")
        font(ctx, DIDOT, 50)
        put(ctx, W / 2, 70, a.nombre, INK, align="center")
        font(ctx, ITAL, 4.8, italic=True)
        put(ctx, W / 2, 82, f"Busto de papel para montar · {a.alto:g} cm", MUTED, align="center")
        png(ctx, kd / "hero.png", 0, 0, box=(M, 88, W - 2 * M, 160))
        # ficha técnica
        y0 = 258
        rule(ctx, M, y0 - 8, W - M, RULE, 0.3)
        specs = [(f"{a.alto:g}", "cm de alto"), (str(npcs), "piezas"), (str(len(pages)), "hojas A4"),
                 (a.horas, "horas de montaje")]
        cw = (W - 2 * M) / 4
        for i, (big, small) in enumerate(specs):
            cx = M + cw * (i + 0.5)
            font(ctx, DIDOT, 11); put(ctx, cx, y0 + 4, big, INK, align="center")
            font(ctx, SANS, 2.6); put(ctx, cx, y0 + 10, small.upper(), MUTED, align="center", tracking=0.6)
            if i:
                ctx.set_source_rgb(*RULE); ctx.set_line_width(0.25)
                ctx.move_to(M + cw * i, y0 - 4); ctx.line_to(M + cw * i, y0 + 12); ctx.stroke()
        rule(ctx, M, y0 + 17, W - M, RULE, 0.3)

    # ------------------------------------------------ 2. antes de empezar
    def legend_sample(ctx, x, y, kind):
        ctx.save()
        if kind == "corte":
            ctx.set_source_rgb(*INK); ctx.set_line_width(0.35); ctx.move_to(x, y); ctx.line_to(x + 16, y); ctx.stroke()
        elif kind == "monte":
            ctx.set_source_rgb(*INK); ctx.set_line_width(0.3); ctx.set_dash([1.6, 1.0])
            ctx.move_to(x, y); ctx.line_to(x + 16, y); ctx.stroke()
        elif kind == "valle":
            ctx.set_source_rgb(*INK); ctx.set_line_width(0.3); ctx.set_dash([2.2, 0.8, 0.4, 0.8])
            ctx.move_to(x, y); ctx.line_to(x + 16, y); ctx.stroke()
        elif kind == "pestaña":
            P = [(x, y + 2), (x + 16, y + 2), (x + 13, y - 3), (x + 3, y - 3)]
            ctx.move_to(*P[0]); [ctx.line_to(*p) for p in P[1:]]; ctx.close_path()
            ctx.set_source_rgb(*TABFILL); ctx.fill_preserve(); ctx.set_source_rgb(*INK); ctx.set_line_width(0.3); ctx.stroke()
            font(ctx, SANS, 2.4, bold=True); put(ctx, x + 8, y + 0.4, "24", MUTED, align="center")
        elif kind == "pieza":
            circle_num(ctx, x + 8, y - 0.5, 3.2, "7")
        ctx.restore()

    def intro(ctx):
        ctx.set_source_rgb(1, 1, 1); ctx.paint()
        kicker(ctx, M, 26, "Materiales y lectura de las hojas")
        font(ctx, DIDOT, 15); put(ctx, M, 42, "Antes de montar", INK)
        y = para(ctx, M, 54, W - 2 * M,
                 f"El modelo está desplegado en {npcs} piezas. Se recortan, se pliegan y se pegan en el orden del paso "
                 f"a paso hasta formar un busto de {a.alto:g} cm. No hace falta experiencia previa: conviene marcar bien "
                 "los pliegues y poner poca cola.", ITAL, 4.0, 1.55, INK, italic=True)
        rule(ctx, M, y + 1, W - M)
        colw = (W - 2 * M - 12) / 2
        # columna izquierda: materiales
        x, yy = M, y + 12
        kicker(ctx, x, yy, "Qué necesitas"); yy += 8
        items = [("Las hojas de piezas", "impresas a tamaño real, en papel de 160–200 g/m² (cartulina fina)."),
                 ("Tijeras pequeñas o cúter", "y una base de corte para las líneas rectas."),
                 ("Regla metálica", "para cortar y para marcar pliegues."),
                 ("Bolígrafo sin tinta", "o un punzón: se marca el pliegue antes de doblar."),
                 ("Cola blanca y un palillo", "para repartirla. Poca: el papel se ondula si se moja."),
                 ("Pinzas o clips", "para sujetar mientras seca.")]
        for t, s in items:
            ctx.arc(x + 1.2, yy - 1.2, 0.9, 0, 2 * math.pi); ctx.set_source_rgb(*ACCENT); ctx.fill()
            font(ctx, SANS, 3.4, bold=True); put(ctx, x + 4.5, yy, t, INK)
            yy = para(ctx, x + 4.5, yy + 4.6, colw - 4.5, s, SANS, 3.2, 1.45, MUTED) + 2.5
        ycol = yy
        # columna derecha: leyenda
        x, yy = M + colw + 12, y + 12
        kicker(ctx, x, yy, "Cómo leer las hojas"); yy += 10
        leg = [("corte", "Línea continua", "Corta por aquí."),
               ("monte", "Rayas", "Pliegue en monte: la arista queda hacia fuera."),
               ("valle", "Raya y punto", "Pliegue en valle: la arista queda hacia dentro."),
               ("pestaña", "Pestaña numerada", "Va por debajo de la arista con su mismo número, impreso fuera del recorte."),
               ("pieza", "Número de pieza", "Círculo junto al recorte. Coincide con el mapa y el paso a paso.")]
        for kind, t, s in leg:
            legend_sample(ctx, x, yy - 1.2, kind)
            font(ctx, SANS, 3.4, bold=True); put(ctx, x + 21, yy, t, INK)
            yy = para(ctx, x + 21, yy + 4.6, colw - 21, s, SANS, 3.2, 1.45, MUTED) + 4
        y = max(ycol, yy) + 4
        rule(ctx, M, y, W - M)
        # trucos
        y += 11
        kicker(ctx, M, y, "Trucos del oficio"); y += 8
        tips = ["Marca todos los pliegues con la regla antes de recortar: salen mucho más limpios.",
                "Dobla cada pieza entera antes de pegar nada; así ves cómo cierra.",
                "Pega de dos en dos pestañas y sujeta unos segundos. Sin prisa.",
                "Si una pestaña no entra, recórtala un poco: por dentro no se ve.",
                "La última pieza cierra el modelo: pégala apretando desde fuera con cuidado."]
        for i, t in enumerate(tips, 1):
            font(ctx, DIDOT, 7); put(ctx, M, y + 1.5, str(i), ACCENT)
            y = para(ctx, M + 9, y, W - 2 * M - 9, t, SANS, 3.5, 1.45, INK) + 2.2
        # en grupo
        y += 5
        bh = 37
        ctx.rectangle(M, y, W - 2 * M, bh); ctx.set_source_rgb(*CREAM); ctx.fill()
        kicker(ctx, M + 8, y + 10, "Para montar en grupo")
        para(ctx, M + 8, y + 18, W - 2 * M - 16,
             "Repartid el trabajo por hojas: cada uno recorta y marca los pliegues de las suyas. Después, una persona "
             "va pegando en el orden del paso a paso mientras el resto le pasa las piezas ya dobladas. El mapa de piezas "
             "ayuda a saber dónde va cada una.", ITAL, 3.7, 1.5, INK, italic=True)

    # ------------------------------------------------ 3. mapa
    def mapa_page(ctx):
        ctx.set_source_rgb(1, 1, 1); ctx.paint()
        kicker(ctx, M, 26, "Mapa de piezas")
        font(ctx, DIDOT, 15); put(ctx, M, 42, "Cada color, una pieza", INK)
        para(ctx, M, 51, W - 2 * M, "El número es el mismo que verás en las hojas. A la izquierda, de frente; "
             "a la derecha, por detrás.", ITAL, 4.0, 1.5, MUTED, italic=True)
        colw = (W - 2 * M - 8) / 2
        for j, nm in enumerate(("frente", "atras")):
            bx = M + j * (colw + 8)
            x, y, w, h, s = png(ctx, kd / f"mapa_{nm}_c.png", 0, 0, box=(bx, 60, colw, 120))
            x0, y0 = boxes[nm][0], boxes[nm][1]
            full = Image.open(kd / f"mapa_{nm}.png").size
            al = np.asarray(Image.open(kd / f"mapa_{nm}_c.png").convert("RGBA"))[..., 3] > 40
            _, (iy, ix) = ndi.distance_transform_edt(~al, return_indices=True)
            for k, (u, v, tot) in mapa["labels"][nm].items():
                px, py = u * full[0] - x0, v * full[1] - y0
                cy_, cx_ = int(np.clip(py, 0, al.shape[0] - 1)), int(np.clip(px, 0, al.shape[1] - 1))
                if not al[cy_, cx_]:  # la etiqueta cayó fuera del dibujo: llévala al punto más cercano de la pieza
                    cy_, cx_ = iy[cy_, cx_], ix[cy_, cx_]
                    px, py = cx_, cy_
                circle_num(ctx, x + px * s, y + py * s, 2.6, str(k), fill=(1, 1, 1), txt=INK, size=2.9, stroke=INK)
            font(ctx, SANS, 2.6, bold=True)
            put(ctx, bx + colw / 2, 186, ("de frente" if nm == "frente" else "por detrás").upper(), MUTED,
                align="center", tracking=0.7)
        # índice de piezas: color, número, hoja
        y = 200
        rule(ctx, M, y - 6, W - M)
        kicker(ctx, M, y + 2, "Dónde está cada pieza"); y += 10
        cols = 4
        cw = (W - 2 * M) / cols
        for i in range(npcs):
            p = i + 1
            cx, cy = M + (i % cols) * cw, y + (i // cols) * 9
            c = [x / 255 for x in PALETTE[i % len(PALETTE)]]
            ctx.rectangle(cx, cy - 3.6, 5, 5); ctx.set_source_rgb(*c); ctx.fill()
            font(ctx, SANS, 3.4, bold=True); put(ctx, cx + 8, cy, f"Pieza {p}", INK)
            font(ctx, SANS, 3.0); put(ctx, cx + 26, cy, f"hoja {sheet_of[p]}", MUTED)

    # ------------------------------------------------ 4. paso a paso
    def steps_pages():
        per = 12
        out = []
        for s0 in range(0, len(steps), per):
            out.append(list(range(s0, min(len(steps), s0 + per))))
        return out

    step_pages = steps_pages()

    def steps_page(ctx, idxs, first):
        ctx.set_source_rgb(1, 1, 1); ctx.paint()
        kicker(ctx, M, 26, "Paso a paso")
        font(ctx, DIDOT, 15); put(ctx, M, 42, "Montaje" if first else "Montaje, continuación", INK)
        para(ctx, M, 51, W - 2 * M, "Del hocico hacia atrás. En naranja, la pieza que toca; lo demás, lo que ya "
             "tienes montado.", ITAL, 4.0, 1.5, MUTED, italic=True)
        cols, rows = 3, 4
        cw, ch = (W - 2 * M) / cols, (H - 64 - 16) / rows
        done = set()
        for st in steps[:idxs[0]]:
            done |= set(st)
        for j, si in enumerate(idxs):
            st = steps[si]
            cx, cy = M + (j % cols) * cw, 64 + (j // cols) * ch
            png(ctx, kd / f"paso_{si + 1:02d}_c.png", 0, 0, box=(cx + 2, cy + 2, cw - 4, ch - 22))
            circle_num(ctx, cx + 5, cy + 5, 3.6, str(si + 1))
            who, lines = step_text(si + 1, st, links, done)
            font(ctx, SANS, 3.3, bold=True); put(ctx, cx + 2, cy + ch - 15, who, INK)
            yy = cy + ch - 10.5
            for ln in lines:
                yy = para(ctx, cx + 2, yy, cw - 6, ln, SANS, 2.7, 1.35, MUTED)
            done |= set(st)

    # ------------------------------------------------ 5. hojas de piezas
    def tf(rot, o, dst):
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        R = np.array([[c, -s], [s, c]])
        return lambda X: (np.asarray(X) @ R.T) - np.array(o) + np.array(dst)

    sources = {n: (cairo.ImageSurface.create_from_png(str(d / f)), cairo.EXTEND_PAD) for n, f in texfiles.items()}

    def small_text(ctx, p, angle, s, size, rgb, alpha=1.0, bold=False):
        ctx.save()
        font(ctx, SANS, size, bold=bold)
        e = ctx.text_extents(s)
        ctx.translate(*p)
        if math.cos(angle) < 0:
            angle += math.pi
        ctx.rotate(angle)
        ctx.move_to(-e.width / 2 - e.x_bearing, -e.height / 2 - e.y_bearing)
        ctx.set_source_rgba(*rgb, alpha); ctx.show_text(s)
        ctx.restore()

    from shapely.geometry import Polygon as SPoly, Point as SPoint, box as sbox
    from shapely.ops import unary_union as sunion
    from shapely.prepared import prep as sprep
    from shapely import affinity as saff

    # lado sin pestaña de cada arista cortada: ahí va el número (fuera del recorte)
    isl_of = L["isl_of"]
    edge_marks = {k: [] for k in islands}
    for num, e in enumerate(cut, 1):
        sides = []
        for fa in edges[e]:
            k = isl_of[fa]
            P = islands[k]["faces"][fa]; loc = list(F[fa])
            p, q = P[loc.index(e[0])], P[loc.index(e[1])]
            third = P[[i for i in range(3) if loc[i] not in e][0]]
            has_tab = any(n == num and ((np.allclose(t[0], p) and np.allclose(t[1], q)) or
                                        (np.allclose(t[0], q) and np.allclose(t[1], p))) for t, n in tabs[k])
            sides.append((k, p, q, third, has_tab))
        for k, p, q, third, _ in ([sd for sd in sides if not sd[4]] or sides):
            edge_marks[k].append((p, q, third, num))
    unplaced = [0]
    why = []

    def margin_marks(items):
        """Posiciones (en mm de página) de números de arista y círculos de pieza, sin tocar nada impreso."""
        shapes, per_isl = [], {}
        for k, rot, o, dst in items:
            T = tf(rot, o, dst)
            iso = islands[k]["poly"]
            gs = [iso] if iso.geom_type == "Polygon" else list(iso.geoms)
            ip = sunion([SPoly(T(np.array(g.exterior.coords))) for g in gs])
            per_isl[k] = (T, ip)
            shapes.append(ip)
        solid = sunion(shapes).buffer(0.05)   # solo la parte visible: las pestañas quedan ocultas al pegar
        solid_p = sprep(solid)
        area = sbox(MARGIN - 3, MARGIN + HEAD - 3, W - MARGIN + 3, H - MARGIN - FOOT + 3)
        taken = []
        out = []
        for k, rot, o, dst in items:  # el número de cada pestaña se respeta
            T = per_isl[k][0]
            for t, _ in tabs[k]:
                taken.append(SPoint(*T(t).mean(0)).buffer(1.9))

        def free(geom):
            return area.contains(geom) and not solid_p.intersects(geom) and not any(geom.intersects(t) for t in taken)

        # círculos de pieza: el hueco libre más cercano al recorte (y al centro de la pieza)
        for k, rot, o, dst in items:
            T, ip = per_isl[k]
            cen = ip.representative_point()
            x0, y0, x1, y1 = ip.bounds
            best = None
            for gx in np.arange(x0 - 14, x1 + 14, 1.5):
                for gy in np.arange(y0 - 14, y1 + 14, 1.5):
                    c = SPoint(gx, gy).buffer(3.4)
                    if not free(c):
                        continue
                    dist = ip.distance(SPoint(gx, gy))
                    sc = dist + 0.12 * SPoint(gx, gy).distance(cen)
                    if best is None or sc < best[0]:
                        best = (sc, gx, gy)
            if best:
                taken.append(SPoint(best[1], best[2]).buffer(3.4))
                out.append(("pieza", (best[1], best[2]), 0.0, str(pn[k])))
        # números de arista, junto a la arista y por fuera
        for k, rot, o, dst in items:
            T, ip = per_isl[k]
            for p, q, third, num in edge_marks[k]:
                pp, qq, tt = T(np.array([p]))[0], T(np.array([q]))[0], T(np.array([third]))[0]
                mid = (pp + qq) / 2
                dv = qq - pp
                Ln = float(np.linalg.norm(dv))
                u = dv / Ln
                nrm = np.array([-u[1], u[0]])
                if nrm @ (tt - mid) > 0:
                    nrm = -nrm
                ang = math.atan2(u[1], u[0])
                w, h = 1.1 * len(str(num)) + 0.6, 1.8
                done = False
                for off in (1.3, 1.8, 2.4, 3.2, 4.2, 5.4, 7.0):
                    for sh in (0.0, 0.12, -0.12, 0.24, -0.24, 0.36, -0.36, 0.46, -0.46):
                        cpt = mid + u * (sh * Ln) + nrm * off
                        r = saff.rotate(sbox(-w / 2, -h / 2, w / 2, h / 2), math.degrees(ang), origin=(0, 0))
                        r = saff.translate(r, cpt[0], cpt[1])
                        if free(r):
                            taken.append(r); out.append(("arista", tuple(cpt), ang, str(num))); done = True
                            break
                    if done:
                        break
                if not done:  # rendijas: en la prolongación de la arista, en la boca del corte
                    for end, sgn in ((pp, -1), (qq, 1)):
                        for dist in (2.0, 3.0, 4.5, 6.5, 9.0, 12.0, 16.0, 21.0):
                            for lat in (0.0, 0.8, -0.8, 1.6, -1.6, 2.6, -2.6):
                                cpt = end + u * sgn * dist + nrm * lat
                                r = saff.rotate(sbox(-w / 2, -h / 2, w / 2, h / 2), math.degrees(ang), origin=(0, 0))
                                r = saff.translate(r, cpt[0], cpt[1])
                                if free(r):
                                    taken.append(r); out.append(("arista", tuple(cpt), ang, str(num))); done = True
                                    break
                            if done:
                                break
                        if done:
                            break
                if not done:
                    unplaced[0] += 1
                    why.append((num, round(Ln, 1), k))
        return out

    def sheet(ctx, items, pageno):
        ctx.set_source_rgb(1, 1, 1); ctx.paint()
        # cabecera
        kicker(ctx, MARGIN, MARGIN + 4, f"{a.nombre} · papercraft", INK, size=2.4)
        font(ctx, SANS, 2.4, bold=True)
        put(ctx, W - MARGIN, MARGIN + 4, f"HOJA {pageno} DE {len(pages)}", MUTED, align="right", tracking=0.6)
        rule(ctx, MARGIN, MARGIN + 7, W - MARGIN, RULE, 0.25)
        for k, rot, o, dst in items:
            T = tf(rot, o, dst)
            isl = islands[k]
            for t, num in tabs[k]:
                P = T(t)
                ctx.move_to(*P[0]); [ctx.line_to(*p) for p in P[1:]]; ctx.close_path()
                ctx.set_source_rgb(*TABFILL); ctx.fill_preserve()
                ctx.set_source_rgb(*INK); ctx.set_line_width(0.25); ctx.stroke()
                m = P.mean(0)
                small_text(ctx, m, math.atan2(P[1][1] - P[0][1], P[1][0] - P[0][0]), str(num), 2.3, MUTED, bold=True)
            for fi, P in isl["faces"].items():
                P = T(P)
                tex, uv = src[fi]
                surf, ext = sources[tex]
                c = P.mean(0)
                Q = c + (P - c) * (1 + 0.4 / max(1e-6, np.linalg.norm(P - c, axis=1).min()))
                ctx.save()
                ctx.move_to(*Q[0]); ctx.line_to(*Q[1]); ctx.line_to(*Q[2]); ctx.close_path(); ctx.clip()
                pat = cairo.SurfacePattern(surf); pat.set_extend(ext); pat.set_filter(cairo.FILTER_GOOD)
                pat.set_matrix(affine_to(P, uv)); ctx.set_source(pat); ctx.paint()
                ctx.restore()
            for fi, P in isl["faces"].items():
                loc = list(F[fi]); PP = T(P)
                for i in range(3):
                    e = (min(loc[i], loc[(i + 1) % 3]), max(loc[i], loc[(i + 1) % 3]))
                    if e in joined and edges[e][0] == fi:
                        ctx.set_source_rgba(*INK, 0.38); ctx.set_line_width(0.16)
                        ctx.set_dash([1.6, 1.0] if info[e][1] else [2.2, 0.8, 0.4, 0.8])
                        ctx.move_to(*PP[i]); ctx.line_to(*PP[(i + 1) % 3]); ctx.stroke()
            ctx.set_dash([])
            outline = isl["poly"]
            for ring in [outline.exterior] if outline.geom_type == "Polygon" else [g.exterior for g in outline.geoms]:
                P = T(np.array(ring.coords))
                ctx.move_to(*P[0]); [ctx.line_to(*p) for p in P[1:]]; ctx.close_path()
                ctx.set_source_rgb(*INK); ctx.set_line_width(0.3); ctx.stroke()
        # números de arista y de pieza: SIEMPRE fuera del recorte (nada impreso sobre la textura visible)
        for kind, pos, ang, txt in margin_marks(items):
            if kind == "pieza":
                circle_num(ctx, pos[0], pos[1], 2.8, txt)
            else:
                small_text(ctx, pos, ang, txt, 2.0, INK, alpha=0.8, bold=True)
        # pie: leyenda + barra de escala
        y = H - MARGIN - 5
        rule(ctx, MARGIN, y - 6, W - MARGIN, RULE, 0.25)
        x = MARGIN
        for kind, t in (("corte", "corte"), ("monte", "monte"), ("valle", "valle")):
            legend_sample(ctx, x, y - 1, kind)
            font(ctx, SANS, 2.5); put(ctx, x + 18, y, t, MUTED); x += 32
        font(ctx, SANS, 2.5); put(ctx, x, y, "la pestaña N va bajo la arista N", MUTED)
        bx = W - MARGIN - 50
        ctx.set_source_rgb(*INK); ctx.set_line_width(0.3)
        ctx.move_to(bx, y - 1); ctx.line_to(bx + 50, y - 1); ctx.stroke()
        for t in range(0, 51, 10):
            ctx.move_to(bx + t, y - 1); ctx.line_to(bx + t, y - (3 if t % 50 == 0 else 2)); ctx.stroke()
        font(ctx, SANS, 2.2); put(ctx, bx + 25, y + 3, "50 mm · si no mide esto, imprime al 100 %", MUTED, align="center")

    # ------------------------------------------------ 6. contraportada
    def back(ctx):
        ctx.set_source_rgb(*CREAM); ctx.paint()
        kicker(ctx, W / 2, 30, f"{a.nombre} · vistas del modelo montado", align="center")
        vu = sorted((d / "vuelta").glob("*.png"))
        if vu:
            sel = [vu[int(len(vu) * f) % len(vu)] for f in (0.0, 0.33, 0.58, 0.83)]
            cw = (W - 2 * M - 6) / 2
            ch = cw * 840 / 720
            for i, pth in enumerate(sel):
                png(ctx, pth, 0, 0, box=(M + (i % 2) * (cw + 6), 42 + (i // 2) * (ch + 6), cw, ch))
        font(ctx, SANS, 2.2); put(ctx, W / 2, 287, a.credito, MUTED, align="center")

    # ------------------------------------------------ composición
    pt = 72 / 25.4
    order = [cover, intro, mapa_page] + [("steps", i) for i in range(len(step_pages))] + \
            [("sheet", i) for i in range(len(pages))] + [back]

    def draw(ctx, item):
        if callable(item):
            item(ctx)
        elif item[0] == "steps":
            steps_page(ctx, step_pages[item[1]], item[1] == 0)
        else:
            sheet(ctx, pages[item[1]], item[1] + 1)

    out = d / f"{a.nombre.lower()}_kit.pdf"
    pdf = cairo.PDFSurface(str(out), W * pt, H * pt)
    pdf.set_metadata(cairo.PDF_METADATA_TITLE, f"{a.nombre} · papercraft")
    ctx = cairo.Context(pdf); ctx.scale(pt, pt)
    for it in order:
        draw(ctx, it); pdf.show_page()
    pdf.finish()
    (kd / "pags").mkdir(exist_ok=True)
    for f in (kd / "pags").glob("*.png"):
        f.unlink()
    px = 70 / 25.4
    for i, it in enumerate(order, 1):
        img = cairo.ImageSurface(cairo.FORMAT_RGB24, int(W * px), int(H * px))
        c = cairo.Context(img); c.scale(px, px); draw(c, it)
        img.write_to_png(str(kd / "pags" / f"{i:02d}.png"))
    print("SIN HUECO (num, largo mm, isla):", sorted(set(why)))
    print(f"aristas sin hueco para su número (cuenta doble: PDF + vistas): {unplaced[0]}")
    print(f"KIT {out} · {len(order)} páginas · {npcs} piezas · {len(steps)} pasos · {len(cut)} pestañas")


if __name__ == "__main__":
    main()
