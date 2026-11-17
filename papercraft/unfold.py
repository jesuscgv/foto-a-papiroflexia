"""Paso 3: malla low-poly + texturas -> piezas desplegadas con pestañas -> PDF A4 imprimible.

python unfold.py carpeta --alto 18
Lee low.obj, photo_clean.png, fur.png, meta.json. Escribe plantilla.pdf, paginas/*.png y modelo.obj/mtl.
"""
import argparse, json, math
from pathlib import Path

import cairo
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity

A4 = (210.0, 297.0)
MARGIN = 9.0
TAB_H = 5.0
FRONT_NZ = 0.3  # caras con normal z mayor usan la foto; el resto, pelo


# ---------------------------------------------------------------- malla
def load_obj(p):
    """Devuelve vértices, caras y (si el OBJ trae UV) las UV de cada esquina de cada cara."""
    V, F, VT, FT = [], [], [], []
    for line in open(p):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "vt":
            VT.append([float(x) for x in t[1:3]])
        elif t[0] == "f":
            idx = [x.split("/") for x in t[1:4]]
            F.append([int(i[0]) - 1 for i in idx])
            if len(idx[0]) > 1 and idx[0][1]:
                FT.append([int(i[1]) - 1 for i in idx])
    V, F = np.array(V), np.array(F)
    UV = np.array(VT)[np.array(FT)] if FT else None  # (caras, 3, 2)
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()
    if vol < 0:
        F = F[:, ::-1]
        if UV is not None:
            UV = UV[:, ::-1]
    return V, F, UV


def tri2d(P):
    """Triángulo 3D -> 2D conservando longitudes, en sentido antihorario (visto desde fuera)."""
    a, b, c = P
    ab = np.linalg.norm(b - a)
    e = (b - a) / ab
    ac = c - a
    x = ac @ e
    y = np.linalg.norm(ac - x * e)
    return np.array([[0, 0], [ab, 0], [x, y]])


def rigid(q0, q1, p0, p1):
    """Rotación + traslación 2D que lleva q0->p0, q1->p1."""
    aq = math.atan2(*(q1 - q0)[::-1])
    ap = math.atan2(*(p1 - p0)[::-1])
    t = ap - aq
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return lambda X: (X - q0) @ R.T + p0


# ---------------------------------------------------------------- texturas
def face_source(V, F, normals, W, H):
    """Para cada cara: (textura, coords 2D de sus 3 vértices en píxeles de esa textura)."""
    out = []
    xc = (V[:, 0].min() + V[:, 0].max()) / 2
    for f, n in zip(F, normals):
        P = V[f]
        if n[2] > FRONT_NZ:
            out.append(("photo", np.stack([P[:, 0], -P[:, 1]], 1)))
        else:
            # proyección continua para espalda, lados, arriba y base: (x, y) desplazados según la
            # profundidad; así caras vecinas comparten textura sin costuras y ninguna degenera
            uv = np.stack([P[:, 0] - 0.6 * P[:, 2] * np.sign(P[:, 0].mean() - xc), -P[:, 1] + 0.6 * P[:, 2]], 1)
            area3 = np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
            d1, d2 = uv[1] - uv[0], uv[2] - uv[0]
            area2 = abs(d1[0] * d2[1] - d1[1] * d2[0])
            if area2 < 0.15 * area3:  # caso raro: plana local centrada en la cara
                c = P.mean(0)
                e2 = np.array([0.0, 1.0, 0.0]) - n[1] * n
                if np.linalg.norm(e2) < 0.3:
                    e2 = np.array([0.0, 0.0, 1.0]) - n[2] * n
                e2 /= np.linalg.norm(e2)
                e1 = np.cross(e2, n)
                d = P - c
                uv = np.stack([c[0] + d @ e1, -(c[1] + d @ e2)], 1)
            out.append(("fur", uv))
    return out


def multiview_source(V, F, normals, meta):
    """Cada cara coge la vista que mejor la ve: frente (foto real deformada), perfil o espalda.
    Caras muy rasantes (arriba de la cabeza, base) usan la espalda desplazada según la profundidad."""
    fy, sx = np.array(meta["fy"]), np.array(meta["sx"])
    rows = np.arange(len(fy))
    cx, cz, axf = meta["cx"], meta["cz"], meta["eje_frente"]
    proj = {
        "frente": lambda P: np.stack([axf + P[:, 0] * np.interp(-P[:, 1], rows, sx), np.interp(-P[:, 1], rows, fy)], 1),
        "perfil": lambda P: np.stack([cz - P[:, 2], -P[:, 1]], 1),
        "atras": lambda P: np.stack([cx - P[:, 0], -P[:, 1]], 1),
    }
    out = []
    for f, n in zip(F, normals):
        P = V[f]
        score = {"frente": n[2] + 0.45, "atras": -n[2], "perfil": abs(n[0])}
        best = max(score, key=score.get)
        if score[best] >= 0.4:
            out.append((best, proj[best](P)))
        else:  # rasante: arriba o abajo
            k = -0.6 if n[1] > 0 else 0.6
            out.append(("atras", np.stack([cx - P[:, 0], -P[:, 1] + k * P[:, 2]], 1)))
    return out


def affine_to(dst, src):
    """Matriz cairo que lleva puntos dst (página) -> src (textura)."""
    A = np.hstack([dst, np.ones((3, 1))])
    M = np.linalg.solve(A, src)  # 3x2
    return cairo.Matrix(M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[2, 0], M[2, 1])


# ---------------------------------------------------------------- despliegue
def unfold(V, F, scale, max_w, max_h, compact=0.5, group=None):
    nF = len(F)
    edges = {}
    for fi, f in enumerate(F):
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            edges.setdefault((min(a, b), max(a, b)), []).append(fi)
    normals = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    def edge_info(e):
        f1, f2 = edges[e]
        ang = math.acos(np.clip(normals[f1] @ normals[f2], -1, 1))
        # convexo (monte) si el tercer vértice de f2 queda por debajo del plano de f1
        c2 = [v for v in F[f2] if v not in e][0]
        convex = (V[c2] - V[e[0]]) @ normals[f1] < 0
        return ang, convex

    info = {e: edge_info(e) for e in edges}
    if group is None:
        group = normals[:, 2] > FRONT_NZ  # foto / pelo
    # islas
    isl_of = list(range(nF))
    islands = {i: {"faces": {i: tri2d(V[F[i]] * scale)}, "poly": None} for i in range(nF)}
    for i in islands:
        islands[i]["poly"] = Polygon(islands[i]["faces"][i])

    def pos(isl, fi, v):
        return islands[isl]["faces"][fi][list(F[fi]).index(v)]

    def fits(poly):
        r = poly.minimum_rotated_rectangle
        xs, ys = r.exterior.coords.xy
        s = sorted([math.dist((xs[0], ys[0]), (xs[1], ys[1])), math.dist((xs[1], ys[1]), (xs[2], ys[2]))])
        return s[0] <= min(max_w, max_h) and s[1] <= max(max_w, max_h)

    joined = set()
    # prioridad: pliegues suaves primero, y aristas largas antes que cortas
    # y antes las que no cruzan de foto a pelo (así la cara queda en pocas piezas)
    order = sorted(edges, key=lambda e: info[e][0] + 2.0 * (group[edges[e][0]] != group[edges[e][1]])
                   - 1e-4 * np.linalg.norm(V[e[0]] - V[e[1]]))
    for e in order:
        f1, f2 = edges[e]
        i1, i2 = isl_of[f1], isl_of[f2]
        if i1 == i2:
            continue
        a, b = e
        T = rigid(pos(i2, f2, a), pos(i2, f2, b), pos(i1, f1, a), pos(i1, f1, b))
        moved = {fi: T(P) for fi, P in islands[i2]["faces"].items()}
        poly2 = unary_union([Polygon(P) for P in moved.values()])
        if islands[i1]["poly"].buffer(-0.05).intersects(poly2.buffer(-0.05)):
            continue
        merged = unary_union([islands[i1]["poly"], poly2])
        if not fits(merged):
            continue
        # piezas compactas: nada de tiras largas y ramificadas, que son un infierno de montar
        if len(islands[i1]["faces"]) + len(moved) > 6 and merged.area / merged.convex_hull.area < compact:
            continue
        islands[i1]["faces"].update(moved)
        islands[i1]["poly"] = merged
        for fi in moved:
            isl_of[fi] = i1
        del islands[i2]
        joined.add(e)
    return edges, info, joined, isl_of, islands, normals


def make_tab(p, q, inward, h):
    """Trapecio sobre la arista p-q, hacia el lado contrario a 'inward'."""
    d = q - p
    L = np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) / L
    if n @ (inward - p) > 0:
        n = -n
    h = min(h, 0.45 * L)
    ins = min(h, 0.3 * L)
    u = d / L
    return np.array([p, q, q + n * h - u * ins, p + n * h + u * ins])


# ---------------------------------------------------------------- main
def layout(d, alto, compacto, head=0.0, foot=0.0, gap=3.0):
    """Despliega y reparte en hojas. Devuelve todo lo necesario para dibujar."""
    d = Path(d)
    meta = json.load(open(d / "meta.json"))
    V, F, UV = load_obj(d / "low.obj")
    scale = alto * 10 / (V[:, 1].max() - V[:, 1].min())  # mm por unidad
    pw, ph = A4[0] - 2 * MARGIN, A4[1] - 2 * MARGIN - head - foot
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if meta.get("baked"):
        src = [("baked", np.stack([uv[:, 0] * meta["W"], (1 - uv[:, 1]) * meta["H"]], 1)) for uv in UV]
    elif meta.get("multiview"):
        src = multiview_source(V, F, nrm, meta)
    else:
        src = face_source(V, F, nrm, meta["W"], meta["H"])
    texfiles = meta.get("textures", {"photo": "photo_clean.png", "fur": "fur.png"})
    names = list(texfiles)
    group = np.array([names.index(t) for t, _ in src])
    edges, info, joined, isl_of, islands, normals = unfold(V, F, scale, pw - 2 * TAB_H, ph - 2 * TAB_H, compacto, group)

    # orden y nombre de piezas: de mayor a menor
    keys = sorted(islands, key=lambda k: -islands[k]["poly"].area)
    piece_name = {k: i + 1 for i, k in enumerate(keys)}

    # --- pestañas y numeración de aristas cortadas
    cut = [e for e in edges if e not in joined]
    cut.sort(key=lambda e: (piece_name[isl_of[edges[e][0]]], piece_name[isl_of[edges[e][1]]]))
    tabs = {k: [] for k in keys}        # isla -> [(poly, num)]
    labels = {k: [] for k in keys}      # isla -> [(pos, ángulo, num)]
    occupied = {k: islands[k]["poly"].buffer(0.01) for k in keys}
    for num, e in enumerate(cut, 1):
        f1, f2 = edges[e]
        placed = False
        for fa in (f1, f2) if f1 < f2 else (f2, f1):
            k = isl_of[fa]
            P = islands[k]["faces"][fa]
            loc = list(F[fa])
            p, q = P[loc.index(e[0])], P[loc.index(e[1])]
            third = P[[i for i in range(3) if loc[i] not in e][0]]
            for h in (TAB_H, TAB_H * 0.6):
                t = make_tab(p, q, third, h)
                tp = Polygon(t)
                if not tp.buffer(-0.05).intersects(occupied[k]):
                    tabs[k].append((t, num))
                    occupied[k] = occupied[k].union(tp)
                    placed = True
                    break
            if placed:
                break
        for fa in (f1, f2):
            k = isl_of[fa]
            P = islands[k]["faces"][fa]
            loc = list(F[fa])
            p, q = P[loc.index(e[0])], P[loc.index(e[1])]
            third = P[[i for i in range(3) if loc[i] not in e][0]]
            mid = (p + q) / 2
            inn = third - mid
            inn /= np.linalg.norm(inn)
            labels[k].append((mid + inn * 2.2, math.atan2(q[1] - p[1], q[0] - p[0]), num))

    # --- colocar piezas en páginas (estanterías), girando a su rectángulo mínimo
    placed_islands = []
    for k in keys:
        full = unary_union([islands[k]["poly"]] + [Polygon(t) for t, _ in tabs[k]])
        r = full.minimum_rotated_rectangle
        xs, ys = r.exterior.coords.xy
        ang = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0]))
        best = None
        for extra in (0, 90):
            g = affinity.rotate(full, -ang + extra, origin=(0, 0))
            x0, y0, x1, y1 = g.bounds
            if x1 - x0 <= pw and y1 - y0 <= ph and (best is None or y1 - y0 < best[2]):
                best = (-ang + extra, (x0, y0), y1 - y0, x1 - x0)
        if best is None:  # no debería pasar (fits() lo impide)
            best = (-ang, full.bounds[:2], full.bounds[3] - full.bounds[1], full.bounds[2] - full.bounds[0])
        placed_islands.append([k, best[0], best[1], best[3], best[2]])
    placed_islands.sort(key=lambda t: -t[4])
    pages, cur, x, y, shelf = [], [], 0.0, 0.0, 0.0
    for k, rot, (x0, y0), w, h in placed_islands:
        if x + w > pw:
            x, y, shelf = 0.0, y + shelf + gap, 0.0
        if y + h > ph:
            pages.append(cur); cur, x, y, shelf = [], 0.0, 0.0, 0.0
        cur.append((k, rot, (x0, y0), (MARGIN + x, MARGIN + head + y)))
        x += w + gap
        shelf = max(shelf, h)
    if cur:
        pages.append(cur)
    return dict(locals())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--alto", type=float, default=18.0, help="altura del modelo montado en cm")
    ap.add_argument("--compacto", type=float, default=0.5, help="0-1: más alto = piezas más redondas pero más piezas")
    ap.add_argument("--portada", help="imagen de vista previa 3D para la primera hoja")
    ap.add_argument("--dpi", type=int, default=110, help="resolución de las vistas previas PNG")
    a = ap.parse_args()
    d = Path(a.dir)
    L = layout(d, a.alto, a.compacto)
    (V, F, src, texfiles, names, edges, info, joined, isl_of, islands, keys, piece_name, cut, tabs, labels,
     pages, scale) = (L[k] for k in "V F src texfiles names edges info joined isl_of islands keys piece_name cut tabs labels pages scale".split())

    def tf(rot, o, dst):
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        R = np.array([[c, -s], [s, c]])
        return lambda X: (np.asarray(X) @ R.T) - np.array(o) + np.array(dst)

    sources = {n: (cairo.ImageSurface.create_from_png(str(d / f)), cairo.EXTEND_PAD) for n, f in texfiles.items()}

    def draw_page(ctx, items, pageno):
        ctx.set_source_rgb(1, 1, 1); ctx.paint()
        for k, rot, o, dst in items:
            T = tf(rot, o, dst)
            isl = islands[k]
            # pestañas
            for t, num in tabs[k]:
                P = T(t)
                ctx.move_to(*P[0]); [ctx.line_to(*p) for p in P[1:]]; ctx.close_path()
                ctx.set_source_rgb(0.93, 0.93, 0.93); ctx.fill_preserve()
                ctx.set_source_rgb(0.15, 0.15, 0.15); ctx.set_line_width(0.25); ctx.stroke()
                m = (P[0] + P[1] + P[2] + P[3]) / 4
                text(ctx, m, math.atan2(P[1][1] - P[0][1], P[1][0] - P[0][0]), str(num), 2.6, (0.3, 0.3, 0.3))
            # textura por cara (con 0.4 mm de sangrado para que no queden huecos blancos)
            for fi, P in isl["faces"].items():
                P = T(P)
                tex, uv = src[fi]
                surf, ext = sources[tex]
                c = P.mean(0)
                Q = c + (P - c) * (1 + 0.4 / max(1e-6, np.linalg.norm(P - c, axis=1).min()))
                ctx.save()
                ctx.move_to(*Q[0]); ctx.line_to(*Q[1]); ctx.line_to(*Q[2]); ctx.close_path(); ctx.clip()
                pat = cairo.SurfacePattern(surf)
                pat.set_extend(ext); pat.set_filter(cairo.FILTER_GOOD)
                pat.set_matrix(affine_to(P, uv))
                ctx.set_source(pat); ctx.paint()
                ctx.restore()
            # pliegues: monte = rayas, valle = raya-punto (finos y semitransparentes)
            for fi, P in isl["faces"].items():
                loc = list(F[fi])
                PP = T(P)
                for i in range(3):
                    e = (min(loc[i], loc[(i + 1) % 3]), max(loc[i], loc[(i + 1) % 3]))
                    if e in joined and edges[e][0] == fi:
                        convex = info[e][1]
                        ctx.set_source_rgba(1, 1, 1, 0.55)
                        ctx.set_line_width(0.18)
                        ctx.set_dash([1.6, 1.0] if convex else [2.2, 0.8, 0.4, 0.8])
                        ctx.move_to(*PP[i]); ctx.line_to(*PP[(i + 1) % 3]); ctx.stroke()
            ctx.set_dash([])
            # contorno de corte
            outline = isl["poly"]
            for ring in [outline.exterior] if outline.geom_type == "Polygon" else [g.exterior for g in outline.geoms]:
                P = T(np.array(ring.coords))
                ctx.move_to(*P[0]); [ctx.line_to(*p) for p in P[1:]]; ctx.close_path()
                ctx.set_source_rgb(0, 0, 0); ctx.set_line_width(0.3); ctx.stroke()
            # números de arista y nombre de pieza
            for p, angle, num in labels[k]:
                pp = T(np.array([p]))[0]
                r = math.radians(rot)
                text(ctx, pp, angle + r, str(num), 2.0, (1, 1, 1), alpha=0.75)
            c = T(np.array(isl["poly"].representative_point().coords))[0]
            text(ctx, c, 0, f"P{piece_name[k]}", 3.2, (1, 1, 1), alpha=0.9, bold=True)
        ctx.set_source_rgb(0.5, 0.5, 0.5)
        ctx.select_font_face("Helvetica"); ctx.set_font_size(3)
        ctx.move_to(MARGIN, A4[1] - 4); ctx.show_text(f"Hoja {pageno}/{len(pages)} · corte: línea continua · monte: rayas · valle: raya-punto · pestaña N se pega bajo la arista N")
        # marca de agua discreta del autor de la herramienta
        ctx.select_font_face("Helvetica"); ctx.set_font_size(2.6); ctx.set_source_rgba(0.5, 0.5, 0.5, 0.8)
        cred = "foto → papiroflexia · @jesuscgv"
        ctx.move_to(A4[0] - MARGIN - ctx.text_extents(cred).width, A4[1] - 4); ctx.show_text(cred)

    def text(ctx, p, angle, s, size, rgb, alpha=1.0, bold=False):
        ctx.save()
        ctx.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(size)
        ext = ctx.text_extents(s)
        ctx.translate(*p)
        a = angle
        if math.cos(a) < 0:  # que el texto nunca quede bocabajo
            a += math.pi
        ctx.rotate(a)
        ctx.move_to(-ext.width / 2 - ext.x_bearing, -ext.height / 2 - ext.y_bearing)
        ctx.text_path(s)
        ctx.set_source_rgba(0, 0, 0, 0.6 * alpha); ctx.set_line_width(0.35); ctx.stroke_preserve()
        ctx.set_source_rgba(*rgb, alpha); ctx.fill()
        ctx.restore()

    def draw_cover(ctx, img_path, n_pieces, n_glue, n_pages):
        ctx.set_source_rgb(1, 1, 1); ctx.paint()
        ctx.set_source_rgb(0.1, 0.1, 0.1)
        ctx.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(9); ctx.move_to(MARGIN + 6, 30); ctx.show_text("Papercraft")
        ctx.select_font_face("Helvetica"); ctx.set_font_size(4.2); ctx.set_source_rgb(0.4, 0.4, 0.4)
        ctx.move_to(MARGIN + 6, 38)
        ctx.show_text(f"Altura montado {a.alto:g} cm · {n_pieces} piezas · {n_glue} pestañas · {n_pages} hojas")
        # crédito de la herramienta
        ctx.set_font_size(3.6); ctx.set_source_rgb(0.55, 0.55, 0.55)
        cred = "generado con foto → papiroflexia · @jesuscgv"
        ctx.move_to(A4[0] - MARGIN - 6 - ctx.text_extents(cred).width, 30); ctx.show_text(cred)
        im = cairo.ImageSurface.create_from_png(img_path)
        sw = (A4[0] - 2 * MARGIN) / im.get_width()
        ctx.save(); ctx.translate(MARGIN, 46); ctx.scale(sw, sw)
        ctx.set_source_surface(im, 0, 0); ctx.paint(); ctx.restore()
        y = 46 + im.get_height() * sw + 14
        steps = ["Imprime al 100 % (sin «ajustar a la página»), en papel de 160-200 g/m².",
                 "Recorta por la línea negra continua, pestañas incluidas.",
                 "Marca los pliegues con regla y un bolígrafo sin tinta. Rayas: monte (hacia fuera).",
                 "Raya-punto: valle (hacia dentro). Dobla todos los pliegues antes de pegar.",
                 "Cada pestaña con el número N se pega por debajo de la arista que lleva el mismo N.",
                 "Empieza por las piezas de la cara y cierra al final la base. Cola blanca o barra, poca."]
        ctx.set_font_size(4.4); ctx.set_source_rgb(0.15, 0.15, 0.15)
        for i, st in enumerate(steps, 1):
            ctx.move_to(MARGIN + 6, y); ctx.show_text(f"{i}.  {st}"); y += 8

    pt = 72 / 25.4
    pdf = cairo.PDFSurface(str(d / "plantilla.pdf"), A4[0] * pt, A4[1] * pt)
    ctx = cairo.Context(pdf); ctx.scale(pt, pt)
    if a.portada:
        draw_cover(ctx, a.portada, len(islands), len(cut), len(pages))
        pdf.show_page()
    for i, items in enumerate(pages, 1):
        draw_page(ctx, items, i); pdf.show_page()
    pdf.finish()
    (d / "paginas").mkdir(exist_ok=True)
    px = a.dpi / 25.4
    for i, items in enumerate(pages, 1):
        img = cairo.ImageSurface(cairo.FORMAT_RGB24, int(A4[0] * px), int(A4[1] * px))
        c = cairo.Context(img); c.scale(px, px)
        draw_page(c, items, i)
        img.write_to_png(str(d / "paginas" / f"hoja_{i:02d}.png"))

    # --- OBJ texturizado para la vista previa 3D
    with open(d / "modelo.mtl", "w") as m:
        for name, fn in texfiles.items():
            m.write(f"newmtl {name}\nKd 1 1 1\nmap_Kd {fn}\n\n")
    with open(d / "modelo.obj", "w") as o:
        o.write("mtllib modelo.mtl\n")
        for v in V * scale:
            o.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        vt = 0
        for name in names:
            o.write(f"usemtl {name}\n")
            for fi, f in enumerate(F):
                if src[fi][0] != name:
                    continue
                uv = src[fi][1]
                w, h = sources[name][0].get_width(), sources[name][0].get_height()
                for u, v in uv:
                    o.write(f"vt {u / w:.5f} {1 - v / h:.5f}\n")
                o.write(f"f {f[0]+1}/{vt+1} {f[1]+1}/{vt+2} {f[2]+1}/{vt+3}\n")
                vt += 3

    nfront = sum(1 for s in src if s[0] == names[0])
    print(f"ESCALA {scale:.3f} mm/u · {len(F)} caras ({nfront} foto) · {len(islands)} piezas · "
          f"{len(cut)} aristas a pegar · {len(pages)} hojas A4")


if __name__ == "__main__":
    main()
