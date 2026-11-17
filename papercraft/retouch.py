"""Retoques sobre la textura horneada (baked.png) de la low-poly:
  1. Tapas del corte sin textura (negras) -> color medio de las caras vecinas.
  2. (opcional) Ojos: sustituye el ojo por pelo del entorno y dibuja un pliegue de ojo cerrado.

python retouch.py carpeta [--ojos ojos.json] [--curva 0.25]
ojos.json: {"L": centro, "L1": comisura, "L2": comisura, "R": ..., "R1": ..., "R2": ...} en coords del OBJ.
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import sys
sys.path.insert(0, str(Path(__file__).parent))
from unfold import load_obj


def raster(uvpx, W, H):
    """Píxeles (x, y) dentro del triángulo uvpx (3x2, en píxeles) y sus baricéntricas."""
    x0, y0 = np.floor(uvpx.min(0)).astype(int)
    x1, y1 = np.ceil(uvpx.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W - 1), min(y1, H - 1)
    if x1 < x0 or y1 < y0:
        return np.zeros((0, 2), int), np.zeros((0, 3))
    X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    P = np.stack([X.ravel() + 0.5, Y.ravel() + 0.5], 1)
    a, b, c = uvpx
    M = np.array([b - a, c - a]).T
    if abs(np.linalg.det(M)) < 1e-9:
        return np.zeros((0, 2), int), np.zeros((0, 3))
    l = np.linalg.solve(M, (P - a).T).T
    bc = np.stack([1 - l[:, 0] - l[:, 1], l[:, 0], l[:, 1]], 1)
    ok = (bc >= -0.02).all(1)
    return P[ok].astype(int), bc[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ojos")
    ap.add_argument("--curva", type=float, default=0.25, help="curvatura del pliegue del ojo cerrado")
    ap.add_argument("--grosor", type=float, default=0.10, help="grosor del pliegue (en radios del ojo)")
    a = ap.parse_args()
    d = Path(a.dir)
    V, F, UV = load_obj(d / "low.obj")
    orig = d / "baked_orig.png"
    if not orig.exists():  # el retoque siempre parte del horneado original (se puede repetir)
        Image.open(d / "baked.png").save(orig)
    img = np.asarray(Image.open(orig).convert("RGB")).astype(np.float32)
    H, W = img.shape[:2]
    uvpx = np.stack([UV[..., 0] * W, (1 - UV[..., 1]) * H], -1)  # (caras, 3, 2)
    normals = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # --- 1. huecos del horneado y tapas del corte
    tex = [raster(uvpx[i], W, H) for i in range(len(F))]
    size = np.ptp(V, 0).max()
    eps = 1e-2 * size
    cen = V[F].mean(1)
    on_base = (np.abs(cen[:, 1] - V[:, 1].min()) < 0.03 * size) & (normals[:, 1] < -0.9)
    on_back = (np.abs(cen[:, 2] - V[:, 2].min()) < 0.03 * size) & (normals[:, 2] < -0.9)
    cap = on_base | on_back
    mean = np.zeros((len(F), 3)); bad = np.zeros(len(F), bool)
    nfix = 0
    for i, (p, _) in enumerate(tex):
        if not len(p) or cap[i]:
            bad[i] = True
            continue
        col = img[p[:, 1], p[:, 0]]
        miss = col.max(1) < 30  # rayo del horneado sin impacto (negro o casi)
        if miss.all() or miss.mean() > 0.85:
            bad[i] = True
            continue
        if miss.any():  # cada texel negro toma el texel bueno más cercano de la misma cara
            from scipy.spatial import cKDTree
            _, j = cKDTree(p[~miss]).query(p[miss])
            col[miss] = col[~miss][j]
            img[p[:, 1], p[:, 0]] = col
            nfix += 1
        mean[i] = col.mean(0)
    adj = {}
    for i, f in enumerate(F):
        for k in range(3):
            adj.setdefault(tuple(sorted((f[k], f[(k + 1) % 3]))), []).append(i)
    neigh = {i: set() for i in range(len(F))}
    for fs in adj.values():
        for i in fs:
            neigh[i].update(j for j in fs if j != i)
    refill = list(np.nonzero(bad)[0])
    todo = list(refill)
    ncap = len(refill)
    for _ in range(40):  # propaga el color desde el borde de la tapa hacia dentro
        nxt = []
        done = []
        for i in todo:
            good = [mean[j] for j in neigh[i] if not bad[j]]
            if good:
                mean[i] = np.mean(good, 0); done.append(i)
            else:
                nxt.append(i)
        for i in done:
            bad[i] = False
        if not nxt:
            break
        todo = nxt
    for i in refill:
        p, _ = tex[i]
        if len(p):
            img[p[:, 1], p[:, 0]] = mean[i]
    print(f"caras con huecos arreglados: {nfix}")

    # --- 2. ojos cerrados
    neyes = 0
    if a.ojos:
        eyes = json.load(open(a.ojos))
        for side in ("L", "R"):
            c = np.array(eyes[side]); c1, c2 = np.array(eyes[side + "1"]), np.array(eyes[side + "2"])
            R = 0.5 * np.linalg.norm(c2 - c1) * 1.25
            cen = np.array([np.linalg.norm(V[f].mean(0) - c) for f in F])
            near = np.nonzero(cen < 2.5 * R)[0]
            # normal media de la zona y marco local del ojo
            n = normals[near].mean(0); n /= np.linalg.norm(n)
            e1 = (c2 - c1) - ((c2 - c1) @ n) * n; e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            if e2[1] < 0:
                e2 = -e2
            # color del pelo alrededor (anillo 1.1R-1.6R)
            ring, inside = [], []
            for i in near:
                p, bc = tex[i]
                if not len(p):
                    continue
                P3 = bc @ V[F[i]]
                s = (P3 - c) @ e1 / R; t = (P3 - c) @ e2 / R
                r = np.sqrt(s ** 2 + (t / 0.75) ** 2)
                ring.append(img[p[r > 1.1][:, 1], p[r > 1.1][:, 0]] if (r > 1.1).any() else np.zeros((0, 3)))
                inside.append((p, s, t, r))
            ring = np.concatenate([x for x in ring if len(x)])
            ring_lum = ring.mean(1)
            fur = np.median(ring, 0)
            soft = ndi.gaussian_filter(img, (6, 6, 0))
            for p, s, t, r in inside:
                # 1) pelo: mezcla suave hacia el color de alrededor (conserva algo de la textura difuminada)
                w = np.clip((1.15 - r) / 0.3, 0, 1)[:, None]
                base = 0.3 * fur + 0.7 * soft[p[:, 1], p[:, 0]]
                col = img[p[:, 1], p[:, 0]] * (1 - w) + base * w
                # 2) pliegue del ojo cerrado: arco suave, más grueso en el centro
                tc = a.curva * (0.45 - s ** 2)
                th = a.grosor * np.sqrt(np.clip(1 - (s / 0.95) ** 2, 0, 1))
                dist = np.abs(t - tc)
                line = np.clip(1 - dist / np.maximum(th, 1e-6), 0, 1) ** 0.8 * (np.abs(s) < 0.95)
                dark_col = np.array([52, 32, 24], np.float32)
                col = col * (1 - line[:, None]) + dark_col * line[:, None]
                img[p[:, 1], p[:, 0]] = col
                neyes += len(p)
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(d / "baked.png")
    print(f"RETOQUE tapas: {ncap} caras · ojos: {neyes} píxeles")


if __name__ == "__main__":
    main()
