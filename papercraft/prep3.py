"""Paso 1 (versión 3 vistas): frente + perfil + espalda -> malla densa cerrada y texturas por vista.

La FORMA sale del perfil y de la espalda (vistas coherentes, casi ortográficas): casco visual
con secciones redondeadas (superelipses). La foto de FRENTE (real, con perspectiva) se usa solo
como textura, deformada con marcas verticales (coronilla, ojos, nariz...) y anchura por fila.

python prep3.py --frente f.png --perfil p.jpg --atras a.jpg --marcas "185:170,335:370,..." \
                --eje-frente 437 --corte 1100 out/
  marcas: fila_en_perfil:fila_en_frente (el perfil y la espalda comparten escala)
  perfil: el perro mira a la IZQUIERDA (si no, usa --perfil-derecha)
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import measure


def load(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def mask_of(img, session):
    from rembg import remove
    m = np.asarray(remove(img, session=session, only_mask=True)) > 127
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == 1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1))))
    return ndi.binary_fill_holes(m)


def bleed(rgb, mask):
    _, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
    return rgb[iy, ix]


def symmetrize(m, axis):
    """Intersección de la máscara con su espejo respecto a la columna 'axis'."""
    H, W = m.shape
    cols = np.arange(W)
    mir = (2 * axis - cols).round().astype(int)
    ok = (mir >= 0) & (mir < W)
    out = np.zeros_like(m)
    out[:, ok] = m[:, ok] & m[:, mir[ok]]
    return out


def half_width(m, axis):
    """Por fila: distancia del eje al borde del tramo que contiene el eje (0 si no hay)."""
    H, W = m.shape
    a = int(round(axis))
    hw = np.zeros(H)
    for y in range(H):
        r = m[y]
        if not r[a]:
            continue
        right = a + np.argmin(r[a:]) if not r[a:].all() else W
        left = a - np.argmin(r[a::-1]) if not r[a::-1].all() else -1
        hw[y] = (right - left) / 2
    return hw


def intervals(row):
    """Para una fila booleana: centro y semiancho del tramo al que pertenece cada píxel."""
    c = np.zeros(len(row)); h = np.zeros(len(row))
    lab, n = ndi.label(row)
    for i in range(1, n + 1):
        idx = np.nonzero(lab == i)[0]
        c[idx] = (idx[0] + idx[-1]) / 2
        h[idx] = (idx[-1] - idx[0] + 1) / 2
    return c, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frente", required=True)
    ap.add_argument("--perfil", required=True)
    ap.add_argument("--atras", required=True)
    ap.add_argument("--marcas", required=True)
    ap.add_argument("--eje-frente", type=float, required=True, help="columna de la nariz en la foto de frente")
    ap.add_argument("--corte", type=int, required=True, help="fila (perfil/espalda) donde se corta el busto")
    ap.add_argument("--perfil-derecha", action="store_true")
    ap.add_argument("--redondez", type=float, default=2.4, help="exponente de la superelipse (2 = elipse)")
    ap.add_argument("--quitar-rosa-hasta", type=int, default=0,
                    help="quita de la silueta del perfil lo rosa (lengua) por encima de esta fila")
    ap.add_argument("--suavizado", type=int, default=3, help="radio (en vóxeles) para quitar pinchos")
    ap.add_argument("--res", type=int, default=200)
    ap.add_argument("out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from rembg import new_session
    ses = new_session("isnet-general-use")
    imf, ims, imb = load(a.frente), load(a.perfil), load(a.atras)
    if a.perfil_derecha:
        ims = ImageOps.mirror(ims)
    if ims.size != imb.size:
        ims = ims.resize(imb.size, Image.LANCZOS)
    mf, ms, mb = mask_of(imf, ses), mask_of(ims, ses), mask_of(imb, ses)
    H, W = mb.shape

    # eje de simetría de la espalda: centro medio de las filas de la cabeza
    top = np.nonzero(mb.any(1))[0][0]
    rows = range(top + 40, min(H, top + 300))
    cx = float(np.median([np.nonzero(mb[y])[0][[0, -1]].mean() for y in rows if mb[y].any()]))
    bsym = symmetrize(mb, cx)
    bsym[a.corte:] = False
    ms_full = ms
    if a.quitar_rosa_hasta:
        f = np.asarray(ims).astype(int)
        pink = (f[..., 2] > f[..., 1] + 5) & (f[..., 0] > f[..., 1] + 20)
        pink[a.quitar_rosa_hasta:] = False
        pink = ndi.binary_dilation(pink, iterations=4)
        ms = ndi.binary_opening(ms & ~pink, iterations=6)
        lab, n = ndi.label(ms)
        ms = lab == 1 + int(np.argmax(ndi.sum(ms, lab, range(1, n + 1))))
        ms = ndi.binary_fill_holes(ms)
    ms = ms.copy(); ms[a.corte:] = False
    cz = float(np.nonzero(ms.any(0))[0].mean())

    # --- casco visual redondeado, en una rejilla reducida
    s = a.res / H
    def small(m):
        return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((round(m.shape[1] * s), round(m.shape[0] * s)), Image.BILINEAR)) > 127
    B, S = small(bsym), small(ms)
    # siluetas "de papel": une orejas a la cabeza (cierre) y quita láminas finas (apertura)
    disk = lambda r: (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r)
    def tidy(m):
        mp = np.pad(m, 8)
        mp = ndi.binary_closing(mp, disk(a.suavizado + 1))
        mp = ndi.binary_opening(mp, disk(a.suavizado))
        m2 = mp[8:-8, 8:-8]
        m2[-1] = m[-1]  # conserva la base plana
        return ndi.binary_fill_holes(m2)
    B, S = tidy(B), tidy(S)
    h, wx = B.shape
    wz = S.shape[1]
    p = a.redondez
    vol = np.zeros((h, wx, wz), bool)
    for y in range(h):
        if not B[y].any() or not S[y].any():
            continue
        xc, ax = intervals(B[y])
        zc, az = intervals(S[y])
        X = np.arange(wx)[:, None]; Z = np.arange(wz)[None, :]
        inside = B[y][:, None] & S[y][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            e = (np.abs(X - xc[:, None]) / ax[:, None]) ** p + (np.abs(Z - zc[None, :]) / az[None, :]) ** p
        vol[y] = inside & (e <= 1.0)
    # base plana: el corte cierra el volumen
    vol = np.pad(vol, 1)
    vol = ndi.gaussian_filter(vol.astype(np.float32), 0.9)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    verts -= 1
    # (fila, col_x, col_z) -> mundo en píxeles de perfil/espalda:
    #   x = cx - u_espalda   (+x = lado izquierdo del perro = derecha en la foto de frente)
    #   y = -fila,  z = cz - u_perfil  (+z = hacia delante)
    V = np.stack([cx - verts[:, 1] / s, -verts[:, 0] / s, cz - verts[:, 2] / s], 1)
    with open(out / "dense.obj", "w") as fo:
        for v in V:
            fo.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for f in faces:
            fo.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")

    # --- deformación de la foto de frente: fila del modelo -> fila en la foto, y escala horizontal
    mk = sorted(tuple(map(float, t.split(":"))) for t in a.marcas.split(","))
    ym, yf = zip(*mk)
    rows_m = np.arange(H)
    fy = np.interp(rows_m, ym, yf, left=None, right=None)
    # extrapolación lineal por los extremos
    lo = (yf[1] - yf[0]) / (ym[1] - ym[0]); hi = (yf[-1] - yf[-2]) / (ym[-1] - ym[-2])
    fy = np.where(rows_m < ym[0], yf[0] + (rows_m - ym[0]) * lo, fy)
    fy = np.where(rows_m > ym[-1], yf[-1] + (rows_m - ym[-1]) * hi, fy)
    fsym = symmetrize(mf, a.eje_frente)
    hw_f = half_width(fsym, a.eje_frente)
    hw_m = half_width(bsym, cx)
    fr = np.clip(fy.round().astype(int), 0, len(hw_f) - 1)
    ratio = np.where((hw_m > 5) & (hw_f[fr] > 5), hw_f[fr] / np.maximum(hw_m, 1), np.nan)
    good = ~np.isnan(ratio)
    ratio = np.interp(rows_m, rows_m[good], ratio[good])
    ratio = ndi.gaussian_filter1d(ndi.median_filter(ratio, 41), 15)

    # --- texturas limpias (bordes extendidos con el color más cercano)
    er = lambda m: ndi.binary_erosion(m, iterations=3)
    Image.fromarray(bleed(np.asarray(imf), er(mf))).save(out / "frente_clean.png")
    Image.fromarray(bleed(np.asarray(ims), er(ms_full))).save(out / "perfil_clean.png")
    Image.fromarray(bleed(np.asarray(imb), er(mb))).save(out / "atras_clean.png")
    for n, m in (("frente", mf), ("perfil", ms_full), ("atras", mb)):
        Image.fromarray((m * 255).astype(np.uint8)).save(out / f"mask_{n}.png")

    json.dump({
        "W": W, "H": H, "multiview": True,
        "textures": {"frente": "frente_clean.png", "perfil": "perfil_clean.png", "atras": "atras_clean.png"},
        "cx": cx, "cz": cz, "eje_frente": a.eje_frente, "corte": a.corte,
        "fy": [round(v, 2) for v in fy], "sx": [round(v, 4) for v in ratio],
    }, open(out / "meta.json", "w"))
    print(f"casco: {len(faces)} caras · eje espalda {cx:.0f} · escala horizontal frente {np.median(ratio):.2f}")


if __name__ == "__main__":
    main()
