"""Paso 1: foto -> máscara, profundidad, textura limpia, textura de pelo y malla densa cerrada.

Uso: python prep.py foto.png carpeta_salida [--grosor 0.9]
"""
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import measure


def segment(rgb):
    """Fondo = píxeles claros y sin saturación conectados al borde (cuadriculado falso o fondo liso).
    Si la imagen trae alfa real, se usa ese."""
    f = rgb.astype(np.float32)
    mx, mn = f.max(2), f.min(2)
    bgish = (mn > 200) & (mx - mn < 22)
    # rellena huecos del fondo (líneas finas de marca de agua) y conecta con el borde
    bgish = ndi.binary_closing(np.pad(bgish, 4, constant_values=True), iterations=2)[4:-4, 4:-4]
    lab, _ = ndi.label(bgish)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, border[border > 0])
    fg = ~bg
    fg = ndi.binary_opening(fg, iterations=2)
    lab, n = ndi.label(fg)
    if n > 1:
        sizes = ndi.sum(fg, lab, range(1, n + 1))
        fg = lab == (1 + int(np.argmax(sizes)))
    fg = ndi.binary_fill_holes(fg)
    return fg


def segment_ai(img):
    """Recorte con IA (rembg / IS-Net) para fondos reales."""
    from rembg import remove, new_session
    m = remove(img, session=new_session("isnet-general-use"), only_mask=True)
    m = np.asarray(m) > 127
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    return ndi.binary_fill_holes(m)


def plain_background(rgb):
    f = rgb.astype(np.int16)
    b = np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]])
    return ((b.min(1) > 200) & (b.max(1) - b.min(1) < 22)).mean() > 0.6


def bleed(rgb, mask):
    """Cada píxel de fuera toma el color del píxel del sujeto más cercano (sin halos)."""
    _, (iy, ix) = ndi.distance_transform_edt(~mask, return_indices=True)
    return rgb[iy, ix]


def back_texture(rgb, mask):
    """Textura para la espalda y los lados: los colores de la foto a gran escala (sin ojos, nariz ni lengua)
    + el detalle de pelo de un parche sin rasgos, en mosaico espejado."""
    f = rgb.astype(np.float32)
    lum = f.mean(2)
    width = mask.any(0).sum()
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    ok = mask & (lum > np.percentile(lum[mask], 10)) & ~(b > g + 3) & ~((r - g > 25) & (lum < 80) & (b > 0.6 * g))  # fuera: lo muy oscuro (ojos, nariz) y lo rosa (lengua, arnés)
    sig = max(6, 0.05 * width)
    w = ndi.gaussian_filter(ok.astype(np.float32), sig)
    low = np.stack([ndi.gaussian_filter(f[..., c] * ok, sig) for c in range(3)], 2) / np.maximum(w, 1e-4)[..., None]
    low = bleed(low, w > 0.05)
    # parche de pelo: dentro del sujeto, con la menor energía en frecuencias medias (sin rasgos)
    p = max(48, int(width) // 6)
    band = np.abs(ndi.gaussian_filter(lum, 2) - ndi.gaussian_filter(lum, 10))
    er = ndi.binary_erosion(mask, iterations=p // 2 + 2)
    best, pos = None, None
    for y in range(0, rgb.shape[0] - p, max(4, p // 4)):
        for x in range(0, rgb.shape[1] - p, max(4, p // 4)):
            if not er[y + p // 2, x + p // 2] or not ok[y:y + p, x:x + p].all():
                continue
            sc = band[y:y + p, x:x + p].mean()
            if best is None or sc < best:
                best, pos = sc, (y, x)
    detail = np.zeros_like(lum)
    if pos:
        y, x = pos
        hp = lum[y:y + p, x:x + p] - ndi.gaussian_filter(lum[y:y + p, x:x + p], 3)
        t = np.concatenate([hp, hp[:, ::-1]], 1)
        t = np.concatenate([t, t[::-1]], 0)
        detail = np.tile(t, (rgb.shape[0] // t.shape[0] + 1, rgb.shape[1] // t.shape[1] + 1))[:rgb.shape[0], :rgb.shape[1]]
    return np.clip(low + 0.9 * detail[..., None], 0, 255).astype(np.uint8), pos


def depth_map(img):
    import torch
    from transformers import pipeline
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=dev)
    d = np.asarray(pipe(img)["predicted_depth"].squeeze().float().cpu().numpy())
    d = np.asarray(Image.fromarray(d).resize(img.size, Image.BILINEAR))
    return d  # mayor = más cerca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("foto")
    ap.add_argument("out")
    ap.add_argument("--grosor", type=float, default=0.9, help="fondo total / anchura del sujeto")
    ap.add_argument("--relieve", type=float, default=0.35, help="peso del mapa de profundidad en la cara")
    ap.add_argument("--recorte", help="x,y,ancho,alto en píxeles de la foto original")
    ap.add_argument("--segmentar", choices=["auto", "ia", "claro"], default="auto")
    ap.add_argument("--res", type=int, default=160, help="resolución de la malla densa")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    img = ImageOps.exif_transpose(Image.open(a.foto))
    if a.recorte:
        x, y, w_, h_ = map(int, a.recorte.split(","))
        img = img.crop((x, y, x + w_, y + h_))
    if max(img.size) > 1600:  # suficiente para imprimir a ~20 cm y la malla va más rápida
        k = 1600 / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    alpha = np.asarray(img.convert("RGBA"))[..., 3] if img.mode in ("RGBA", "LA", "P") else None
    img = img.convert("RGB")
    rgb = np.asarray(img)
    H, W = rgb.shape[:2]
    if alpha is not None and alpha.min() < 128:
        mask = alpha > 127
    elif a.segmentar == "claro" or (a.segmentar == "auto" and plain_background(rgb)):
        mask = segment(rgb)
    else:
        mask = segment_ai(img)
    img.save(out / "foto.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "mask.png")

    clean = bleed(rgb, ndi.binary_erosion(mask, iterations=max(2, W // 160)))
    Image.fromarray(clean).save(out / "photo_clean.png")
    fur, fpos = back_texture(rgb, mask)
    Image.fromarray(fur).save(out / "fur.png")

    depth = depth_map(img)
    dm = depth[mask]
    dn = np.clip((depth - np.percentile(dm, 2)) / (np.percentile(dm, 98) - np.percentile(dm, 2)), 0, 1)
    Image.fromarray((dn * mask * 255).astype(np.uint8)).save(out / "depth.png")

    # --- volumen: frente = inflado + relieve; espalda = inflado. Abajo se corta plano si toca el borde.
    s = a.res / max(H, W)
    w, h = int(round(W * s)), int(round(H * s))
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127
    d = np.asarray(Image.fromarray((dn * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    d = ndi.gaussian_filter(d, 1.2)
    touches_bottom = m[-1].sum() > 0.15 * w
    pad = h if touches_bottom else 0
    mp = np.pad(m, ((0, pad), (0, 0)), mode="edge")
    dt = ndi.distance_transform_edt(mp)[:h]
    inflate = np.sqrt(dt / dt.max())  # 0 en el borde, 1 en el centro, perfil redondo
    width_px = m.any(0).sum()
    T = a.grosor * width_px
    front = (0.5 * (1 - a.relieve) * inflate + 0.5 * a.relieve * d * inflate ** 0.5) * T
    back = 0.42 * inflate * T
    front = ndi.gaussian_filter(front, 1.0) * m
    back = ndi.gaussian_filter(back, 1.0) * m
    zmin, zmax = -int(np.ceil(back.max())) - 2, int(np.ceil(front.max())) + 2
    zs = np.arange(zmin, zmax + 1)
    vol = (zs[None, None, :] <= front[..., None]) & (zs[None, None, :] >= -back[..., None]) & m[..., None]
    vol = np.pad(vol, 1)  # cierra todas las caras (incluida la base)
    vol = ndi.gaussian_filter(vol.astype(np.float32), 0.8)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    verts -= 1
    # (fila, col, z) -> coordenadas: x = col, y = -fila (arriba +), z = z
    V = np.stack([verts[:, 1], -verts[:, 0], verts[:, 2] + zmin], 1) / s  # en píxeles de la foto
    with open(out / "dense.obj", "w") as fobj:
        for v in V:
            fobj.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for f in faces:
            fobj.write(f"f {f[0]+1} {f[2]+1} {f[1]+1}\n")
    json.dump({"W": W, "H": H, "flat_bottom": bool(touches_bottom), "fur_patch": fpos,
               "dense_faces": int(len(faces))}, open(out / "meta.json", "w"))
    print(f"máscara {mask.mean():.0%} · malla densa {len(faces)} caras · base plana {touches_bottom}")


if __name__ == "__main__":
    main()
