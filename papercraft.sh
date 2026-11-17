#!/usr/bin/env bash
# Foto -> plantilla de papercraft imprimible (PDF A4).
# Uso: ./papercraft.sh foto.png [--recorte x,y,w,h] [--segmentar auto|ia|claro] [--relieve 0.35] [--caras 220] [--alto 18] [--compacto 0.6] [--grosor 0.9] [--salida out/nombre]
set -euo pipefail
cd "$(dirname "$0")"
FOTO="$1"; shift
CARAS=220; ALTO=18; COMPACTO=0.6; GROSOR=0.9; EXTRA=(); OUT="out/$(basename "${FOTO%.*}")"
while [[ $# -gt 0 ]]; do case "$1" in
  --caras) CARAS="$2"; shift 2;; --alto) ALTO="$2"; shift 2;; --compacto) COMPACTO="$2"; shift 2;;
  --grosor) GROSOR="$2"; shift 2;; --salida) OUT="$2"; shift 2;;
  --recorte|--segmentar|--relieve) EXTRA+=("$1" "$2"); shift 2;; *) echo "opción desconocida: $1"; exit 1;; esac; done
PY=.venv/bin/python
BL=/Applications/Blender.app/Contents/MacOS/Blender
echo "1/4 recorte + profundidad";  $PY papercraft/prep.py "$FOTO" "$OUT" --grosor "$GROSOR" ${EXTRA[@]+"${EXTRA[@]}"} 2>/dev/null | tail -1
echo "2/4 low-poly";               $BL -b -P papercraft/decimate.py -- "$OUT/dense.obj" "$OUT/low.obj" "$CARAS" 2>&1 | grep DECIMATE
echo "3/4 despliegue";             $PY papercraft/unfold.py "$OUT" --alto "$ALTO" --compacto "$COMPACTO" >/dev/null
echo "4/4 vista previa + PDF"
( cd "$OUT" && "$BL" -b -P ../../papercraft/preview.py -- modelo.obj prev3d >/dev/null 2>&1 \
  && magick prev3d.*.png +append vista_3d.png && rm prev3d.*.png )
$PY papercraft/unfold.py "$OUT" --alto "$ALTO" --compacto "$COMPACTO" --portada "$OUT/vista_3d.png"
echo "→ $OUT/plantilla.pdf"
