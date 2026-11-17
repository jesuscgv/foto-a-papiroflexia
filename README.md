# foto → papiroflexia 🦊✂️

> Convierte **una foto** en una **plantilla de papercraft imprimible** (PDF A4, con pestañas numeradas y líneas de monte/valle). La recortas, la doblas, la pegas… y tienes la figura en papel sobre la mesa.

Un fin de semana me obsesioné con los vídeos de papercraft en los que alguien imprime unas hojas, las recorta y acaba montando un bicho de papel con todo el volumen. Me pregunté *"¿y si pudiera salir de una foto cualquiera, en automático?"*. Esto es lo que salió: un pipeline que va de la imagen plana al modelo 3D, lo simplifica a low-poly, lo **despliega** en piezas planas y lo maqueta en un PDF listo para imprimir.

**Autor:** [@jesuscgv](https://github.com/jesuscgv) · Hecho por diversión, publicado por si a alguien le sirve o le apetece cacharrear.

---

## Qué hace, en una línea

```
foto.png  ─►  máscara + profundidad  ─►  malla 3D  ─►  low-poly  ─►  piezas desplegadas  ─►  plantilla.pdf
```

Una sola orden:

```bash
./papercraft.sh foto.png
# → out/foto/plantilla.pdf   (portada con instrucciones + hojas de piezas)
# → out/foto/vista_3d.png    (cómo va a quedar montado)
```

---

## El pipeline, paso a paso

| # | Script | Entra | Sale | Qué hace |
|---|--------|-------|------|----------|
| 1 | `prep.py` | foto | `dense.obj` + texturas | Recorta el fondo (por luminancia o con IA), estima un mapa de **profundidad**, limpia la textura y genera una malla densa cerrada con relieve. |
| 2 | `decimate.py` | `dense.obj` | `low.obj` | **Blender headless**: suaviza el escalonado y reduce a una low-poly manejable (p. ej. 220 caras), cuidando que quede *manifold* para poder desplegarla. |
| 3 | `unfold.py` | `low.obj` + texturas | `plantilla.pdf` | **Despliega** la malla en islas planas, añade **pestañas de pegado numeradas**, reparte las piezas en hojas A4 y dibuja corte / monte / valle. También saca el OBJ texturizado. |
| 4 | `preview.py` | `modelo.obj` | `vista_3d.png` | **Blender headless**: render de cómo quedará la figura ya montada, para la portada del PDF. |

El corazón "divertido" está en `unfold.py`: planta una cara, va pegando caras adyacentes abatiéndolas sobre el plano mientras no se solapen, y cuando una no cabe abre una **isla nueva**. Cada arista compartida entre dos islas recibe una pestaña con un número N, y la arista gemela lleva el mismo N → montar es "pestaña N bajo arista N".

---

## Instalación

Necesitas **Python 3.13**, **Blender** (se usa en modo headless) e **ImageMagick** (`magick`).

```bash
git clone https://github.com/jesuscgv/foto-a-papiroflexia.git
cd foto-a-papiroflexia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

En macOS, el script espera Blender en `/Applications/Blender.app`. Si lo tienes en otra ruta, edita la variable `BL` de `papercraft.sh`.

---

## Uso

```bash
./papercraft.sh foto.png [opciones]
```

| Opción | Por defecto | Para qué |
|--------|-------------|----------|
| `--caras N` | 220 | Nº de caras de la low-poly. Menos = más fácil de montar, más facetado. |
| `--alto cm` | 18 | Altura de la figura montada (reescala las piezas). |
| `--compacto f` | 0.6 | Cómo de apretadas van las piezas en las hojas (ahorra papel). |
| `--grosor f` | 0.9 | Relieve de la profundidad estimada. |
| `--segmentar auto\|ia\|claro` | auto | Recorte del fondo. `ia` usa rembg (fondos reales); `auto` va por luminancia. |
| `--recorte x,y,w,h` | — | Recorta la foto antes de empezar. |
| `--salida ruta` | `out/<foto>` | Dónde dejar los resultados. |

**Consejo de montaje** (va también en la portada del PDF): imprime al 100 % en papel de 160–200 g/m², marca los pliegues con un boli sin tinta, dobla todo antes de pegar y empieza por la cara dejando la base para el final.

---

## Rutas avanzadas

El pipeline básico es de una sola foto, pero el repo trae piezas para ir más allá:

- **3 vistas** (`prep3.py`): frente + perfil + espalda → la forma sale del perfil/espalda y la cara se usa como textura. Mucho más fiel.
- **Desde un modelo 3D** (`from_model.py`): si ya tienes un `.glb` texturizado (p. ej. de un *image-to-3D* como MakerWorld o Tripo), lo convierte en busto low-poly con la textura horneada, listo para desplegar.
- **Cuadernillo de regalo** (`kit.py`): monta un PDF estilo revista con portada, materiales, mapa de piezas de colores y montaje paso a paso (apoyado en `piecemap.py` y `render.py`).

---

## Cómo funciona por dentro (para curiosos)

- **Segmentación**: por defecto marca como fondo los píxeles claros y poco saturados conectados al borde; opcionalmente IS-Net vía `rembg`.
- **Profundidad**: un mapa suave a partir de la silueta y el sombreado, extruido para cerrar una malla con volumen.
- **Low-poly estable**: suavizado laplaciano + *decimate* por colapso, con reintentos si quedan aristas no-manifold (una low-poly no-manifold no se puede desplegar).
- **Unfold**: BFS sobre las caras abatiendo cada una al plano; detección de solape con *shapely*; pestañas generadas por geometría; maquetación A4 en **cairo** (PDF vectorial de verdad, no un PNG metido en una página).

---

## Créditos e inspiración

Inspirado por los vídeos de papercraft que circulan por TikTok/Instagram, del estilo de
[**@the_paperkingdom**](https://www.tiktok.com/@the_paperkingdom/video/7489844694990458117) —
esos montajes en los que una figura de papel va cobrando volumen hoja a hoja.
Esto es mi versión *"hazlo automático desde una foto"*.

Las plantillas generadas llevan una pequeña marca de agua **@jesuscgv** en la portada y el pie de cada hoja.

## Licencia

[MIT](LICENSE) © 2026 Jesús ([@jesuscgv](https://github.com/jesuscgv)). Úsalo, modifícalo y comparte lo que montes. Si te sale algo chulo, etiquétame 🙂
