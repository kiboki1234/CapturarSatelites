from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    ROOT
    / "outputs"
    / "images"
    / "msumr_10s"
)

OUTPUT_DIR = INPUT_DIR


CHANNEL_FILES = {
    1: INPUT_DIR / "meteor_msumr_canal_1.png",
    2: INPUT_DIR / "meteor_msumr_canal_2.png",
    3: INPUT_DIR / "meteor_msumr_canal_3.png",
}


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Los segmentos MSU-MR tienen 8 px de alto.
# Por eso buscamos desplazamientos verticales
# únicamente en múltiplos de 8.
ALIGN_STEP = 8

MAX_VERTICAL_SHIFT = 16


# Percentiles para contraste.
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


# Gamma visual.
GAMMA = 0.90


# ============================================================
# CARGAR CANALES
# ============================================================

def load_gray(path):

    if not path.exists():

        raise FileNotFoundError(
            f"No existe:\n{path}"
        )

    return np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8
    )


print()
print("================================================")
print("      COMPOSICIÓN RGB METEOR MSU-MR")
print("================================================")
print()


channels = {}

for channel, path in CHANNEL_FILES.items():

    channels[channel] = load_gray(path)

    print(
        f"Canal {channel}: "
        f"{channels[channel].shape[1]} x "
        f"{channels[channel].shape[0]}"
    )


# ============================================================
# VALIDAR DIMENSIONES
# ============================================================

widths = {
    img.shape[1]
    for img in channels.values()
}

if len(widths) != 1:

    raise RuntimeError(
        "Los canales no tienen el mismo ancho."
    )


# ============================================================
# CORRELACIÓN NORMALIZADA
# ============================================================

def normalized_correlation(a, b):

    a = a.astype(
        np.float64
    )

    b = b.astype(
        np.float64
    )


    # Ignorar zonas negras de datos faltantes.
    mask = (
        (a > 5)
        &
        (b > 5)
    )


    if np.count_nonzero(mask) < 100:

        return -1.0


    x = a[mask]

    y = b[mask]


    x = (
        x
        - np.mean(x)
    )

    y = (
        y
        - np.mean(y)
    )


    sx = np.std(x)

    sy = np.std(y)


    if (
        sx < 1e-12
        or sy < 1e-12
    ):

        return -1.0


    return float(
        np.mean(
            (
                x / sx
            )
            *
            (
                y / sy
            )
        )
    )


# ============================================================
# BUSCAR DESPLAZAMIENTO VERTICAL
#
# Devuelve cuánto debemos MOVER la imagen para alinearla
# con la referencia.
#
# shift > 0 = mover hacia abajo
# shift < 0 = mover hacia arriba
# ============================================================

def find_vertical_shift(
    reference,
    image,
    max_shift=16,
    step=8
):

    best_score = -999.0

    best_move = 0


    for move in range(
        -max_shift,
        max_shift + 1,
        step
    ):

        # -----------------------------------------------
        # move > 0:
        # imagen desplazada hacia abajo
        # -----------------------------------------------

        if move > 0:

            ref_part = reference[
                move:
            ]

            img_part = image[
                :-move
            ]


        # -----------------------------------------------
        # move < 0:
        # imagen desplazada hacia arriba
        # -----------------------------------------------

        elif move < 0:

            amount = (
                -move
            )

            ref_part = reference[
                :-amount
            ]

            img_part = image[
                amount:
            ]


        else:

            ref_part = reference

            img_part = image


        # -----------------------------------------------
        # Altura común
        # -----------------------------------------------

        h = min(
            ref_part.shape[0],
            img_part.shape[0]
        )


        if h <= 0:

            continue


        score = normalized_correlation(
            ref_part[:h],
            img_part[:h]
        )


        if score > best_score:

            best_score = score

            best_move = move


    return (
        best_move,
        best_score
    )


# ============================================================
# ALINEAMIENTO
#
# Usaremos Canal 2 como referencia.
# ============================================================

reference = channels[2]


shift_1, corr_1 = find_vertical_shift(
    reference,
    channels[1],
    max_shift=MAX_VERTICAL_SHIFT,
    step=ALIGN_STEP
)

shift_2 = 0
corr_2 = 1.0

shift_3, corr_3 = find_vertical_shift(
    reference,
    channels[3],
    max_shift=MAX_VERTICAL_SHIFT,
    step=ALIGN_STEP
)


shifts = {
    1: shift_1,
    2: shift_2,
    3: shift_3,
}


print()
print("================================================")
print("              ALINEAMIENTO")
print("================================================")
print()

print(
    f"Canal 1 -> shift={shift_1:+d} px | "
    f"corr={corr_1:.4f}"
)

print(
    f"Canal 2 -> shift={shift_2:+d} px | "
    f"corr={corr_2:.4f}  [referencia]"
)

print(
    f"Canal 3 -> shift={shift_3:+d} px | "
    f"corr={corr_3:.4f}"
)


# ============================================================
# DESPLAZAR CON NaN
#
# NaN permite saber qué filas no contienen datos después
# de mover cada imagen.
# ============================================================

def shift_image(
    image,
    move
):

    h, w = image.shape

    result = np.full(
        (
            h,
            w
        ),
        np.nan,
        dtype=np.float32
    )


    if move > 0:

        result[
            move:
        ] = image[
            :-move
        ]


    elif move < 0:

        amount = (
            -move
        )

        result[
            :-amount
        ] = image[
            amount:
        ]


    else:

        result[:] = image


    return result


aligned = {}

for channel in (
    1,
    2,
    3
):

    aligned[channel] = shift_image(
        channels[channel],
        shifts[channel]
    )


# ============================================================
# RECORTAR SOLO REGIÓN VERTICAL COMÚN
# ============================================================

valid_rows = np.ones(
    aligned[1].shape[0],
    dtype=bool
)


for channel in (
    1,
    2,
    3
):

    valid_rows &= np.all(
        ~np.isnan(
            aligned[channel]
        ),
        axis=1
    )


row_indices = np.where(
    valid_rows
)[0]


if len(row_indices) == 0:

    raise RuntimeError(
        "No existe una región común entre los tres canales."
    )


y0 = int(
    row_indices[0]
)

y1 = int(
    row_indices[-1]
    + 1
)


for channel in (
    1,
    2,
    3
):

    aligned[channel] = aligned[channel][
        y0:y1
    ].astype(
        np.uint8
    )


height = (
    y1 - y0
)

width = (
    aligned[1].shape[1]
)


print()
print(
    f"Región común: "
    f"y={y0}:{y1}"
)

print(
    f"Dimensión RGB: "
    f"{width} x {height}"
)


# ============================================================
# GUARDAR CANALES ALINEADOS
# ============================================================

for channel in (
    1,
    2,
    3
):

    file = (
        OUTPUT_DIR
        /
        f"meteor_msumr_canal_{channel}_aligned.png"
    )

    Image.fromarray(
        aligned[channel],
        mode="L"
    ).save(
        file
    )


# ============================================================
# STRETCH POR PERCENTILES
# ============================================================

def percentile_stretch(
    image,
    low=1.0,
    high=99.0
):

    x = image.astype(
        np.float32
    )


    # Evitar que bloques negros faltantes alteren
    # el histograma.
    valid = (
        x > 0
    )


    if not np.any(valid):

        return np.zeros_like(
            image
        )


    p_low = np.percentile(
        x[valid],
        low
    )

    p_high = np.percentile(
        x[valid],
        high
    )


    if (
        p_high
        <= p_low
    ):

        return image.copy()


    y = (
        x
        - p_low
    ) / (
        p_high
        - p_low
    )


    y = np.clip(
        y,
        0.0,
        1.0
    )


    return np.rint(
        y * 255.0
    ).astype(
        np.uint8
    )


# ============================================================
# GAMMA
# ============================================================

def apply_gamma(
    image,
    gamma=1.0
):

    x = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    x = np.power(
        np.clip(
            x,
            0.0,
            1.0
        ),
        gamma
    )


    return np.rint(
        x * 255.0
    ).astype(
        np.uint8
    )


# ============================================================
# CANALES ORIGINALES ALINEADOS
# ============================================================

c1 = aligned[1]

c2 = aligned[2]

c3 = aligned[3]


# ============================================================
# COMPOSICIÓN 1-2-3
#
# R = canal 1
# G = canal 2
# B = canal 3
#
# Esto es una composición experimental.
# ============================================================

rgb_123_raw = np.dstack(
    [
        c1,
        c2,
        c3
    ]
)


# ============================================================
# COMPOSICIÓN 3-2-1
#
# También la generamos para comparar visualmente.
#
# R = canal 3
# G = canal 2
# B = canal 1
# ============================================================

rgb_321_raw = np.dstack(
    [
        c3,
        c2,
        c1
    ]
)


# ============================================================
# REALCE INDIVIDUAL DE CADA CANAL
# ============================================================

c1_enh = percentile_stretch(
    c1,
    LOW_PERCENTILE,
    HIGH_PERCENTILE
)

c2_enh = percentile_stretch(
    c2,
    LOW_PERCENTILE,
    HIGH_PERCENTILE
)

c3_enh = percentile_stretch(
    c3,
    LOW_PERCENTILE,
    HIGH_PERCENTILE
)


c1_enh = apply_gamma(
    c1_enh,
    GAMMA
)

c2_enh = apply_gamma(
    c2_enh,
    GAMMA
)

c3_enh = apply_gamma(
    c3_enh,
    GAMMA
)


# ============================================================
# RGB REALZADO
# ============================================================

rgb_123_enhanced = np.dstack(
    [
        c1_enh,
        c2_enh,
        c3_enh
    ]
)


rgb_321_enhanced = np.dstack(
    [
        c3_enh,
        c2_enh,
        c1_enh
    ]
)


# ============================================================
# GUARDAR
# ============================================================

outputs = {
    "meteor_rgb_123_raw.png":
        rgb_123_raw,

    "meteor_rgb_123_enhanced.png":
        rgb_123_enhanced,

    "meteor_rgb_321_raw.png":
        rgb_321_raw,

    "meteor_rgb_321_enhanced.png":
        rgb_321_enhanced,
}


print()
print("================================================")
print("                 GUARDANDO")
print("================================================")
print()


for filename, array in outputs.items():

    path = (
        OUTPUT_DIR
        /
        filename
    )

    Image.fromarray(
        array,
        mode="RGB"
    ).save(
        path
    )

    print(
        path
    )


# ============================================================
# PREVIEWS VERTICALES 4X
#
# La franja tiene solo ~56-64 px de alto.
# La ampliamos únicamente para visualizarla.
# NEAREST evita introducir píxeles inventados.
# ============================================================

for filename in (
    "meteor_rgb_123_enhanced.png",
    "meteor_rgb_321_enhanced.png"
):

    source = (
        OUTPUT_DIR
        /
        filename
    )

    img = Image.open(
        source
    )


    preview = img.resize(
        (
            img.width,
            img.height * 4
        ),
        resample=Image.Resampling.NEAREST
    )


    preview_path = (
        OUTPUT_DIR
        /
        filename.replace(
            ".png",
            "_preview_4x.png"
        )
    )


    preview.save(
        preview_path
    )


# ============================================================
# ESTADÍSTICAS
# ============================================================

print()
print("================================================")
print("                   RESUMEN")
print("================================================")
print()

print(
    f"Canal 1 shift : "
    f"{shift_1:+d} px"
)

print(
    f"Canal 2 shift : "
    f"{shift_2:+d} px"
)

print(
    f"Canal 3 shift : "
    f"{shift_3:+d} px"
)

print()

print(
    f"RGB final     : "
    f"{width} x {height}"
)

print()

print(
    f"Stretch       : "
    f"P{LOW_PERCENTILE:.1f} - "
    f"P{HIGH_PERCENTILE:.1f}"
)

print(
    f"Gamma         : "
    f"{GAMMA:.2f}"
)

print()

print(
    "IMPORTANTE:"
)

print(
    "Las composiciones 123 y 321 son falso color "
    "exploratorio; todavía no son productos "
    "radiométricamente calibrados."
)

print()

print(
    "Archivos en:"
)

print(
    OUTPUT_DIR
)

print()
print("================================================")
print("FIN")
print("================================================")