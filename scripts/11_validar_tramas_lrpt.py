from pathlib import Path
import numpy as np


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

ARCHIVO = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_soft_symbols.npy"
)


# ============================================================
# CONFIGURACION
# ============================================================

SYNC_WORD = 0xFCA2B63DB00D9794

# Resultado obtenido en el paso 10
PRIMER_SYNC = 103_778

# METEOR LRPT clásico:
# 1024 bytes * 8 bits * rate 1/2
FRAME_SOFT_BITS = 16_384

MASK64 = 0xFFFFFFFFFFFFFFFF


# ============================================================
# FUNCIONES DE ROTACION DEL SYNC
# ============================================================

def rotate64(word, phase):

    i = word & 0xAAAAAAAAAAAAAAAA
    q = word & 0x5555555555555555

    if phase == 0:

        pass

    elif phase == 1:

        word = (
            ((i ^ 0xAAAAAAAAAAAAAAAA) >> 1)
            | ((q << 1) & MASK64)
        )

    elif phase == 2:

        word ^= MASK64

    elif phase == 3:

        word = (
            (i >> 1)
            |
            (((q ^ 0x5555555555555555) << 1) & MASK64)
        )

    return (
        ((word & 0x5555555555555555) << 1)
        |
        ((word & 0xAAAAAAAAAAAAAAAA) >> 1)
    ) & MASK64


def word_to_bits(word):

    return np.array(
        [
            (word >> (63 - i)) & 1
            for i in range(64)
        ],
        dtype=np.uint8
    )


# ============================================================
# CARGAR SOFT BITS
# ============================================================

soft = np.load(
    ARCHIVO
).astype(np.int8)

hard = (
    soft > 0
).astype(np.uint8)


print()
print("================================================")
print("        VALIDACION DE TRAMAS LRPT")
print("================================================")
print()

print(f"Soft bits disponibles : {len(soft):,}")
print(f"Primer sync conocido  : {PRIMER_SYNC:,}")
print(f"Longitud trama        : {FRAME_SOFT_BITS:,} soft bits")


# ============================================================
# SYNC ROT_90
# ============================================================

sync_rot90 = rotate64(
    SYNC_WORD,
    1
)

pattern = word_to_bits(
    sync_rot90
)

print()
print(
    f"Sync ROT_90          : "
    f"0x{sync_rot90:016X}"
)


# ============================================================
# MEDIR CORRELACION
# ============================================================

def correlation_at(position):

    if (
        position < 0
        or position + 64 > len(hard)
    ):
        return None

    block = hard[
        position:
        position + 64
    ]

    errors = np.count_nonzero(
        block != pattern
    )

    return (
        64 - errors,
        errors
    )


# ============================================================
# COMPROBAR TRAMAS ANTERIORES Y SIGUIENTES
# ============================================================

print()
print("================================================")
print("   SYNC ESPERADO CADA 16384 SOFT BITS")
print("================================================")
print()

resultados = []


for n in range(-8, 15):

    expected = (
        PRIMER_SYNC
        + n * FRAME_SOFT_BITS
    )

    if (
        expected < 0
        or expected + 64 > len(hard)
    ):
        continue


    # Buscamos cerca del lugar esperado.
    # Permitimos una desviacion de +/- 30 bits.
    best_corr = -1
    best_error = None
    best_pos = None

    for delta in range(-30, 31, 2):

        pos = expected + delta

        result = correlation_at(
            pos
        )

        if result is None:
            continue

        corr, errors = result

        if corr > best_corr:

            best_corr = corr
            best_error = errors
            best_pos = pos


    desplazamiento = (
        best_pos - expected
    )

    fuerte = (
        best_corr > 45
    )

    resultados.append(
        (
            n,
            expected,
            best_pos,
            best_corr,
            best_error,
            desplazamiento,
            fuerte
        )
    )

    estado = (
        "SYNC"
        if fuerte
        else "----"
    )

    print(
        f"Trama {n:+3d} | "
        f"esperado={expected:7d} | "
        f"encontrado={best_pos:7d} | "
        f"delta={desplazamiento:+3d} | "
        f"corr={best_corr:2d}/64 | "
        f"errores={best_error:2d} | "
        f"{estado}"
    )


# ============================================================
# RESUMEN
# ============================================================

validas = [
    r
    for r in resultados
    if r[-1]
]

print()
print("================================================")
print("                 RESUMEN")
print("================================================")
print()

print(
    f"Tramas comprobadas : {len(resultados)}"
)

print(
    f"Syncs fuertes      : {len(validas)}"
)

if len(validas) >= 2:

    posiciones = np.array(
        [
            r[2]
            for r in validas
        ],
        dtype=np.int64
    )

    diferencias = np.diff(
        posiciones
    )

    print()

    print(
        "Separaciones entre syncs:"
    )

    print(
        diferencias
    )

    if len(diferencias) > 0:

        print()

        print(
            f"Mediana separación : "
            f"{np.median(diferencias):.1f} soft bits"
        )


print()

if len(validas) >= 3:

    print(
        ">>> ESTRUCTURA DE TRAMAS LRPT CONFIRMADA <<<"
    )

else:

    print(
        "Todavia no hay suficientes syncs repetidos."
    )