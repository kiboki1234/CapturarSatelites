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
    / "lrpt_channel_288ksps_freqcorr.npy"
)

SALIDA_SOFT_NPY = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_soft_symbols.npy"
)

SALIDA_SOFT_BIN = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_soft_symbols.bin"
)


# ============================================================
# PARAMETROS LRPT
# ============================================================

FS = 288_000
SYMBOL_RATE = 72_000
SPS = 4

# SatDump usa alpha = 0.5 para METEOR M2 LRPT 72k
RRC_ALPHA = 0.5
RRC_TAPS = 31

# Trabajamos con una sección grande,
# pero no con los 10 segundos completos para ahorrar RAM.
START = 200_000
MAX_SYMBOLS = None

# Sync usado por el decoder clásico METEOR LRPT
SYNC_WORD = 0xFCA2B63DB00D9794

MASK64 = 0xFFFFFFFFFFFFFFFF


# ============================================================
# ROOT RAISED COSINE
# ============================================================

def rrc_taps(alpha, sps, num_taps):

    if num_taps % 2 == 0:
        raise ValueError("num_taps debe ser impar")

    half = num_taps // 2

    t = (
        np.arange(-half, half + 1, dtype=np.float64)
        / sps
    )

    h = np.zeros_like(t)

    for i, ti in enumerate(t):

        if abs(ti) < 1e-12:

            h[i] = (
                1
                + alpha * (4 / np.pi - 1)
            )

        elif (
            alpha > 0
            and abs(
                abs(ti) - 1 / (4 * alpha)
            ) < 1e-10
        ):

            h[i] = (
                alpha
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi)
                    * np.sin(np.pi / (4 * alpha))
                    +
                    (1 - 2 / np.pi)
                    * np.cos(np.pi / (4 * alpha))
                )
            )

        else:

            numerator = (
                np.sin(
                    np.pi * ti * (1 - alpha)
                )
                +
                4
                * alpha
                * ti
                * np.cos(
                    np.pi * ti * (1 + alpha)
                )
            )

            denominator = (
                np.pi
                * ti
                * (
                    1
                    - (4 * alpha * ti) ** 2
                )
            )

            h[i] = numerator / denominator

    h /= np.sqrt(
        np.sum(h ** 2)
    )

    return h


# ============================================================
# FUNCIONES QUE REPRODUCEN LAS VARIANTES DEL CORRELADOR
# DE SATDUMP
# ============================================================

def swap_iq_word(word):

    i = word & 0xAAAAAAAAAAAAAAAA
    q = word & 0x5555555555555555

    return (
        (i >> 1)
        | ((q << 1) & MASK64)
    )


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

    else:

        raise ValueError("Fase invalida")

    # Igual que el correlador de SatDump:
    # intercambio final de I/Q dentro de cada pareja
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


def build_sync_variants(sync):

    variants = []

    # Primeras 4 rotaciones
    for phase in range(4):

        variants.append(
            (
                f"ROT_{phase * 90}",
                rotate64(sync, phase)
            )
        )

    # Variantes con intercambio / inversión
    swapped = (
        swap_iq_word(sync)
        ^ MASK64
    )

    for phase in range(4):

        variants.append(
            (
                f"SWAP_ROT_{phase * 90}",
                rotate64(swapped, phase)
            )
        )

    return variants


# ============================================================
# CARGAR SEÑAL
# ============================================================

print()
print("================================================")
print("        BUSQUEDA DE SINCRONISMO LRPT")
print("================================================")
print()

x = np.load(
    ARCHIVO
).astype(np.complex64)

print("Archivo:")
print(ARCHIVO)

print()
print(f"Muestras disponibles : {len(x):,}")


# ============================================================
# QUITAR DC / NORMALIZAR
# ============================================================

x = x - np.mean(x)

rms = np.sqrt(
    np.mean(np.abs(x) ** 2)
)

x = x / (
    rms + 1e-12
)


# ============================================================
# RRC EXACTO DEL PIPELINE
# ============================================================

h = rrc_taps(
    RRC_ALPHA,
    SPS,
    RRC_TAPS
)

xf = np.convolve(
    x,
    h,
    mode="same"
)


# ============================================================
# BUSCAR MEJOR FASE TEMPORAL
# ============================================================

best_phase = None
best_error = None
best_symbols = None
best_freq = None


for phase in range(SPS):

    z = xf[
        START + phase::SPS
    ]

    z = z[
        :MAX_SYMBOLS
    ]

    # --------------------------------------------------------
    # CORRECCION FINA DE FRECUENCIA CON x^4
    # --------------------------------------------------------

    y4 = z ** 4

    N = min(
        65536,
        len(y4)
    )

    Y = np.fft.fftshift(
        np.fft.fft(
            y4[:N]
            * np.hanning(N)
        )
    )

    freq = np.fft.fftshift(
        np.fft.fftfreq(
            N,
            d=1 / SYMBOL_RATE
        )
    )

    idx = np.argmax(
        np.abs(Y)
    )

    freq_offset = (
        freq[idx]
        / 4
    )

    n = np.arange(
        len(z),
        dtype=np.float64
    )

    z = (
        z
        * np.exp(
            -1j
            * 2
            * np.pi
            * freq_offset
            * n
            / SYMBOL_RATE
        )
    )


    # --------------------------------------------------------
    # ALINEACION DE FASE
    # --------------------------------------------------------

    fourth = np.mean(
        z ** 4
    )

    phase_offset = (
        np.angle(
            -fourth
        )
        / 4
    )

    z = (
        z
        * np.exp(
            -1j * phase_offset
        )
    )


    # --------------------------------------------------------
    # NORMALIZAR
    # --------------------------------------------------------

    zrms = np.sqrt(
        np.mean(
            np.abs(z) ** 2
        )
    )

    z = z / (
        zrms + 1e-12
    )


    # --------------------------------------------------------
    # MEDIDA DE CALIDAD QPSK
    # --------------------------------------------------------

    ideal = (
        1 / np.sqrt(2)
    )

    error = np.mean(
        (
            np.abs(z.real)
            - ideal
        ) ** 2
        +
        (
            np.abs(z.imag)
            - ideal
        ) ** 2
    )

    print(
        f"Fase {phase} | "
        f"offset={freq_offset:9.3f} Hz | "
        f"error={error:.6f}"
    )

    if (
        best_error is None
        or error < best_error
    ):

        best_error = error
        best_phase = phase
        best_symbols = z
        best_freq = freq_offset


print()
print("-----------------------------------------------")
print(f"Mejor fase temporal : {best_phase}")
print(f"Offset fino         : {best_freq:.3f} Hz")
print(f"Error QPSK          : {best_error:.6f}")
print("-----------------------------------------------")


# ============================================================
# CONVERTIR LOS SIMBOLOS EN SOFT VALUES
#
# SatDump trabaja esencialmente con una secuencia:
#
# I, Q, I, Q, I, Q...
#
# Los signos dan los bits duros.
# La magnitud conserva confianza para Viterbi.
# ============================================================

soft_float = np.empty(
    len(best_symbols) * 2,
    dtype=np.float32
)

soft_float[0::2] = (
    best_symbols.real
)

soft_float[1::2] = (
    best_symbols.imag
)


# Escalado robusto a int8
p99 = np.percentile(
    np.abs(soft_float),
    99
)

scale = (
    120.0
    / (p99 + 1e-12)
)

soft_i8 = np.clip(
    np.rint(
        soft_float * scale
    ),
    -127,
    127
).astype(np.int8)


np.save(
    SALIDA_SOFT_NPY,
    soft_i8
)

soft_i8.tofile(
    SALIDA_SOFT_BIN
)


print()
print(f"Simbolos QPSK       : {len(best_symbols):,}")
print(f"Soft values I/Q     : {len(soft_i8):,}")
print(f"Escala int8         : {scale:.3f}")


# ============================================================
# HARD BITS PARA CORRELACION
# ============================================================

# SatDump considera:
# soft > 0  --> bit 1
# soft <= 0 --> bit 0

hard_bits = (
    soft_i8 > 0
).astype(np.uint8)


# ============================================================
# CREAR VENTANAS DE 64 BITS
#
# En QPSK SatDump avanza de 2 en 2 bits.
# ============================================================

windows = np.lib.stride_tricks.sliding_window_view(
    hard_bits,
    64
)

windows = windows[
    ::2
]


# ============================================================
# PROBAR 8 AMBIGUEDADES QPSK
# ============================================================

variants = build_sync_variants(
    SYNC_WORD
)

results = []


print()
print("================================================")
print("          CORRELACION DE SYNC WORD")
print("================================================")
print()

for name, word in variants:

    pattern = word_to_bits(
        word
    )

    distances = np.count_nonzero(
        windows != pattern,
        axis=1
    )

    idx = int(
        np.argmin(distances)
    )

    distance = int(
        distances[idx]
    )

    correlation = (
        64 - distance
    )

    bit_position = (
        idx * 2
    )

    symbol_position = (
        bit_position // 2
    )

    results.append(
        (
            correlation,
            distance,
            bit_position,
            symbol_position,
            name,
            word
        )
    )

    print(
        f"{name:15s} | "
        f"cor={correlation:2d}/64 | "
        f"errores={distance:2d} | "
        f"bit={bit_position:7d} | "
        f"simbolo={symbol_position:7d}"
    )


# ============================================================
# MEJOR RESULTADO
# ============================================================

results.sort(
    reverse=True,
    key=lambda x: x[0]
)

best = results[0]

(
    correlation,
    distance,
    bit_position,
    symbol_position,
    variant_name,
    variant_word
) = best


print()
print("================================================")
print("             MEJOR CANDIDATO")
print("================================================")
print()

print(
    f"Variante              : {variant_name}"
)

print(
    f"Sync transformado     : 0x{variant_word:016X}"
)

print(
    f"Correlacion           : {correlation}/64"
)

print(
    f"Bits distintos        : {distance}/64"
)

print(
    f"Posicion en soft bits : {bit_position:,}"
)

print(
    f"Posicion simbolo      : {symbol_position:,}"
)


# SatDump considera >45 una coincidencia fuerte
if correlation > 45:

    print()
    print(
        ">>> SINCRONISMO LRPT ENCONTRADO <<<"
    )

else:

    print()
    print(
        "No hay una coincidencia suficientemente fuerte."
    )

    print(
        "Todavia NO seguimos a Viterbi."
    )


print()
print("Soft symbols guardados en:")
print(SALIDA_SOFT_NPY)

print()
print(SALIDA_SOFT_BIN)