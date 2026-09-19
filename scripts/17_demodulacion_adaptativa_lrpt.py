from pathlib import Path
import math
import numpy as np


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_channel_288ksps_freqcorr.npy"
)

OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_soft_symbols_adaptive.npy"
)

OUTPUT_BIN = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_soft_symbols_adaptive.bin"
)


# ============================================================
# PARÁMETROS METEOR LRPT
# ============================================================

FS = 288_000.0
SYMBOL_RATE = 72_000.0

SPS = (
    FS
    / SYMBOL_RATE
)

# RRC oficial del pipeline METEOR M2 LRPT 72k
RRC_ALPHA = 0.5
RRC_TAPS = 31


# ============================================================
# COSTAS LOOP
#
# SatDump METEOR:
# pll_bw = 0.002
# orden 4 = QPSK
# ============================================================

COSTAS_BW = 0.002

COSTAS_FREQ_LIMIT = 1.0


# ============================================================
# MUELLER & MULLER
#
# Valores por defecto de SatDump
# ============================================================

CLOCK_ALPHA = 8.7e-3

CLOCK_GAIN_MU = CLOCK_ALPHA

CLOCK_GAIN_OMEGA = (
    CLOCK_ALPHA ** 2
    / 4.0
)

CLOCK_MU_INITIAL = 0.5

CLOCK_OMEGA_REL_LIMIT = 0.005


# ============================================================
# ROOT RAISED COSINE
# ============================================================

def rrc_taps(
    alpha,
    sps,
    num_taps
):

    if num_taps % 2 == 0:
        raise ValueError(
            "num_taps debe ser impar"
        )

    half = (
        num_taps
        // 2
    )

    t = (
        np.arange(
            -half,
            half + 1,
            dtype=np.float64
        )
        / sps
    )

    h = np.zeros_like(
        t
    )

    for i, ti in enumerate(t):

        if abs(ti) < 1e-12:

            h[i] = (
                1
                +
                alpha
                * (
                    4 / np.pi
                    - 1
                )
            )

        elif (
            alpha > 0
            and abs(
                abs(ti)
                - 1 / (4 * alpha)
            ) < 1e-10
        ):

            h[i] = (
                alpha
                / np.sqrt(2)
                * (
                    (
                        1
                        + 2 / np.pi
                    )
                    * np.sin(
                        np.pi
                        / (4 * alpha)
                    )
                    +
                    (
                        1
                        - 2 / np.pi
                    )
                    * np.cos(
                        np.pi
                        / (4 * alpha)
                    )
                )
            )

        else:

            numerator = (
                np.sin(
                    np.pi
                    * ti
                    * (
                        1
                        - alpha
                    )
                )
                +
                4
                * alpha
                * ti
                * np.cos(
                    np.pi
                    * ti
                    * (
                        1
                        + alpha
                    )
                )
            )

            denominator = (
                np.pi
                * ti
                * (
                    1
                    -
                    (
                        4
                        * alpha
                        * ti
                    ) ** 2
                )
            )

            h[i] = (
                numerator
                / denominator
            )

    # energía unitaria
    h /= np.sqrt(
        np.sum(
            h ** 2
        )
    )

    return h


# ============================================================
# COSTAS LOOP QPSK
#
# Replica la estructura usada por SatDump:
#
# error =
# sign(I)*Q - sign(Q)*I
#
# freq  += beta*error
# phase += freq + alpha*error
# ============================================================

def costas_qpsk(
    x,
    loop_bw=0.002,
    freq_limit=1.0
):

    damping = (
        math.sqrt(2.0)
        / 2.0
    )

    denom = (
        1.0
        +
        2.0
        * damping
        * loop_bw
        +
        loop_bw
        * loop_bw
    )

    alpha = (
        4.0
        * damping
        * loop_bw
        / denom
    )

    beta = (
        4.0
        * loop_bw
        * loop_bw
        / denom
    )

    print()
    print(
        f"Costas alpha : {alpha:.8f}"
    )

    print(
        f"Costas beta  : {beta:.8f}"
    )

    out = np.empty(
        len(x),
        dtype=np.complex64
    )

    phase = 0.0
    freq = 0.0

    freq_history = []

    for i in range(
        len(x)
    ):

        # -----------------------------------------------
        # VCO / corrección de fase
        # -----------------------------------------------

        c = math.cos(
            -phase
        )

        s = math.sin(
            -phase
        )

        sample = (
            x[i]
            * complex(
                c,
                s
            )
        )

        out[i] = sample

        real = float(
            sample.real
        )

        imag = float(
            sample.imag
        )

        # -----------------------------------------------
        # Detector QPSK
        # -----------------------------------------------

        sign_i = (
            1.0
            if real > 0
            else -1.0
        )

        sign_q = (
            1.0
            if imag > 0
            else -1.0
        )

        error = (
            sign_i
            * imag
            -
            sign_q
            * real
        )

        # SatDump clampa error
        if error > 1.0:
            error = 1.0

        elif error < -1.0:
            error = -1.0

        # -----------------------------------------------
        # Loop filter
        # -----------------------------------------------

        freq += (
            beta
            * error
        )

        if freq > freq_limit:
            freq = freq_limit

        elif freq < -freq_limit:
            freq = -freq_limit

        phase += (
            freq
            +
            alpha
            * error
        )

        # wrap
        if phase > (
            2.0
            * math.pi
        ):

            phase -= (
                2.0
                * math.pi
            )

        elif phase < (
            -2.0
            * math.pi
        ):

            phase += (
                2.0
                * math.pi
            )

        # Guardar diagnóstico cada segundo aprox.
        if i % 288_000 == 0:

            freq_hz = (
                freq
                * FS
                / (
                    2.0
                    * math.pi
                )
            )

            freq_history.append(
                freq_hz
            )

    return (
        out,
        freq_history
    )


# ============================================================
# INTERPOLACIÓN CÚBICA
#
# El M&M original de SatDump usa banco polifásico.
#
# Aquí usamos interpolación cúbica porque estamos haciendo
# la implementación didáctica en Python.
# ============================================================

def cubic_interpolate(
    x,
    t
):

    i = int(
        math.floor(
            t
        )
    )

    mu = (
        t - i
    )

    if i < 1:

        i = 1

    if i + 2 >= len(x):

        i = (
            len(x)
            - 3
        )

    y0 = x[
        i - 1
    ]

    y1 = x[
        i
    ]

    y2 = x[
        i + 1
    ]

    y3 = x[
        i + 2
    ]

    # Catmull-Rom
    a0 = (
        -0.5 * y0
        + 1.5 * y1
        - 1.5 * y2
        + 0.5 * y3
    )

    a1 = (
        y0
        - 2.5 * y1
        + 2.0 * y2
        - 0.5 * y3
    )

    a2 = (
        -0.5 * y0
        + 0.5 * y2
    )

    a3 = y1

    return (
        (
            (
                a0
                * mu
                + a1
            )
            * mu
            + a2
        )
        * mu
        + a3
    )


# ============================================================
# MUELLER & MULLER CLOCK RECOVERY
#
# Sigue la lógica del M&M de SatDump:
#
# p_2T / p_1T / p_0T
# c_2T / c_1T / c_0T
#
# phase_error =
#
# real[
#   (p0-p2)*conj(c1)
#   -
#   (c0-c2)*conj(p1)
# ]
# ============================================================

def mm_clock_recovery(
    x,
    omega_mid=4.0,
    gain_omega=CLOCK_GAIN_OMEGA,
    mu_initial=CLOCK_MU_INITIAL,
    gain_mu=CLOCK_GAIN_MU,
    omega_relative_limit=CLOCK_OMEGA_REL_LIMIT
):

    omega = float(
        omega_mid
    )

    omega_limit = (
        omega_relative_limit
        * omega_mid
    )

    mu = float(
        mu_initial
    )

    p_2T = 0j
    p_1T = 0j
    p_0T = 0j

    c_2T = 0j
    c_1T = 0j
    c_0T = 0j

    # Estimación de máximo
    expected = int(
        len(x)
        / omega_mid
        * 1.01
    )

    out = np.empty(
        expected,
        dtype=np.complex64
    )

    omega_history = []

    out_index = 0

    position = 2.0

    while (
        position
        < len(x) - 3
        and out_index < expected
    ):

        # -----------------------------------------------
        # Historial
        # -----------------------------------------------

        p_2T = p_1T
        p_1T = p_0T

        c_2T = c_1T
        c_1T = c_0T

        # -----------------------------------------------
        # Interpolación
        # -----------------------------------------------

        p_0T = cubic_interpolate(
            x,
            position
        )

        # SatDump usa 1/0 para el slicer del M&M
        c_0T = complex(
            1.0
            if p_0T.real > 0
            else 0.0,

            1.0
            if p_0T.imag > 0
            else 0.0
        )

        # -----------------------------------------------
        # M&M error
        # -----------------------------------------------

        term1 = (
            (
                p_0T
                - p_2T
            )
            * np.conj(
                c_1T
            )
        )

        term2 = (
            (
                c_0T
                - c_2T
            )
            * np.conj(
                p_1T
            )
        )

        phase_error = float(
            (
                term1
                - term2
            ).real
        )

        # Clamp igual que SatDump
        if phase_error > 1.0:

            phase_error = 1.0

        elif phase_error < -1.0:

            phase_error = -1.0

        # -----------------------------------------------
        # Salida
        # -----------------------------------------------

        out[
            out_index
        ] = p_0T

        out_index += 1

        # -----------------------------------------------
        # Ajustar omega
        # -----------------------------------------------

        omega += (
            gain_omega
            * phase_error
        )

        min_omega = (
            omega_mid
            - omega_limit
        )

        max_omega = (
            omega_mid
            + omega_limit
        )

        if omega < min_omega:

            omega = min_omega

        elif omega > max_omega:

            omega = max_omega

        # -----------------------------------------------
        # Ajustar mu
        # -----------------------------------------------

        mu += (
            omega
            +
            gain_mu
            * phase_error
        )

        advance = int(
            math.floor(
                mu
            )
        )

        position += advance

        mu -= advance

        if (
            out_index
            % 72_000
            == 0
        ):

            omega_history.append(
                omega
            )

    out = out[
        :out_index
    ]

    return (
        out,
        omega_history
    )


# ============================================================
# MAIN
# ============================================================

print()
print(
    "================================================"
)

print(
    " DEMODULACIÓN ADAPTATIVA METEOR LRPT"
)

print(
    "================================================"
)

print()

print(
    "Archivo:"
)

print(
    INPUT_FILE
)

print()


# ============================================================
# CARGAR
# ============================================================

x = np.load(
    INPUT_FILE
).astype(
    np.complex64
)

print(
    f"Muestras entrada : "
    f"{len(x):,}"
)

print(
    f"Duración         : "
    f"{len(x) / FS:.3f} s"
)


# ============================================================
# QUITAR DC
# ============================================================

x = (
    x
    - np.mean(
        x
    )
)


# ============================================================
# NORMALIZAR
# ============================================================

rms = np.sqrt(
    np.mean(
        np.abs(x) ** 2
    )
)

x = (
    x
    / (
        rms
        + 1e-12
    )
)


# ============================================================
# RRC
# ============================================================

print()
print(
    "1/4 RRC..."
)

h = rrc_taps(
    RRC_ALPHA,
    SPS,
    RRC_TAPS
)

x_rrc = np.convolve(
    x,
    h,
    mode="same"
).astype(
    np.complex64
)


# ============================================================
# AGC / NORMALIZACIÓN
# ============================================================

rms_rrc = np.sqrt(
    np.mean(
        np.abs(
            x_rrc
        ) ** 2
    )
)

x_rrc /= (
    rms_rrc
    + 1e-12
)


# ============================================================
# COSTAS
# ============================================================

print(
    "2/4 Costas QPSK..."
)

x_costas, freq_history = (
    costas_qpsk(
        x_rrc,
        loop_bw=COSTAS_BW,
        freq_limit=COSTAS_FREQ_LIMIT
    )
)


# ============================================================
# M&M
# ============================================================

print(
    "3/4 Recuperación de reloj M&M..."
)

symbols, omega_history = (
    mm_clock_recovery(
        x_costas,
        omega_mid=SPS
    )
)


print()

print(
    f"Símbolos recuperados : "
    f"{len(symbols):,}"
)

print(
    f"Duración equivalente : "
    f"{len(symbols) / SYMBOL_RATE:.3f} s"
)


# ============================================================
# ELIMINAR TRANSITORIO INICIAL
#
# Muy pequeño; no queremos perder datos innecesariamente.
# ============================================================

TRANSIENT = 100

if len(symbols) > TRANSIENT:

    symbols = symbols[
        TRANSIENT:
    ]


# ============================================================
# NORMALIZACIÓN FINAL
# ============================================================

sym_rms = np.sqrt(
    np.mean(
        np.abs(
            symbols
        ) ** 2
    )
)

symbols = (
    symbols
    / (
        sym_rms
        + 1e-12
    )
)


# ============================================================
# COMPLEX -> SOFT I/Q
# ============================================================

soft_float = np.empty(
    len(symbols) * 2,
    dtype=np.float32
)

soft_float[
    0::2
] = symbols.real

soft_float[
    1::2
] = symbols.imag


# ============================================================
# ESCALADO INT8
# ============================================================

p99 = np.percentile(
    np.abs(
        soft_float
    ),
    99
)

scale = (
    120.0
    / (
        p99
        + 1e-12
    )
)

soft_i8 = np.clip(
    np.rint(
        soft_float
        * scale
    ),
    -127,
    127
).astype(
    np.int8
)


# ============================================================
# GUARDAR
# ============================================================

np.save(
    OUTPUT_FILE,
    soft_i8
)

soft_i8.tofile(
    OUTPUT_BIN
)


# ============================================================
# DIAGNÓSTICO
# ============================================================

print()
print(
    "4/4 Guardando soft bits..."
)

print()

print(
    f"Soft values : "
    f"{len(soft_i8):,}"
)

print(
    f"Escala      : "
    f"{scale:.3f}"
)

print()

if freq_history:

    print(
        "Costas frecuencia estimada por segundo:"
    )

    for i, f in enumerate(
        freq_history
    ):

        print(
            f"  {i:02d}s : "
            f"{f:+.3f} Hz"
        )


if omega_history:

    print()

    print(
        "M&M omega aproximado por segundo:"
    )

    for i, omega in enumerate(
        omega_history
    ):

        ppm = (
            (
                omega
                / SPS
                - 1.0
            )
            * 1e6
        )

        print(
            f"  {i + 1:02d}s : "
            f"{omega:.7f} samples/symbol "
            f"({ppm:+.2f} ppm)"
        )


print()

print(
    "Archivo generado:"
)

print(
    OUTPUT_FILE
)

print()

print(
    "================================================"
)

print(
    "FIN"
)

print(
    "================================================"
)