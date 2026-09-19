from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# RUTAS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

ARCHIVO = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_channel_288ksps_freqcorr.npy"
)

SALIDA = (
    ROOT
    / "outputs"
    / "plots"
    / "constelacion_simbolos.png"
)


# --------------------------------------------------
# CONFIGURACION
# --------------------------------------------------

FS = 288_000
SYMBOL_RATE = 72_000
SPS = 4

ALPHA = 0.6
SPAN = 8

START = 300_000
LEN = 500_000


# --------------------------------------------------
# FILTRO ROOT RAISED COSINE
# --------------------------------------------------

def rrc_taps(alpha, sps, span):

    N = span * sps

    t = np.arange(
        -N,
        N + 1,
        dtype=np.float64
    ) / sps

    h = np.zeros_like(t)

    for i, ti in enumerate(t):

        if abs(ti) < 1e-12:

            h[i] = (
                1
                + alpha
                * (4 / np.pi - 1)
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
                    (1 + 2 / np.pi)
                    * np.sin(
                        np.pi
                        / (4 * alpha)
                    )
                    +
                    (1 - 2 / np.pi)
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
                    * (1 - alpha)
                )
                +
                4
                * alpha
                * ti
                * np.cos(
                    np.pi
                    * ti
                    * (1 + alpha)
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

            h[i] = (
                numerator
                / denominator
            )

    # Normalizar energia
    h /= np.sqrt(
        np.sum(h ** 2)
    )

    return h


# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------

x = np.load(
    ARCHIVO
).astype(np.complex64)

x = x[
    START:
    START + LEN
]


print()
print("======================================")
print("   FILTRO RRC + TIMING DE SIMBOLOS")
print("======================================")
print()

print(f"Sample rate           : {FS:,} Hz")
print(f"Symbol rate           : {SYMBOL_RATE:,} sym/s")
print(f"Muestras/simbolo      : {SPS}")
print(f"RRC alpha             : {ALPHA}")
print(f"RRC span              : {SPAN} simbolos")


# --------------------------------------------------
# NORMALIZAR
# --------------------------------------------------

x = x - np.mean(x)

rms = np.sqrt(
    np.mean(
        np.abs(x) ** 2
    )
)

x = x / (
    rms + 1e-12
)


# --------------------------------------------------
# FILTRO ADAPTADO RRC
# --------------------------------------------------

h = rrc_taps(
    ALPHA,
    SPS,
    SPAN
)

xf = np.convolve(
    x,
    h,
    mode="same"
)


# --------------------------------------------------
# PROBAR LAS 4 FASES DE MUESTREO
# --------------------------------------------------

best_phase = None
best_error = None
best_symbols = None
best_freq = None


for phase in range(SPS):

    z = xf[
        phase::SPS
    ]

    # usamos un bloque razonable
    z = z[:50_000]

    # ------------------------------
    # CORRECCION FINA DE FRECUENCIA
    # usando cuarta potencia
    # ------------------------------

    y4 = z ** 4

    N = min(
        32768,
        len(y4)
    )

    window = np.hanning(N)

    Y = np.fft.fftshift(
        np.fft.fft(
            y4[:N] * window
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

    f4 = freq[idx]

    freq_offset = (
        f4 / 4
    )

    n = np.arange(
        len(z)
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


    # ------------------------------
    # CORRECCION DE FASE
    # ------------------------------

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
            -1j
            * phase_offset
        )
    )


    # ------------------------------
    # NORMALIZAR
    # ------------------------------

    zrms = np.sqrt(
        np.mean(
            np.abs(z) ** 2
        )
    )

    z = z / (
        zrms + 1e-12
    )


    # ------------------------------
    # SCORE QPSK
    # Cuanto mas cerca estén de
    # +/-1/sqrt(2), mejor.
    # ------------------------------

    ideal = (
        1
        / np.sqrt(2)
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
        f"Fase {phase}: "
        f"offset fino={freq_offset:8.2f} Hz | "
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


# --------------------------------------------------
# RESULTADO
# --------------------------------------------------

print()
print("======================================")
print("        MEJOR FASE ENCONTRADA")
print("======================================")
print()

print(
    f"Fase seleccionada     : {best_phase}"
)

print(
    f"Offset fino           : {best_freq:.2f} Hz"
)

print(
    f"Error QPSK            : {best_error:.6f}"
)


# --------------------------------------------------
# PLOT
# --------------------------------------------------

plot_symbols = (
    best_symbols[
        :20_000:3
    ]
)

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    plot_symbols.real,
    plot_symbols.imag,
    s=4,
    alpha=0.25
)

plt.title(
    "Constelacion tras filtro RRC y sincronizacion"
)

plt.xlabel("I")
plt.ylabel("Q")

plt.grid(
    alpha=0.3
)

plt.axis("equal")

plt.tight_layout()

plt.savefig(
    SALIDA,
    dpi=160
)

plt.show()


print()
print("Grafico guardado en:")
print(SALIDA)