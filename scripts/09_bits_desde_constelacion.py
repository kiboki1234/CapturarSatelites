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

SALIDA_BITS = (
    ROOT
    / "outputs"
    / "decoded"
    / "bitstream_prueba.bin"
)

SALIDA_TXT = (
    ROOT
    / "outputs"
    / "decoded"
    / "bitstream_prueba.txt"
)

SALIDA_PLOT = (
    ROOT
    / "outputs"
    / "plots"
    / "constelacion_decidida.png"
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
# FILTRO RRC
# --------------------------------------------------

def rrc_taps(alpha, sps, span):
    N = span * sps
    t = np.arange(-N, N + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 + alpha * (4 / np.pi - 1)

        elif alpha > 0 and abs(abs(ti) - 1 / (4 * alpha)) < 1e-10:
            h[i] = (
                alpha / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1 - alpha))
                + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            )
            denominator = (
                np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            )
            h[i] = numerator / denominator

    h /= np.sqrt(np.sum(h ** 2))
    return h


# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------

x = np.load(ARCHIVO).astype(np.complex64)
x = x[START:START + LEN]

x = x - np.mean(x)
x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)

h = rrc_taps(ALPHA, SPS, SPAN)
xf = np.convolve(x, h, mode="same")


# --------------------------------------------------
# BUSCAR MEJOR FASE
# --------------------------------------------------

best_phase = None
best_error = None
best_symbols = None

for phase in range(SPS):

    z = xf[phase::SPS]
    z = z[:50_000]

    # corrección fina por cuarta potencia
    y4 = z ** 4

    N = min(32768, len(y4))
    window = np.hanning(N)

    Y = np.fft.fftshift(np.fft.fft(y4[:N] * window))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1 / SYMBOL_RATE))

    idx = np.argmax(np.abs(Y))
    f4 = freq[idx]
    freq_offset = f4 / 4

    n = np.arange(len(z))
    z = z * np.exp(-1j * 2 * np.pi * freq_offset * n / SYMBOL_RATE)

    fourth = np.mean(z ** 4)
    phase_offset = np.angle(-fourth) / 4
    z = z * np.exp(-1j * phase_offset)

    z = z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-12)

    ideal = 1 / np.sqrt(2)
    error = np.mean(
        (np.abs(z.real) - ideal) ** 2
        + (np.abs(z.imag) - ideal) ** 2
    )

    if best_error is None or error < best_error:
        best_error = error
        best_phase = phase
        best_symbols = z


print()
print("======================================")
print("      PRIMEROS BITS DESDE QPSK")
print("======================================")
print()
print(f"Fase seleccionada: {best_phase}")
print(f"Error estimado  : {best_error:.6f}")


# --------------------------------------------------
# HARD DECISION POR CUADRANTE
# --------------------------------------------------
# Mapeo Gray simple:
# Q1 (+I,+Q) -> 00
# Q2 (-I,+Q) -> 01
# Q3 (-I,-Q) -> 11
# Q4 (+I,-Q) -> 10

bits = []

for s in best_symbols:
    i = s.real
    q = s.imag

    if i >= 0 and q >= 0:
        bits.extend([0, 0])
    elif i < 0 and q >= 0:
        bits.extend([0, 1])
    elif i < 0 and q < 0:
        bits.extend([1, 1])
    else:
        bits.extend([1, 0])

bits = np.array(bits, dtype=np.uint8)

print(f"Simbolos usados : {len(best_symbols):,}")
print(f"Bits generados  : {len(bits):,}")


# --------------------------------------------------
# GUARDAR COMO TEXTO Y BINARIO
# --------------------------------------------------

# txt con los primeros bits
primeros = "".join(str(b) for b in bits[:400])

with open(SALIDA_TXT, "w", encoding="utf-8") as f:
    f.write(primeros)

# empaquetar a bytes
pad = (-len(bits)) % 8
if pad:
    bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

bytes_out = np.packbits(bits)

with open(SALIDA_BITS, "wb") as f:
    f.write(bytes_out.tobytes())


print()
print("Primeros 400 bits:")
print(primeros)

print()
print("Archivos guardados:")
print(SALIDA_TXT)
print(SALIDA_BITS)


# --------------------------------------------------
# GRAFICA CON DECISION
# --------------------------------------------------

plot_symbols = best_symbols[:5000]

plt.figure(figsize=(7, 7))
plt.scatter(plot_symbols.real, plot_symbols.imag, s=4, alpha=0.25)

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.title("Constelacion con decision por cuadrantes")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(alpha=0.3)
plt.axis("equal")
plt.tight_layout()

plt.savefig(SALIDA_PLOT, dpi=160)
plt.show()

print()
print("Grafico guardado en:")
print(SALIDA_PLOT)