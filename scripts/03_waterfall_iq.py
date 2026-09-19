from pathlib import Path
import wave
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# RUTAS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

ARCHIVO = (
    ROOT
    / "data"
    / "raw"
    / "meteor_137100"
    / "meteor_137100_iq.wav"
)

SALIDA = (
    ROOT
    / "outputs"
    / "waterfalls"
    / "waterfall_137100.png"
)


# --------------------------------------------------
# CONFIGURACION
# --------------------------------------------------

FRECUENCIA_CENTRAL = 137_100_000  # Hz

SEGUNDO_INICIO = 100
DURACION_SEGUNDOS = 40

NFFT = 4096
HOP = 2048


# --------------------------------------------------
# LEER BLOQUE COMPLETO
# --------------------------------------------------

with wave.open(str(ARCHIVO), "rb") as w:

    fs = w.getframerate()

    frame_inicio = int(SEGUNDO_INICIO * fs)
    num_frames = int(DURACION_SEGUNDOS * fs)

    w.setpos(frame_inicio)

    raw = w.readframes(num_frames)


data = np.frombuffer(
    raw,
    dtype=np.int16
)

iq = data.reshape(-1, 2)

I = iq[:, 0].astype(np.float64)
Q = iq[:, 1].astype(np.float64)

x = I + 1j * Q

x = x - np.mean(x)


print()
print("======================================")
print("             WATERFALL")
print("======================================")
print()

print(f"Segundo inicio       : {SEGUNDO_INICIO}")
print(f"Duracion analizada   : {DURACION_SEGUNDOS} s")
print(f"Sample rate          : {fs:,} Hz")
print(f"Muestras totales     : {len(x):,}")
print(f"NFFT                 : {NFFT}")
print(f"HOP                  : {HOP}")


# --------------------------------------------------
# STFT MANUAL
# --------------------------------------------------

window = np.hanning(NFFT)

spectra = []

for start in range(0, len(x) - NFFT, HOP):
    frame = x[start:start + NFFT]
    frame = frame * window

    X = np.fft.fftshift(np.fft.fft(frame))
    P = 20 * np.log10(np.abs(X) + 1e-12)

    spectra.append(P)

S = np.array(spectra)

# Normalización visual
S = S - np.max(S)

freq_baseband = np.fft.fftshift(
    np.fft.fftfreq(NFFT, d=1/fs)
)

freq_rf = FRECUENCIA_CENTRAL + freq_baseband

tiempo = np.arange(S.shape[0]) * (HOP / fs) + SEGUNDO_INICIO


# --------------------------------------------------
# GRAFICA
# --------------------------------------------------

plt.figure(figsize=(12, 7))

plt.imshow(
    S,
    aspect="auto",
    origin="lower",
    extent=[
        freq_rf[0] / 1e6,
        freq_rf[-1] / 1e6,
        tiempo[0],
        tiempo[-1]
    ],
    cmap="viridis",
    vmin=-80,
    vmax=0
)

plt.colorbar(label="Potencia relativa (dB)")

plt.title("Waterfall de la grabacion I/Q")
plt.xlabel("Frecuencia RF (MHz)")
plt.ylabel("Tiempo (s)")

plt.tight_layout()
plt.savefig(SALIDA, dpi=160)

print()
print("Waterfall guardado en:")
print(SALIDA)

plt.show()