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
    / "spectra"
    / "espectro_segundo_120.png"
)


# --------------------------------------------------
# CONFIGURACION
# --------------------------------------------------

FRECUENCIA_CENTRAL = 137_100_000  # Hz
SEGUNDO = 120

# 262144 muestras ≈ 0.52 segundos a 500 kS/s
N = 262_144


# --------------------------------------------------
# LEER BLOQUE I/Q
# --------------------------------------------------

with wave.open(str(ARCHIVO), "rb") as w:

    fs = w.getframerate()

    inicio = int(
        SEGUNDO * fs
    )

    w.setpos(inicio)

    raw = w.readframes(N)


data = np.frombuffer(
    raw,
    dtype=np.int16
)

iq = data.reshape(-1, 2)

I = iq[:, 0].astype(np.float64)
Q = iq[:, 1].astype(np.float64)

x = I + 1j * Q


print()
print("======================================")
print("            ANALISIS FFT")
print("======================================")
print()

print(f"Segundo analizado     : {SEGUNDO}")
print(f"Sample rate           : {fs:,} Hz")
print(f"Muestras FFT          : {len(x):,}")

print()
print(f"I min/max             : {I.min()} / {I.max()}")
print(f"Q min/max             : {Q.min()} / {Q.max()}")
print(f"Desviacion I          : {np.std(I):.4f}")
print(f"Desviacion Q          : {np.std(Q):.4f}")


# --------------------------------------------------
# PREPROCESAMIENTO
# --------------------------------------------------

# Eliminar componente DC
x = x - np.mean(x)

# Ventana Hann para reducir leakage espectral
window = np.hanning(len(x))

x_windowed = x * window


# --------------------------------------------------
# FFT
# --------------------------------------------------

X = np.fft.fftshift(
    np.fft.fft(x_windowed)
)

freq_baseband = np.fft.fftshift(
    np.fft.fftfreq(
        len(x),
        d=1 / fs
    )
)

freq_rf = (
    FRECUENCIA_CENTRAL
    + freq_baseband
)


# Magnitud relativa en dB
power_db = 20 * np.log10(
    np.abs(X) + 1e-12
)

power_db -= np.max(power_db)


# --------------------------------------------------
# ENCONTRAR PICO PRINCIPAL
# --------------------------------------------------

peak_index = np.argmax(power_db)

peak_frequency = (
    freq_rf[peak_index]
)

print()
print(
    f"Pico mas fuerte       : "
    f"{peak_frequency / 1e6:.6f} MHz"
)


# --------------------------------------------------
# GRAFICA
# --------------------------------------------------

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    freq_rf / 1e6,
    power_db,
    linewidth=0.7
)

plt.title(
    f"Espectro I/Q - segundo {SEGUNDO}"
)

plt.xlabel(
    "Frecuencia RF (MHz)"
)

plt.ylabel(
    "Potencia relativa (dB)"
)

plt.grid(
    alpha=0.3
)

plt.xlim(
    (FRECUENCIA_CENTRAL - fs / 2) / 1e6,
    (FRECUENCIA_CENTRAL + fs / 2) / 1e6
)

plt.ylim(
    -100,
    5
)

plt.tight_layout()

plt.savefig(
    SALIDA,
    dpi=160
)

print()
print("Grafico guardado en:")
print(SALIDA)

plt.show()