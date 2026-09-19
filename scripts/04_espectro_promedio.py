from pathlib import Path
import wave
import numpy as np
import matplotlib.pyplot as plt


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
    / "espectro_promedio_137100.png"
)


FRECUENCIA_CENTRAL = 137_100_000

SEGUNDO_INICIO = 100
DURACION = 40

NFFT = 8192
HOP = 4096


with wave.open(str(ARCHIVO), "rb") as w:

    fs = w.getframerate()

    w.setpos(
        int(SEGUNDO_INICIO * fs)
    )

    raw = w.readframes(
        int(DURACION * fs)
    )


data = np.frombuffer(
    raw,
    dtype=np.int16
)

iq = data.reshape(-1, 2)

I = iq[:, 0].astype(np.float64)
Q = iq[:, 1].astype(np.float64)

x = I + 1j * Q

x -= np.mean(x)


window = np.hanning(NFFT)

spectra = []


for start in range(
    0,
    len(x) - NFFT,
    HOP
):

    bloque = (
        x[start:start + NFFT]
        * window
    )

    X = np.fft.fftshift(
        np.fft.fft(bloque)
    )

    power = (
        np.abs(X) ** 2
    )

    spectra.append(power)


spectra = np.array(spectra)

# Promedio temporal
mean_power = np.mean(
    spectra,
    axis=0
)

power_db = 10 * np.log10(
    mean_power + 1e-15
)

power_db -= np.max(power_db)


freq_baseband = np.fft.fftshift(
    np.fft.fftfreq(
        NFFT,
        d=1/fs
    )
)

freq_rf = (
    FRECUENCIA_CENTRAL
    + freq_baseband
)


# --------------------------------------------------
# SOLO MOSTRAR ZONA INTERESANTE
# --------------------------------------------------

fmin = 136_980_000
fmax = 137_180_000

mask = (
    (freq_rf >= fmin)
    &
    (freq_rf <= fmax)
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    freq_rf[mask] / 1e6,
    power_db[mask],
    linewidth=1
)

plt.title(
    "Espectro promedio - posible señal LRPT"
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
    fmin / 1e6,
    fmax / 1e6
)

plt.tight_layout()

plt.savefig(
    SALIDA,
    dpi=160
)

plt.show()


print()
print("======================================")
print("       ESPECTRO PROMEDIO")
print("======================================")
print()

print(
    f"Rango analizado: "
    f"{fmin/1e6:.3f} - "
    f"{fmax/1e6:.3f} MHz"
)

print()
print("Grafico guardado en:")
print(SALIDA)