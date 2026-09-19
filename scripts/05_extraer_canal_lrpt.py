from pathlib import Path
import wave

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfilt, resample_poly


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

SALIDA_DATOS = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_channel_288ksps.npy"
)

SALIDA_GRAFICO = (
    ROOT
    / "outputs"
    / "spectra"
    / "lrpt_channel_288ksps.png"
)


# --------------------------------------------------
# CONFIGURACION
# --------------------------------------------------

SEGUNDO_INICIO = 110

# Por ahora procesamos solo 10 segundos
DURACION = 10

# Ancho desde el centro:
# conservamos aproximadamente -70 kHz a +70 kHz
CUTOFF = 70_000

FS_SALIDA = 288_000


# --------------------------------------------------
# LEER I/Q
# --------------------------------------------------

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

I = iq[:, 0].astype(np.float32)
Q = iq[:, 1].astype(np.float32)

x = (
    I + 1j * Q
).astype(np.complex64)


print()
print("========================================")
print("       EXTRACCION DEL CANAL LRPT")
print("========================================")
print()

print(f"Sample rate original : {fs:,} Hz")
print(f"Duracion             : {DURACION} s")
print(f"Muestras originales  : {len(x):,}")
print(f"Filtro               : +/- {CUTOFF/1000:.0f} kHz")


# --------------------------------------------------
# QUITAR DC
# --------------------------------------------------

x = x - np.mean(x)


# --------------------------------------------------
# FILTRO PASA-BAJOS
# --------------------------------------------------

sos = butter(
    N=6,
    Wn=CUTOFF,
    btype="lowpass",
    fs=fs,
    output="sos"
)

x_filtrado = sosfilt(
    sos,
    x
)


# --------------------------------------------------
# RESAMPLEO
#
# 500000 * 72 / 125 = 288000
# --------------------------------------------------

x_288 = resample_poly(
    x_filtrado,
    up=72,
    down=125
)

x_288 = x_288.astype(
    np.complex64
)


print()
print(f"Sample rate nuevo    : {FS_SALIDA:,} Hz")
print(f"Muestras nuevas      : {len(x_288):,}")

print()
print(
    "Muestras por simbolo "
    f"a 72 ksym/s         : "
    f"{FS_SALIDA / 72_000:.2f}"
)


# --------------------------------------------------
# GUARDAR
# --------------------------------------------------

np.save(
    SALIDA_DATOS,
    x_288
)


# --------------------------------------------------
# ESPECTRO DEL CANAL RESULTANTE
# --------------------------------------------------

NFFT = 262144

bloque = x_288[
    :min(NFFT, len(x_288))
]

window = np.hanning(
    len(bloque)
)

X = np.fft.fftshift(
    np.fft.fft(
        bloque * window
    )
)

freq = np.fft.fftshift(
    np.fft.fftfreq(
        len(bloque),
        d=1 / FS_SALIDA
    )
)

power = 20 * np.log10(
    np.abs(X) + 1e-12
)

power -= np.max(power)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    freq / 1000,
    power,
    linewidth=0.8
)

plt.title(
    "Canal LRPT aislado - baseband"
)

plt.xlabel(
    "Frecuencia respecto a 137.100 MHz (kHz)"
)

plt.ylabel(
    "Potencia relativa (dB)"
)

plt.xlim(
    -144,
    144
)

plt.ylim(
    -80,
    5
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    SALIDA_GRAFICO,
    dpi=160
)

plt.show()


print()
print("Datos guardados:")
print(SALIDA_DATOS)

print()
print("Espectro guardado:")
print(SALIDA_GRAFICO)