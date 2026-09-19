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
    / "lrpt_channel_288ksps.npy"
)

SALIDA_NPY = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_channel_288ksps_freqcorr.npy"
)

SALIDA_PLOT = (
    ROOT
    / "outputs"
    / "plots"
    / "constelacion_freqcorr.png"
)


# --------------------------------------------------
# CONFIGURACION
# --------------------------------------------------

FS = 288_000  # Hz

# segmento para estimar frecuencia
EST_START = 200_000
EST_LEN = 400_000

# segmento para graficar constelacion
PLOT_START = 300_000
PLOT_LEN = 120_000
PLOT_STEP = 8


# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------

x = np.load(ARCHIVO).astype(np.complex64)

print()
print("======================================")
print("   CORRECCION GRUESA DE FRECUENCIA")
print("======================================")
print()

print("Archivo cargado:")
print(ARCHIVO)
print()
print(f"Muestras totales: {len(x):,}")


# --------------------------------------------------
# PREPROCESAMIENTO
# --------------------------------------------------

# quitar DC
x = x - np.mean(x)

# trozo para estimacion
xe = x[EST_START:EST_START + EST_LEN]

print(f"Segmento de estimacion: inicio={EST_START:,}, longitud={len(xe):,}")


# --------------------------------------------------
# METODO DE LA CUARTA POTENCIA
# --------------------------------------------------

# Para QPSK/OQPSK, elevar a la 4ta elimina la modulación angular principal
y = xe ** 4

# ventana
window = np.hanning(len(y))

Y = np.fft.fftshift(
    np.fft.fft(y * window)
)

freq = np.fft.fftshift(
    np.fft.fftfreq(len(y), d=1 / FS)
)

power = np.abs(Y)

peak_idx = np.argmax(power)
freq_peak_4x = freq[peak_idx]

# offset estimado en la señal original
freq_offset = freq_peak_4x / 4.0

print()
print(f"Pico en x^4             : {freq_peak_4x:.3f} Hz")
print(f"Offset estimado         : {freq_offset:.3f} Hz")


# --------------------------------------------------
# CORREGIR FRECUENCIA
# --------------------------------------------------

n = np.arange(len(x), dtype=np.float64)

correction = np.exp(
    -1j * 2 * np.pi * freq_offset * n / FS
)

x_corr = x * correction
x_corr = x_corr.astype(np.complex64)

np.save(SALIDA_NPY, x_corr)

print()
print("Archivo corregido guardado en:")
print(SALIDA_NPY)


# --------------------------------------------------
# CONSTELACION CORREGIDA
# --------------------------------------------------

trozo = x_corr[PLOT_START:PLOT_START + PLOT_LEN:PLOT_STEP]

rms = np.sqrt(np.mean(np.abs(trozo) ** 2))
if rms > 0:
    trozo = trozo / rms

I = trozo.real
Q = trozo.imag

print()
print(f"Muestras graficadas     : {len(trozo):,}")
print(f"RMS normalizacion       : {rms:.6f}")


plt.figure(figsize=(7, 7))
plt.scatter(I, Q, s=3, alpha=0.25)

plt.title("Constelacion tras correccion gruesa de frecuencia")
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