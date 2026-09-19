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

SALIDA = (
    ROOT
    / "outputs"
    / "plots"
    / "constelacion_bruta.png"
)


# --------------------------------------------------
# CARGAR DATOS
# --------------------------------------------------

x = np.load(ARCHIVO)

print()
print("======================================")
print("        CONSTELACION BRUTA")
print("======================================")
print()

print("Archivo cargado:")
print(ARCHIVO)
print()
print(f"Muestras totales: {len(x):,}")


# --------------------------------------------------
# TOMAR UNA SUBMUESTRA PARA PLOTEAR
# --------------------------------------------------

# saltamos un poco al inicio por seguridad
inicio = 200_000

# cantidad a visualizar
cantidad = 120_000

# submuestreo visual para no saturar la grafica
step = 8

trozo = x[inicio:inicio + cantidad:step]

# normalizar amplitud
escala = np.sqrt(np.mean(np.abs(trozo) ** 2))
if escala > 0:
    trozo = trozo / escala

I = trozo.real
Q = trozo.imag

print(f"Muestras graficadas: {len(trozo):,}")
print(f"RMS de normalizacion: {escala:.6f}")


# --------------------------------------------------
# GRAFICA DE CONSTELACION
# --------------------------------------------------

plt.figure(figsize=(7, 7))
plt.scatter(I, Q, s=3, alpha=0.25)

plt.title("Constelacion bruta del canal LRPT")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(alpha=0.3)
plt.axis("equal")
plt.tight_layout()

plt.savefig(SALIDA, dpi=160)
plt.show()

print()
print("Grafico guardado en:")
print(SALIDA)