from pathlib import Path
import wave
import numpy as np


# --------------------------------------------------
# RUTAS DEL PROYECTO
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

ARCHIVO = (
    ROOT
    / "data"
    / "raw"
    / "meteor_137100"
    / "meteor_137100_iq.wav"
)


# --------------------------------------------------
# LEER METADATOS DEL WAV
# --------------------------------------------------

with wave.open(str(ARCHIVO), "rb") as w:

    canales = w.getnchannels()
    sample_rate = w.getframerate()
    bits = w.getsampwidth() * 8
    frames = w.getnframes()
    duracion = frames / sample_rate

    print()
    print("======================================")
    print("       INSPECCION DE ARCHIVO I/Q")
    print("======================================")
    print()

    print("Archivo:")
    print(ARCHIVO)

    print()
    print(f"Canales            : {canales}")
    print(f"Sample rate        : {sample_rate:,} Hz")
    print(f"Bits por muestra   : {bits}")
    print(f"Frames             : {frames:,}")
    print(f"Duracion           : {duracion:.2f} segundos")
    print(f"Duracion           : {duracion / 60:.2f} minutos")

    # Leer solo 20 muestras I/Q
    raw = w.readframes(20)


# --------------------------------------------------
# CONVERTIR PCM 16 bits A I/Q
# --------------------------------------------------

data = np.frombuffer(
    raw,
    dtype=np.int16
)

iq = data.reshape(-1, 2)

I = iq[:, 0]
Q = iq[:, 1]

signal_complex = (
    I.astype(np.float32)
    + 1j * Q.astype(np.float32)
)


# --------------------------------------------------
# MOSTRAR PRIMERAS MUESTRAS
# --------------------------------------------------

print()
print("Primeras muestras:")

for i in range(len(signal_complex)):

    print(
        f"{i:02d} | "
        f"I={I[i]:6d} | "
        f"Q={Q[i]:6d} | "
        f"IQ={signal_complex[i]}"
    )