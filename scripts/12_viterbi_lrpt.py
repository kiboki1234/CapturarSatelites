from pathlib import Path
import numpy as np


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

ARCHIVO = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_soft_symbols.npy"
)

SALIDA_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "viterbi"
)

SALIDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMETROS CCSDS
# ============================================================

K = 7
NUM_STATES = 2 ** (K - 1)

POLYS = [
    79,
    109
]

FRAME_SOFT_BITS = 16_384
DECODED_BITS = FRAME_SOFT_BITS // 2
DECODED_BYTES = DECODED_BITS // 8


# Tres tramas consecutivas que ya comprobamos
FRAME_POSITIONS = [
    71_010,
    87_394,
    103_778
]


# Resultado del correlador:
#
# Variante ROT_90
#
# En SatDump, para esta variante:
#   swap IQ = True
#   phase = 90 grados
#
PHASE = 1
IQ_SWAP = True


# ============================================================
# PARIDAD
# ============================================================

def parity(x):
    return x.bit_count() & 1


# ============================================================
# ROTACION DE SOFT SYMBOLS
#
# Replica la idea de rotate_soft() de SatDump.
# ============================================================

def rotate_soft(data, phase, iq_swap):

    x = data.astype(
        np.int16
    ).copy()

    if len(x) % 2 != 0:
        x = x[:-1]

    pairs = x.reshape(
        -1,
        2
    )

    # --------------------------------------------------------
    # SWAP I / Q
    # --------------------------------------------------------

    if iq_swap:

        temp = pairs[:, 0].copy()

        pairs[:, 0] = pairs[:, 1]
        pairs[:, 1] = temp


    # --------------------------------------------------------
    # ROTACION
    # --------------------------------------------------------

    if phase == 0:
        pass

    elif phase == 1:
        # 90 grados:
        # I' = Q
        # Q' = -I

        old_i = pairs[:, 0].copy()
        old_q = pairs[:, 1].copy()

        pairs[:, 0] = old_q
        pairs[:, 1] = -old_i

    elif phase == 2:
        # 180 grados

        pairs *= -1

    elif phase == 3:
        # 270 grados:
        # I' = -Q
        # Q' = I

        old_i = pairs[:, 0].copy()
        old_q = pairs[:, 1].copy()

        pairs[:, 0] = -old_q
        pairs[:, 1] = old_i

    else:
        raise ValueError(
            "Fase invalida"
        )

    return np.clip(
        pairs.reshape(-1),
        -127,
        127
    ).astype(np.int8)


# ============================================================
# TABLA DEL TRELLIS
# ============================================================

NEXT_STATE = np.zeros(
    (NUM_STATES, 2),
    dtype=np.int16
)

OUTPUT_BITS = np.zeros(
    (NUM_STATES, 2, 2),
    dtype=np.uint8
)


for state in range(NUM_STATES):

    for input_bit in (0, 1):

        # SatDump:
        # my_state = (my_state << 1) | input_bit

        register = (
            (state << 1)
            | input_bit
        )

        next_state = (
            register
            & (NUM_STATES - 1)
        )

        NEXT_STATE[
            state,
            input_bit
        ] = next_state

        OUTPUT_BITS[
            state,
            input_bit,
            0
        ] = parity(
            register & POLYS[0]
        )

        OUTPUT_BITS[
            state,
            input_bit,
            1
        ] = parity(
            register & POLYS[1]
        )


# ============================================================
# VITERBI SOFT
# ============================================================

def viterbi_decode_soft(soft):

    if len(soft) != FRAME_SOFT_BITS:
        raise ValueError(
            f"Se esperaban {FRAME_SOFT_BITS} soft bits"
        )

    received = soft.astype(
        np.float64
    ).reshape(-1, 2)

    steps = len(received)

    # SatDump comienza sin sesgo fuerte de estado.
    # Permitimos cualquier estado inicial.
    metrics = np.zeros(
        NUM_STATES,
        dtype=np.float64
    )

    previous_state = np.zeros(
        (steps, NUM_STATES),
        dtype=np.int16
    )


    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    for t in range(steps):

        r0 = received[t, 0]
        r1 = received[t, 1]

        new_metrics = np.full(
            NUM_STATES,
            -np.inf,
            dtype=np.float64
        )

        for state in range(NUM_STATES):

            base_metric = metrics[state]

            for input_bit in (0, 1):

                ns = NEXT_STATE[
                    state,
                    input_bit
                ]

                b0 = OUTPUT_BITS[
                    state,
                    input_bit,
                    0
                ]

                b1 = OUTPUT_BITS[
                    state,
                    input_bit,
                    1
                ]

                # Soft > 0 representa bit 1
                # Soft < 0 representa bit 0

                s0 = (
                    1.0
                    if b0
                    else -1.0
                )

                s1 = (
                    1.0
                    if b1
                    else -1.0
                )

                branch_metric = (
                    r0 * s0
                    +
                    r1 * s1
                )

                candidate = (
                    base_metric
                    +
                    branch_metric
                )

                if candidate > new_metrics[ns]:

                    new_metrics[ns] = candidate

                    previous_state[
                        t,
                        ns
                    ] = state

        metrics = new_metrics


    # --------------------------------------------------------
    # TRACEBACK
    # --------------------------------------------------------

    state = int(
        np.argmax(metrics)
    )

    decoded = np.zeros(
        steps,
        dtype=np.uint8
    )

    for t in range(
        steps - 1,
        -1,
        -1
    ):

        # El bit nuevo queda como LSB del estado
        decoded[t] = (
            state & 1
        )

        state = int(
            previous_state[
                t,
                state
            ]
        )


    initial_state = state

    return (
        decoded,
        initial_state
    )


# ============================================================
# RE-ENCODER
#
# Lo usamos para estimar cuánto coincide el resultado Viterbi
# con la señal recibida.
# ============================================================

def convolutional_encode(bits, initial_state):

    state = int(
        initial_state
    )

    encoded = np.zeros(
        len(bits) * 2,
        dtype=np.uint8
    )

    p = 0

    for bit in bits:

        register = (
            (state << 1)
            | int(bit)
        )

        encoded[p] = parity(
            register & POLYS[0]
        )

        encoded[p + 1] = parity(
            register & POLYS[1]
        )

        p += 2

        state = (
            register
            & (NUM_STATES - 1)
        )

    return encoded


# ============================================================
# BITS -> BYTES
# ============================================================

def bits_to_bytes(bits):

    usable = (
        len(bits)
        // 8
        * 8
    )

    bits = bits[
        :usable
    ]

    return np.packbits(
        bits
    )


# ============================================================
# CARGAR SOFT DATA
# ============================================================

soft_all = np.load(
    ARCHIVO
).astype(np.int8)


print()
print("================================================")
print("           VITERBI METEOR LRPT")
print("================================================")
print()

print(
    f"Soft bits disponibles : {len(soft_all):,}"
)

print(
    f"Frame soft bits       : {FRAME_SOFT_BITS:,}"
)

print(
    f"Bits decodificados    : {DECODED_BITS:,}"
)

print(
    f"Bytes por trama       : {DECODED_BYTES:,}"
)

print()

print(
    f"Polinomios            : {POLYS}"
)

print(
    "Ambigüedad corregida : ROT_90 + IQ_SWAP"
)

print()


# ============================================================
# DECODIFICAR TRAMAS
# ============================================================

frames_decoded = []


for index, position in enumerate(
    FRAME_POSITIONS
):

    print()
    print(
        "------------------------------------------------"
    )

    print(
        f"TRAMA {index}"
    )

    print(
        f"Posicion soft bits : {position:,}"
    )


    raw_frame = soft_all[
        position:
        position + FRAME_SOFT_BITS
    ]


    if len(raw_frame) != FRAME_SOFT_BITS:

        print(
            "Trama incompleta, se omite."
        )

        continue


    # --------------------------------------------------------
    # CORREGIR ROTACION / IQ
    # --------------------------------------------------------

    corrected = rotate_soft(
        raw_frame,
        PHASE,
        IQ_SWAP
    )


    # --------------------------------------------------------
    # VITERBI
    # --------------------------------------------------------

    bits, initial_state = (
        viterbi_decode_soft(
            corrected
        )
    )


    frame_bytes = bits_to_bytes(
        bits
    )


    # --------------------------------------------------------
    # RE-ENCODIFICAR PARA MEDIR BER APROXIMADO
    # --------------------------------------------------------

    encoded = convolutional_encode(
        bits,
        initial_state
    )

    hard_received = (
        corrected > 0
    ).astype(np.uint8)

    bit_errors = np.count_nonzero(
        encoded != hard_received
    )

    ber = (
        bit_errors
        / len(encoded)
    )


    print(
        f"Estado inicial      : {initial_state}"
    )

    print(
        f"Errores re-encode   : "
        f"{bit_errors:,} / {len(encoded):,}"
    )

    print(
        f"BER aproximado      : {ber:.5f}"
    )


    # --------------------------------------------------------
    # MOSTRAR PRIMEROS BYTES
    # --------------------------------------------------------

    first_bytes = frame_bytes[:32]

    hex_string = " ".join(
        f"{b:02X}"
        for b in first_bytes
    )

    print()

    print(
        "Primeros 32 bytes:"
    )

    print(
        hex_string
    )


    # --------------------------------------------------------
    # GUARDAR
    # --------------------------------------------------------

    salida = (
        SALIDA_DIR
        / f"frame_{index:02d}.bin"
    )

    frame_bytes.tofile(
        salida
    )

    frames_decoded.append(
        frame_bytes
    )

    print()

    print(
        "Guardado:"
    )

    print(
        salida
    )


# ============================================================
# GUARDAR TODAS JUNTAS
# ============================================================

if frames_decoded:

    all_frames = np.concatenate(
        frames_decoded
    )

    salida_total = (
        SALIDA_DIR
        / "frames_viterbi_raw.bin"
    )

    all_frames.tofile(
        salida_total
    )

    print()
    print(
        "================================================"
    )

    print(
        "Archivo conjunto:"
    )

    print(
        salida_total
    )