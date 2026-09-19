from pathlib import Path
import csv
import numpy as np
import reedsolo as rs


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOFT_FILE = (
    ROOT
    / "outputs"
    / "decoded"
    / "lrpt_soft_symbols_adaptive.npy"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "full_decode"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CADU_FILE = (
    OUT_DIR
    / "meteor_lrpt_valid_10s.cadu"
)

CSV_FILE = (
    OUT_DIR
    / "meteor_lrpt_valid_10s.csv"
)


# ============================================================
# LRPT
# ============================================================

SYNC_WORD = 0xFCA2B63DB00D9794

MASK64 = 0xFFFFFFFFFFFFFFFF

# Una CADU = 1024 bytes
# Código convolucional rate 1/2:
#
# 1024 * 8 * 2 = 16384 soft bits
FRAME_SOFT_BITS = 16_384

FRAME_BYTES = 1024

# Radio de búsqueda alrededor de la posición nominal
LOCAL_SYNC_RADIUS = 96

# Número máximo de ambigüedades QPSK que intentaremos
# por cada trama antes de descartarla.
MAX_VARIANTS_TO_TRY = 4


# ============================================================
# VITERBI CCSDS
# ============================================================

K = 7

NUM_STATES = 2 ** (K - 1)

POLYS = [
    79,
    109
]


# ============================================================
# REED-SOLOMON CCSDS
# ============================================================

PRIMITIVE_POLY = 0x187

FCR = 112

# alpha^11 en GF(256) con 0x187
GENERATOR = 0xAD

NSYM = 32

INTERLEAVE = 4


# ============================================================
# ASM CCSDS
# ============================================================

ASM = np.array(
    [
        0x1D,
        0xCF,
        0xFC,
        0x1D
    ],
    dtype=np.uint8
)


# ============================================================
# RANDOMIZADOR CCSDS
# ============================================================

CCSDS_PN = np.array([
    0xFF, 0x48, 0x0E, 0xC0, 0x9A, 0x0D, 0x70, 0xBC,
    0x8E, 0x2C, 0x93, 0xAD, 0xA7, 0xB7, 0x46, 0xCE,
    0x5A, 0x97, 0x7D, 0xCC, 0x32, 0xA2, 0xBF, 0x3E,
    0x0A, 0x10, 0xF1, 0x88, 0x94, 0xCD, 0xEA, 0xB1,
    0xFE, 0x90, 0x1D, 0x81, 0x34, 0x1A, 0xE1, 0x79,
    0x1C, 0x59, 0x27, 0x5B, 0x4F, 0x6E, 0x8D, 0x9C,
    0xB5, 0x2E, 0xFB, 0x98, 0x65, 0x45, 0x7E, 0x7C,
    0x14, 0x21, 0xE3, 0x11, 0x29, 0x9B, 0xD5, 0x63,
    0xFD, 0x20, 0x3B, 0x02, 0x68, 0x35, 0xC2, 0xF2,
    0x38, 0xB2, 0x4E, 0xB6, 0x9E, 0xDD, 0x1B, 0x39,
    0x6A, 0x5D, 0xF7, 0x30, 0xCA, 0x8A, 0xFC, 0xF8,
    0x28, 0x43, 0xC6, 0x22, 0x53, 0x37, 0xAA, 0xC7,
    0xFA, 0x40, 0x76, 0x04, 0xD0, 0x6B, 0x85, 0xE4,
    0x71, 0x64, 0x9D, 0x6D, 0x3D, 0xBA, 0x36, 0x72,
    0xD4, 0xBB, 0xEE, 0x61, 0x95, 0x15, 0xF9, 0xF0,
    0x50, 0x87, 0x8C, 0x44, 0xA6, 0x6F, 0x55, 0x8F,
    0xF4, 0x80, 0xEC, 0x09, 0xA0, 0xD7, 0x0B, 0xC8,
    0xE2, 0xC9, 0x3A, 0xDA, 0x7B, 0x74, 0x6C, 0xE5,
    0xA9, 0x77, 0xDC, 0xC3, 0x2A, 0x2B, 0xF3, 0xE0,
    0xA1, 0x0F, 0x18, 0x89, 0x4C, 0xDE, 0xAB, 0x1F,
    0xE9, 0x01, 0xD8, 0x13, 0x41, 0xAE, 0x17, 0x91,
    0xC5, 0x92, 0x75, 0xB4, 0xF6, 0xE8, 0xD9, 0xCB,
    0x52, 0xEF, 0xB9, 0x86, 0x54, 0x57, 0xE7, 0xC1,
    0x42, 0x1E, 0x31, 0x12, 0x99, 0xBD, 0x56, 0x3F,
    0xD2, 0x03, 0xB0, 0x26, 0x83, 0x5C, 0x2F, 0x23,
    0x8B, 0x24, 0xEB, 0x69, 0xED, 0xD1, 0xB3, 0x96,
    0xA5, 0xDF, 0x73, 0x0C, 0xA8, 0xAF, 0xCF, 0x82,
    0x84, 0x3C, 0x62, 0x25, 0x33, 0x7A, 0xAC, 0x7F,
    0xA4, 0x07, 0x60, 0x4D, 0x06, 0xB8, 0x5E, 0x47,
    0x16, 0x49, 0xD6, 0xD3, 0xDB, 0xA3, 0x67, 0x2D,
    0x4B, 0xBE, 0xE6, 0x19, 0x51, 0x5F, 0x9F, 0x05,
    0x08, 0x78, 0xC4, 0x4A, 0x66, 0xF5, 0x58
], dtype=np.uint8)

assert len(CCSDS_PN) == 255


# ============================================================
# UTILIDADES
# ============================================================

def parity(x):
    return x.bit_count() & 1


# ============================================================
# CORRELADOR QPSK
# ============================================================

def swap_iq_word(word):

    i = word & 0xAAAAAAAAAAAAAAAA
    q = word & 0x5555555555555555

    return (
        (i >> 1)
        |
        ((q << 1) & MASK64)
    )


def rotate64(word, phase):

    i = word & 0xAAAAAAAAAAAAAAAA
    q = word & 0x5555555555555555

    if phase == 0:
        pass

    elif phase == 1:

        word = (
            ((i ^ 0xAAAAAAAAAAAAAAAA) >> 1)
            |
            ((q << 1) & MASK64)
        )

    elif phase == 2:

        word ^= MASK64

    elif phase == 3:

        word = (
            (i >> 1)
            |
            (
                (
                    q ^ 0x5555555555555555
                ) << 1
            )
        )

    else:

        raise ValueError(
            "Fase inválida"
        )

    return (
        (
            (
                word
                & 0x5555555555555555
            )
            << 1
        )
        |
        (
            (
                word
                & 0xAAAAAAAAAAAAAAAA
            )
            >> 1
        )
    ) & MASK64


def word_to_bits(word):

    return np.array(
        [
            (
                word
                >> (63 - i)
            )
            & 1
            for i in range(64)
        ],
        dtype=np.uint8
    )


def build_sync_variants():

    variants = []

    # --------------------------------------------------------
    # SatDump:
    #
    # variantes 0..3:
    # phase = p
    # swap  = True
    # --------------------------------------------------------

    for p in range(4):

        variants.append(
            (
                p,
                f"ROT_{p * 90}",
                rotate64(
                    SYNC_WORD,
                    p
                )
            )
        )

    # --------------------------------------------------------
    # variantes 4..7:
    # phase = p
    # swap  = False
    # --------------------------------------------------------

    swapped = (
        swap_iq_word(
            SYNC_WORD
        )
        ^ MASK64
    )

    for p in range(4):

        variants.append(
            (
                p + 4,
                f"SWAP_ROT_{p * 90}",
                rotate64(
                    swapped,
                    p
                )
            )
        )

    return variants


VARIANTS = build_sync_variants()

PATTERNS = [
    word_to_bits(
        word
    )
    for _, _, word in VARIANTS
]


def phase_swap_from_variant(p):

    phase = p % 4

    swap = (
        p < 4
    )

    return (
        phase,
        swap
    )


# ============================================================
# ROTACIÓN SOFT
#
# Replica rotate_soft() de SatDump.
# ============================================================

def rotate_soft(
    data,
    phase,
    iq_swap
):

    x = (
        data
        .astype(np.int16)
        .copy()
    )

    if len(x) % 2:

        x = x[:-1]

    pairs = x.reshape(
        -1,
        2
    )

    # --------------------------------------------------------
    # IQ SWAP
    # --------------------------------------------------------

    if iq_swap:

        old_i = pairs[:, 0].copy()
        old_q = pairs[:, 1].copy()

        pairs[:, 0] = old_q
        pairs[:, 1] = old_i

    # --------------------------------------------------------
    # ROTACIÓN
    # --------------------------------------------------------

    if phase == 0:

        pass

    elif phase == 1:

        old_i = pairs[:, 0].copy()
        old_q = pairs[:, 1].copy()

        pairs[:, 0] = old_q
        pairs[:, 1] = -old_i

    elif phase == 2:

        pairs *= -1

    elif phase == 3:

        old_i = pairs[:, 0].copy()
        old_q = pairs[:, 1].copy()

        pairs[:, 0] = -old_q
        pairs[:, 1] = old_i

    else:

        raise ValueError(
            "Fase inválida"
        )

    return np.clip(
        pairs.reshape(-1),
        -127,
        127
    ).astype(np.int8)


# ============================================================
# TRELLIS VITERBI
# ============================================================

PRED0 = np.array(
    [
        state >> 1
        for state in range(
            NUM_STATES
        )
    ],
    dtype=np.int16
)

PRED1 = (
    PRED0
    | 32
)


SIGN0 = np.empty(
    (
        NUM_STATES,
        2
    ),
    dtype=np.int8
)

SIGN1 = np.empty(
    (
        NUM_STATES,
        2
    ),
    dtype=np.int8
)


for next_state in range(
    NUM_STATES
):

    bit = (
        next_state
        & 1
    )

    for predecessor, table in (
        (
            int(
                PRED0[next_state]
            ),
            SIGN0
        ),
        (
            int(
                PRED1[next_state]
            ),
            SIGN1
        )
    ):

        register = (
            (predecessor << 1)
            | bit
        )

        b0 = parity(
            register
            & POLYS[0]
        )

        b1 = parity(
            register
            & POLYS[1]
        )

        table[
            next_state,
            0
        ] = (
            1
            if b0
            else -1
        )

        table[
            next_state,
            1
        ] = (
            1
            if b1
            else -1
        )


# ============================================================
# VITERBI SOFT
# ============================================================

def viterbi_decode_soft(
    soft
):

    received = (
        soft
        .astype(np.float64)
        .reshape(-1, 2)
    )

    steps = len(
        received
    )

    # Estado inicial no sesgado
    metrics = np.zeros(
        NUM_STATES,
        dtype=np.float64
    )

    decisions = np.zeros(
        (
            steps,
            NUM_STATES
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    for t in range(
        steps
    ):

        r0 = received[
            t,
            0
        ]

        r1 = received[
            t,
            1
        ]

        candidate0 = (
            metrics[PRED0]
            +
            r0 * SIGN0[:, 0]
            +
            r1 * SIGN0[:, 1]
        )

        candidate1 = (
            metrics[PRED1]
            +
            r0 * SIGN1[:, 0]
            +
            r1 * SIGN1[:, 1]
        )

        use1 = (
            candidate1
            > candidate0
        )

        decisions[t] = (
            use1
        )

        metrics = np.where(
            use1,
            candidate1,
            candidate0
        )

    # --------------------------------------------------------
    # TRACEBACK
    # --------------------------------------------------------

    state = int(
        np.argmax(
            metrics
        )
    )

    decoded = np.empty(
        steps,
        dtype=np.uint8
    )

    for t in range(
        steps - 1,
        -1,
        -1
    ):

        decoded[t] = (
            state
            & 1
        )

        predecessor_high = int(
            decisions[
                t,
                state
            ]
        )

        state = (
            (state >> 1)
            |
            (
                predecessor_high
                << 5
            )
        )

    return decoded


# ============================================================
# BITS -> BYTES
# ============================================================

def bits_to_bytes(
    bits
):

    usable = (
        len(bits)
        // 8
        * 8
    )

    return np.packbits(
        bits[:usable]
    )


# ============================================================
# DERANDOMIZACIÓN
# ============================================================

def derandomize(
    frame
):

    output = (
        frame
        .copy()
    )

    indices = np.arange(
        4,
        len(output)
    )

    output[4:] ^= (
        CCSDS_PN[
            (
                indices
                - 4
            )
            % 255
        ]
    )

    return output


# ============================================================
# REED-SOLOMON
# ============================================================

rs.init_tables(
    prim=PRIMITIVE_POLY,
    generator=GENERATOR,
    c_exp=8
)


def decode_rs_codeword(
    codeword
):

    raw = bytearray(
        codeword.tobytes()
    )

    syndromes = rs.rs_calc_syndromes(
        raw,
        NSYM,
        fcr=FCR,
        generator=GENERATOR
    )

    # --------------------------------------------------------
    # Ya está limpio
    # --------------------------------------------------------

    if all(
        s == 0
        for s in syndromes[1:]
    ):

        return (
            True,
            codeword.copy(),
            0
        )

    # --------------------------------------------------------
    # Intentar corrección
    # --------------------------------------------------------

    try:

        (
            message,
            ecc,
            errata_pos
        ) = rs.rs_correct_msg(
            raw,
            NSYM,
            fcr=FCR,
            generator=GENERATOR
        )

        corrected = np.frombuffer(
            (
                bytes(message)
                + bytes(ecc)
            ),
            dtype=np.uint8
        ).copy()

        syndromes_after = rs.rs_calc_syndromes(
            bytearray(
                corrected.tobytes()
            ),
            NSYM,
            fcr=FCR,
            generator=GENERATOR
        )

        ok = all(
            s == 0
            for s in syndromes_after[1:]
        )

        return (
            ok,
            corrected,
            len(errata_pos)
        )

    except rs.ReedSolomonError:

        return (
            False,
            codeword.copy(),
            -1
        )


def rs_correct_frame(
    frame
):

    output = (
        frame
        .copy()
    )

    area = (
        output[4:]
        .copy()
    )

    errors = []

    for branch in range(
        INTERLEAVE
    ):

        codeword = area[
            branch::INTERLEAVE
        ]

        (
            ok,
            corrected,
            n_errors
        ) = decode_rs_codeword(
            codeword
        )

        errors.append(
            n_errors
        )

        if not ok:

            return (
                False,
                frame,
                errors
            )

        area[
            branch::INTERLEAVE
        ] = corrected

    output[:4] = ASM
    output[4:] = area

    return (
        True,
        output,
        errors
    )


# ============================================================
# VCDU
# ============================================================

def parse_vcdu(
    cadu
):

    version = int(
        cadu[4]
        >> 6
    )

    spacecraft_id = int(
        (
            (
                cadu[4]
                & 0x3F
            )
            << 2
        )
        |
        (
            cadu[5]
            >> 6
        )
    )

    vcid = int(
        cadu[5]
        & 0x3F
    )

    counter = int(
        (
            int(cadu[6])
            << 16
        )
        |
        (
            int(cadu[7])
            << 8
        )
        |
        int(cadu[8])
    )

    return (
        version,
        spacecraft_id,
        vcid,
        counter
    )


# ============================================================
# CORRELACIÓN EN POSICIÓN EXACTA
# ============================================================

def correlation_at(
    hard,
    position
):

    if (
        position < 0
        or position + 64 > len(hard)
    ):

        return [
            -1
            for _ in PATTERNS
        ]

    block = hard[
        position:
        position + 64
    ]

    return [
        (
            64
            -
            int(
                np.count_nonzero(
                    block
                    != pattern
                )
            )
        )
        for pattern in PATTERNS
    ]


# ============================================================
# RESINCRONIZACIÓN LOCAL
#
# La separación nominal entre tramas es 16384 soft bits.
#
# En una señal real puede existir una pequeña deriva de timing.
# Por eso NO confiamos ciegamente en:
#
# posición_anterior + 16384
#
# Buscamos el sync alrededor de la posición esperada.
# ============================================================

def find_local_sync(
    hard,
    expected_position,
    radius=LOCAL_SYNC_RADIUS
):

    best_position = None
    best_correlations = None
    best_correlation = -1
    best_variant = None

    start = max(
        0,
        expected_position - radius
    )

    end = min(
        len(hard) - 64,
        expected_position + radius
    )

    # --------------------------------------------------------
    # QPSK:
    # debemos conservar la paridad de parejas I/Q
    # --------------------------------------------------------

    if (
        start % 2
        != expected_position % 2
    ):

        start += 1

    for position in range(
        start,
        end + 1,
        2
    ):

        correlations = correlation_at(
            hard,
            position
        )

        variant = int(
            np.argmax(
                correlations
            )
        )

        correlation = int(
            correlations[
                variant
            ]
        )

        if (
            correlation
            > best_correlation
        ):

            best_correlation = (
                correlation
            )

            best_position = (
                position
            )

            best_correlations = (
                correlations
            )

            best_variant = (
                variant
            )

    return (
        best_position,
        best_correlations,
        best_correlation,
        best_variant
    )


# ============================================================
# CARGAR SOFT BITS
# ============================================================

soft = np.load(
    SOFT_FILE
).astype(np.int8)

hard = (
    soft > 0
).astype(np.uint8)


print()

print(
    "================================================"
)

print(
    "    DECODIFICACION AUTOMATICA LRPT - 10 s"
)

print(
    "================================================"
)

print()

print(
    f"Soft bits disponibles : "
    f"{len(soft):,}"
)

print()


# ============================================================
# ENCONTRAR ANCLA GLOBAL
# ============================================================

windows = (
    np.lib.stride_tricks
    .sliding_window_view(
        hard,
        64
    )
    [::2]
)


global_best_corr = -1

global_best_pos = None

global_best_variant = None


print(
    "Máximo por variante:"
)

print()


for p, name, _ in VARIANTS:

    distances = np.count_nonzero(
        windows
        != PATTERNS[p],
        axis=1
    )

    index = int(
        np.argmin(
            distances
        )
    )

    correlation = (
        64
        -
        int(
            distances[index]
        )
    )

    position = (
        index
        * 2
    )

    print(
        f"{name:15s} | "
        f"{correlation:2d}/64 | "
        f"pos={position:9,d}"
    )

    if (
        correlation
        > global_best_corr
    ):

        global_best_corr = (
            correlation
        )

        global_best_pos = (
            position
        )

        global_best_variant = (
            p
        )

    del distances


anchor_name = (
    VARIANTS[
        global_best_variant
    ][1]
)


print()

print(
    "------------------------------------------------"
)

print(
    f"ANCLA       : {anchor_name}"
)

print(
    f"Correlación : "
    f"{global_best_corr}/64"
)

print(
    f"Posición    : "
    f"{global_best_pos:,}"
)

print(
    "------------------------------------------------"
)


del windows


# ============================================================
# RETÍCULA NOMINAL
#
# Esta retícula sirve SOLO como posición esperada.
#
# Después find_local_sync() mueve cada frame a su sync real.
# ============================================================

first_position = (
    global_best_pos
)


while (
    first_position
    - FRAME_SOFT_BITS
    >= 0
):

    first_position -= (
        FRAME_SOFT_BITS
    )


frame_positions = []

position = (
    first_position
)


while (
    position
    + FRAME_SOFT_BITS
    <= len(soft)
):

    frame_positions.append(
        position
    )

    position += (
        FRAME_SOFT_BITS
    )


print()

print(
    f"Tramas candidatas : "
    f"{len(frame_positions)}"
)

print()

print(
    "================================================"
)

print(
    "              DECODIFICANDO CADUs"
)

print(
    "================================================"
)

print()


# ============================================================
# DECODIFICACIÓN
# ============================================================

valid_frames = []

report_rows = []

last_valid_variant = None

last_counter = None


for (
    frame_index,
    expected_position
) in enumerate(
    frame_positions
):

    # ========================================================
    # NUEVO:
    # RESINCRONIZACIÓN LOCAL
    # ========================================================

    (
        position,
        correlations,
        local_best_corr,
        local_best_variant
    ) = find_local_sync(
        hard,
        expected_position,
        radius=LOCAL_SYNC_RADIUS
    )


    sync_delta = (
        position
        - expected_position
    )


    # --------------------------------------------------------
    # Ordenar variantes QPSK por correlación
    # --------------------------------------------------------

    ordered_variants = list(
        np.argsort(
            correlations
        )[::-1]
    )


    # --------------------------------------------------------
    # Probar las mejores variantes
    # --------------------------------------------------------

    trial_variants = []


    # Primero la mejor correlación local
    for variant in ordered_variants[
        :MAX_VARIANTS_TO_TRY
    ]:

        variant = int(
            variant
        )

        if (
            variant
            not in trial_variants
        ):

            trial_variants.append(
                variant
            )


    # También probar la variante anterior válida.
    #
    # Puede ser útil cuando el sync está degradado.
    if (
        last_valid_variant
        is not None
        and last_valid_variant
        not in trial_variants
    ):

        trial_variants.append(
            last_valid_variant
        )


    accepted = None


    # ========================================================
    # PROBAR VARIANTES
    # ========================================================

    for variant in trial_variants:

        phase, swap = (
            phase_swap_from_variant(
                variant
            )
        )


        raw_frame = soft[
            position:
            position
            + FRAME_SOFT_BITS
        ]


        if (
            len(raw_frame)
            != FRAME_SOFT_BITS
        ):

            continue


        # ----------------------------------------------------
        # Rotación / IQ
        # ----------------------------------------------------

        corrected_soft = (
            rotate_soft(
                raw_frame,
                phase,
                swap
            )
        )


        # ----------------------------------------------------
        # Viterbi
        # ----------------------------------------------------

        decoded_bits = (
            viterbi_decode_soft(
                corrected_soft
            )
        )


        # ----------------------------------------------------
        # Bytes
        # ----------------------------------------------------

        frame = (
            bits_to_bytes(
                decoded_bits
            )
        )


        if (
            len(frame)
            != FRAME_BYTES
        ):

            continue


        # ----------------------------------------------------
        # Derandomización CCSDS
        # ----------------------------------------------------

        frame = (
            derandomize(
                frame
            )
        )


        # ----------------------------------------------------
        # Edge case:
        # CADU completamente invertida
        # ----------------------------------------------------

        if (
            frame[9]
            == 0xFF
        ):

            frame ^= (
                0xFF
            )


        # ----------------------------------------------------
        # Reed-Solomon
        # ----------------------------------------------------

        (
            rs_ok,
            corrected_frame,
            rs_errors
        ) = rs_correct_frame(
            frame
        )


        if not rs_ok:

            continue


        # ----------------------------------------------------
        # VCDU
        # ----------------------------------------------------

        (
            version,
            spacecraft_id,
            vcid,
            counter
        ) = parse_vcdu(
            corrected_frame
        )


        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        if (
            version > 1
        ):

            continue


        accepted = (
            variant,
            correlations[
                variant
            ],
            corrected_frame,
            rs_errors,
            version,
            spacecraft_id,
            vcid,
            counter
        )

        break


    # ========================================================
    # FALLÓ
    # ========================================================

    if (
        accepted
        is None
    ):

        best_variant = int(
            ordered_variants[0]
        )

        best_name = (
            VARIANTS[
                best_variant
            ][1]
        )

        print(
            f"{frame_index:03d} | "
            f"esperado={expected_position:9,d} | "
            f"pos={position:9,d} | "
            f"delta={sync_delta:+4d} | "
            f"{best_name:12s} "
            f"{correlations[best_variant]:2d}/64 | "
            f"RS FAIL"
        )


        report_rows.append(
            {
                "index": frame_index,
                "expected_pos": expected_position,
                "soft_pos": position,
                "sync_delta": sync_delta,
                "valid": 0,
                "variant": best_name,
                "sync_corr": correlations[
                    best_variant
                ],
                "vcid": "",
                "counter": "",
                "counter_hex": "",
                "counter_gap": "",
                "rs_errors": ""
            }
        )

        continue


    # ========================================================
    # CADU VÁLIDA
    # ========================================================

    (
        variant,
        sync_corr,
        corrected_frame,
        rs_errors,
        version,
        spacecraft_id,
        vcid,
        counter
    ) = accepted


    variant_name = (
        VARIANTS[
            variant
        ][1]
    )


    # --------------------------------------------------------
    # Diferencia de contador respecto a CADU válida anterior
    # --------------------------------------------------------

    if (
        last_counter
        is None
    ):

        counter_gap = None

    else:

        counter_gap = (
            (
                counter
                - last_counter
            )
            & 0xFFFFFF
        )


    last_counter = (
        counter
    )

    last_valid_variant = (
        variant
    )


    valid_frames.append(
        corrected_frame
    )


    if (
        counter_gap
        is None
    ):

        gap_text = ""

    else:

        gap_text = (
            f" gap={counter_gap}"
        )


    print(
        f"{frame_index:03d} | "
        f"esperado={expected_position:9,d} | "
        f"pos={position:9,d} | "
        f"delta={sync_delta:+4d} | "
        f"{variant_name:12s} "
        f"{sync_corr:2d}/64 | "
        f"RS={rs_errors} | "
        f"VCID={vcid:2d} | "
        f"CNT=0x{counter:06X}"
        f"{gap_text} | "
        f"OK"
    )


    report_rows.append(
        {
            "index": frame_index,
            "expected_pos": expected_position,
            "soft_pos": position,
            "sync_delta": sync_delta,
            "valid": 1,
            "variant": variant_name,
            "sync_corr": sync_corr,
            "vcid": vcid,
            "counter": counter,
            "counter_hex": f"0x{counter:06X}",
            "counter_gap": (
                ""
                if counter_gap is None
                else counter_gap
            ),
            "rs_errors": ",".join(
                map(
                    str,
                    rs_errors
                )
            )
        }
    )


# ============================================================
# GUARDAR CADUs
# ============================================================

if valid_frames:

    all_valid = np.concatenate(
        valid_frames
    ).astype(
        np.uint8
    )

    all_valid.tofile(
        CADU_FILE
    )

else:

    CADU_FILE.write_bytes(
        b""
    )


# ============================================================
# CSV DIAGNÓSTICO
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "index",
            "expected_pos",
            "soft_pos",
            "sync_delta",
            "valid",
            "variant",
            "sync_corr",
            "vcid",
            "counter",
            "counter_hex",
            "counter_gap",
            "rs_errors"
        ]
    )

    writer.writeheader()

    writer.writerows(
        report_rows
    )


# ============================================================
# RESUMEN
# ============================================================

valid_rows = [
    row
    for row in report_rows
    if row["valid"] == 1
]


vcid5_count = sum(
    1
    for row in valid_rows
    if row["vcid"] == 5
)


# ============================================================
# RACHA DE CONTADORES CONSECUTIVOS
# ============================================================

longest_run = 0

current_run = 0

previous_counter = None


for row in valid_rows:

    counter = int(
        row["counter"]
    )

    if (
        previous_counter
        is not None
        and (
            (
                counter
                - previous_counter
            )
            & 0xFFFFFF
        )
        == 1
    ):

        current_run += 1

    else:

        current_run = 1


    longest_run = max(
        longest_run,
        current_run
    )


    previous_counter = (
        counter
    )


# ============================================================
# ESTADÍSTICAS DELTA
# ============================================================

if report_rows:

    deltas = np.array(
        [
            row["sync_delta"]
            for row in report_rows
        ],
        dtype=np.int64
    )

    delta_min = int(
        np.min(
            deltas
        )
    )

    delta_max = int(
        np.max(
            deltas
        )
    )

    delta_median = float(
        np.median(
            deltas
        )
    )

else:

    delta_min = 0
    delta_max = 0
    delta_median = 0.0


print()

print(
    "================================================"
)

print(
    "                    RESUMEN"
)

print(
    "================================================"
)

print()

print(
    f"Tramas candidatas     : "
    f"{len(frame_positions)}"
)

print(
    f"CADUs válidas         : "
    f"{len(valid_frames)}"
)

print(
    f"CADUs VCID 5          : "
    f"{vcid5_count}"
)

print(
    f"Racha consecutiva max : "
    f"{longest_run}"
)

print()

print(
    "Desplazamiento de sincronismo:"
)

print(
    f"  mínimo  : {delta_min:+d}"
)

print(
    f"  mediana : {delta_median:+.1f}"
)

print(
    f"  máximo  : {delta_max:+d}"
)

print()

print(
    "Archivo CADU:"
)

print(
    CADU_FILE
)

print()

print(
    "Índice CSV:"
)

print(
    CSV_FILE
)

print()

print(
    "================================================"
)

print(
    "FIN"
)

print(
    "================================================"
)