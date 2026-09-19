from pathlib import Path
import csv
import gc
import math
import time

import numpy as np
from scipy.io import wavfile
from scipy import signal
import reedsolo as rs

try:
    from numba import njit
except ImportError:
    print()
    print("ERROR: este script necesita Numba.")
    print()
    print("Instala con:")
    print()
    print("python -m pip install numba")
    print()
    raise SystemExit(1)


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_WAV = (
    ROOT
    / "data"
    / "raw"
    / "meteor_137100"
    / "meteor_137100_iq.wav"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "fullpass"
)

CHUNKS_DIR = (
    OUTPUT_DIR
    / "chunks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHUNKS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FINAL_CADU = (
    OUTPUT_DIR
    / "meteor_lrpt_full.cadu"
)

FINAL_CSV = (
    OUTPUT_DIR
    / "fullpass_frames.csv"
)

CHUNKS_CSV = (
    OUTPUT_DIR
    / "fullpass_chunks.csv"
)


# ============================================================
# CONFIGURACIÓN DE LA GRABACIÓN
# ============================================================

EXPECTED_INPUT_FS = 500_000

OUTPUT_FS = 288_000.0

SYMBOL_RATE = 72_000.0

SPS = (
    OUTPUT_FS
    / SYMBOL_RATE
)


# ============================================================
# PROCESAMIENTO POR CHUNKS
#
# 12 s procesados
# cada 10 s
#
# => 2 s de solape
# ============================================================

CHUNK_SECONDS = 12.0

HOP_SECONDS = 10.0

# Eliminamos 0.5 s de cada extremo una vez recuperados
# los símbolos. El solape evita perder datos entre chunks.
EDGE_DISCARD_SECONDS = 0.5

MIN_CHUNK_SECONDS = 3.0


# ============================================================
# FILTRO LRPT
# ============================================================

LOWPASS_HZ = 70_000.0

LOWPASS_TAPS = 161


# ============================================================
# RRC
# ============================================================

RRC_ALPHA = 0.5

RRC_TAPS = 31


# ============================================================
# COSTAS LOOP
# ============================================================

COSTAS_BW = 0.002

# rad/sample
COSTAS_FREQ_LIMIT = 1.0


# ============================================================
# MUELLER & MULLER
# ============================================================

CLOCK_ALPHA = 8.7e-3

CLOCK_GAIN_MU = CLOCK_ALPHA

CLOCK_GAIN_OMEGA = (
    CLOCK_ALPHA ** 2
    / 4.0
)

CLOCK_MU_INITIAL = 0.5

CLOCK_OMEGA_REL_LIMIT = 0.005


# ============================================================
# FRAME LRPT
# ============================================================

SYNC_WORD = 0xFCA2B63DB00D9794

MASK64 = 0xFFFFFFFFFFFFFFFF

FRAME_SOFT_BITS = 16_384

FRAME_BYTES = 1024

LOCAL_SYNC_RADIUS = 128

MIN_SYNC_CORRELATION = 42

MAX_VARIANTS_TO_TRY = 4


# ============================================================
# CONVOLUTIONAL CODE
# ============================================================

K = 7

NUM_STATES = 64

POLYS = [
    79,
    109
]


# ============================================================
# REED SOLOMON
# ============================================================

PRIMITIVE_POLY = 0x187

FCR = 112

GENERATOR = 0xAD

NSYM = 32

INTERLEAVE = 4


# ============================================================
# ASM
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
# CCSDS PN
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


PN_FRAME = np.resize(
    CCSDS_PN,
    FRAME_BYTES - 4
)


# ============================================================
# POPCOUNT PARA CORRELADOR
# ============================================================

POPCOUNT = np.array(
    [
        bin(i).count("1")
        for i in range(256)
    ],
    dtype=np.uint8
)


# ============================================================
# RRC
# ============================================================

def make_rrc_taps(
    alpha,
    sps,
    num_taps
):

    half = (
        num_taps
        // 2
    )

    t = (
        np.arange(
            -half,
            half + 1,
            dtype=np.float64
        )
        / sps
    )

    h = np.zeros_like(
        t
    )

    for i, ti in enumerate(
        t
    ):

        if abs(ti) < 1e-12:

            h[i] = (
                1
                +
                alpha
                * (
                    4 / np.pi
                    - 1
                )
            )

        elif (
            abs(
                abs(ti)
                -
                1 / (
                    4 * alpha
                )
            )
            < 1e-10
        ):

            h[i] = (
                alpha
                / np.sqrt(2)
                * (
                    (
                        1
                        +
                        2 / np.pi
                    )
                    * np.sin(
                        np.pi
                        / (
                            4 * alpha
                        )
                    )
                    +
                    (
                        1
                        -
                        2 / np.pi
                    )
                    * np.cos(
                        np.pi
                        / (
                            4 * alpha
                        )
                    )
                )
            )

        else:

            numerator = (
                np.sin(
                    np.pi
                    * ti
                    * (
                        1 - alpha
                    )
                )
                +
                4
                * alpha
                * ti
                * np.cos(
                    np.pi
                    * ti
                    * (
                        1 + alpha
                    )
                )
            )

            denominator = (
                np.pi
                * ti
                * (
                    1
                    -
                    (
                        4
                        * alpha
                        * ti
                    ) ** 2
                )
            )

            h[i] = (
                numerator
                /
                denominator
            )

    h /= np.sqrt(
        np.sum(
            h ** 2
        )
    )

    return h.astype(
        np.float32
    )


RRC_FILTER = make_rrc_taps(
    RRC_ALPHA,
    SPS,
    RRC_TAPS
)


# ============================================================
# LOWPASS
# ============================================================

LOWPASS_FILTER = signal.firwin(
    LOWPASS_TAPS,
    LOWPASS_HZ,
    fs=EXPECTED_INPUT_FS,
    window="hann"
).astype(
    np.float32
)


# ============================================================
# COSTAS - NUMBA
# ============================================================

@njit(cache=True)
def costas_qpsk_numba(
    x,
    loop_bw,
    freq_limit
):

    damping = (
        math.sqrt(2.0)
        /
        2.0
    )

    denom = (
        1.0
        +
        2.0
        * damping
        * loop_bw
        +
        loop_bw
        * loop_bw
    )

    alpha = (
        4.0
        * damping
        * loop_bw
        /
        denom
    )

    beta = (
        4.0
        * loop_bw
        * loop_bw
        /
        denom
    )

    out = np.empty(
        len(x),
        dtype=np.complex64
    )

    phase = 0.0

    freq = 0.0


    for i in range(
        len(x)
    ):

        c = math.cos(
            -phase
        )

        s = math.sin(
            -phase
        )

        sample = (
            x[i]
            * complex(
                c,
                s
            )
        )

        out[i] = sample


        real = sample.real

        imag = sample.imag


        sign_i = (
            1.0
            if real > 0.0
            else -1.0
        )

        sign_q = (
            1.0
            if imag > 0.0
            else -1.0
        )


        error = (
            sign_i
            * imag
            -
            sign_q
            * real
        )


        if error > 1.0:

            error = 1.0

        elif error < -1.0:

            error = -1.0


        freq += (
            beta
            * error
        )


        if freq > freq_limit:

            freq = freq_limit

        elif freq < -freq_limit:

            freq = -freq_limit


        phase += (
            freq
            +
            alpha
            * error
        )


        if phase > (
            2.0
            * math.pi
        ):

            phase -= (
                2.0
                * math.pi
            )

        elif phase < (
            -2.0
            * math.pi
        ):

            phase += (
                2.0
                * math.pi
            )


    return (
        out,
        freq
    )


# ============================================================
# INTERPOLACIÓN CÚBICA - NUMBA
# ============================================================

@njit(cache=True)
def cubic_interpolate_numba(
    x,
    t
):

    i = int(
        math.floor(
            t
        )
    )

    mu = (
        t - i
    )


    y0 = x[
        i - 1
    ]

    y1 = x[
        i
    ]

    y2 = x[
        i + 1
    ]

    y3 = x[
        i + 2
    ]


    a0 = (
        -0.5
        * y0
        +
        1.5
        * y1
        -
        1.5
        * y2
        +
        0.5
        * y3
    )

    a1 = (
        y0
        -
        2.5
        * y1
        +
        2.0
        * y2
        -
        0.5
        * y3
    )

    a2 = (
        -0.5
        * y0
        +
        0.5
        * y2
    )

    a3 = y1


    return (
        (
            (
                a0
                * mu
                +
                a1
            )
            * mu
            +
            a2
        )
        * mu
        +
        a3
    )


# ============================================================
# M&M - NUMBA
# ============================================================

@njit(cache=True)
def mm_clock_recovery_numba(
    x,
    omega_mid,
    gain_omega,
    mu_initial,
    gain_mu,
    omega_relative_limit
):

    omega = omega_mid

    omega_limit = (
        omega_relative_limit
        * omega_mid
    )

    min_omega = (
        omega_mid
        -
        omega_limit
    )

    max_omega = (
        omega_mid
        +
        omega_limit
    )

    mu = mu_initial


    p_2 = 0.0 + 0.0j
    p_1 = 0.0 + 0.0j
    p_0 = 0.0 + 0.0j

    c_2 = 0.0 + 0.0j
    c_1 = 0.0 + 0.0j
    c_0 = 0.0 + 0.0j


    expected = int(
        len(x)
        /
        omega_mid
        *
        1.02
    ) + 100


    out = np.empty(
        expected,
        dtype=np.complex64
    )


    out_index = 0

    position = 2.0


    while (
        position
        < len(x) - 3
        and
        out_index
        < expected
    ):

        p_2 = p_1

        p_1 = p_0

        c_2 = c_1

        c_1 = c_0


        p_0 = cubic_interpolate_numba(
            x,
            position
        )


        c_0 = complex(
            1.0
            if p_0.real > 0.0
            else 0.0,

            1.0
            if p_0.imag > 0.0
            else 0.0
        )


        phase_error = (
            (
                (
                    p_0
                    -
                    p_2
                )
                *
                c_1.conjugate()
            )
            -
            (
                (
                    c_0
                    -
                    c_2
                )
                *
                p_1.conjugate()
            )
        ).real


        if phase_error > 1.0:

            phase_error = 1.0

        elif phase_error < -1.0:

            phase_error = -1.0


        out[
            out_index
        ] = p_0

        out_index += 1


        omega += (
            gain_omega
            *
            phase_error
        )


        if omega < min_omega:

            omega = min_omega

        elif omega > max_omega:

            omega = max_omega


        mu += (
            omega
            +
            gain_mu
            *
            phase_error
        )


        advance = int(
            math.floor(
                mu
            )
        )


        position += (
            advance
        )


        mu -= (
            advance
        )


    return (
        out[
            :out_index
        ],
        omega
    )


# ============================================================
# OFFSET QPSK x^4
# ============================================================

def estimate_qpsk_offset(
    x,
    fs
):

    if len(x) < 4096:

        return (
            0.0,
            0.0
        )


    nfft = min(
        262_144,
        len(x)
    )


    start = (
        len(x)
        -
        nfft
    ) // 2


    segment = (
        x[
            start:
            start + nfft
        ]
        .astype(
            np.complex64
        )
    )


    rms = np.sqrt(
        np.mean(
            np.abs(
                segment
            ) ** 2
        )
    )


    if rms < 1e-12:

        return (
            0.0,
            0.0
        )


    segment = (
        segment
        /
        rms
    )


    x4 = (
        segment ** 4
    )


    window = np.hanning(
        nfft
    )


    spectrum = np.fft.fftshift(
        np.fft.fft(
            x4
            *
            window
        )
    )


    power = (
        np.abs(
            spectrum
        ) ** 2
    )


    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            nfft,
            d=1.0 / fs
        )
    )


    # QPSK offset aparece multiplicado por 4.
    # ±20 kHz equivale a ±5 kHz de offset real.
    mask = (
        np.abs(
            freqs
        )
        <= 20_000.0
    )


    search_power = (
        power[
            mask
        ]
    )

    search_freqs = (
        freqs[
            mask
        ]
    )


    idx = int(
        np.argmax(
            search_power
        )
    )


    peak = float(
        search_power[
            idx
        ]
    )


    median = float(
        np.median(
            search_power
        )
        +
        1e-20
    )


    quality_db = (
        10.0
        *
        np.log10(
            (
                peak
                +
                1e-20
            )
            /
            median
        )
    )


    offset = (
        float(
            search_freqs[
                idx
            ]
        )
        /
        4.0
    )


    # Si no hay una componente x^4 clara,
    # dejamos que Costas haga el seguimiento.
    if quality_db < 5.0:

        offset = 0.0


    return (
        offset,
        quality_db
    )


# ============================================================
# SYNC QPSK
# ============================================================

def swap_iq_word(
    word
):

    i = (
        word
        &
        0xAAAAAAAAAAAAAAAA
    )

    q = (
        word
        &
        0x5555555555555555
    )


    return (
        (i >> 1)
        |
        (
            (
                q << 1
            )
            &
            MASK64
        )
    )


def rotate64(
    word,
    phase
):

    i = (
        word
        &
        0xAAAAAAAAAAAAAAAA
    )

    q = (
        word
        &
        0x5555555555555555
    )


    if phase == 0:

        pass


    elif phase == 1:

        word = (
            (
                (
                    i
                    ^
                    0xAAAAAAAAAAAAAAAA
                )
                >> 1
            )
            |
            (
                (
                    q << 1
                )
                &
                MASK64
            )
        )


    elif phase == 2:

        word ^= (
            MASK64
        )


    elif phase == 3:

        word = (
            (i >> 1)
            |
            (
                (
                    q
                    ^
                    0x5555555555555555
                )
                << 1
            )
        )


    return (
        (
            (
                word
                &
                0x5555555555555555
            )
            << 1
        )
        |
        (
            (
                word
                &
                0xAAAAAAAAAAAAAAAA
            )
            >> 1
        )
    ) & MASK64


def word_to_bits(
    word
):

    return np.array(
        [
            (
                word
                >>
                (
                    63 - i
                )
            )
            &
            1
            for i in range(
                64
            )
        ],
        dtype=np.uint8
    )


def build_sync_variants():

    variants = []


    for p in range(
        4
    ):

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


    swapped = (
        swap_iq_word(
            SYNC_WORD
        )
        ^
        MASK64
    )


    for p in range(
        4
    ):

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


PATTERN_BITS = [
    word_to_bits(
        word
    )
    for _, _, word in VARIANTS
]


PATTERN_BYTES = [
    np.packbits(
        pattern
    )
    for pattern in PATTERN_BITS
]


def phase_swap_from_variant(
    p
):

    return (
        p % 4,
        p < 4
    )


# ============================================================
# ENCONTRAR ANCLA GLOBAL EFICIENTEMENTE
# ============================================================

def find_global_anchor(
    hard
):

    best_corr = -1

    best_pos = None

    best_variant = None


    # Como el stream es I,Q,I,Q...
    # una trama debe empezar en una posición par.
    #
    # Probamos los posibles offsets pares módulo 8.
    for bit_offset in (
        0,
        2,
        4,
        6
    ):

        available = (
            len(hard)
            -
            bit_offset
        )


        usable_bits = (
            available
            //
            8
            *
            8
        )


        if usable_bits < 64:

            continue


        packed = np.packbits(
            hard[
                bit_offset:
                bit_offset
                +
                usable_bits
            ]
        )


        if len(packed) < 8:

            continue


        windows = (
            np.lib.stride_tricks
            .sliding_window_view(
                packed,
                8
            )
        )


        for variant in range(
            8
        ):

            xor = np.bitwise_xor(
                windows,
                PATTERN_BYTES[
                    variant
                ]
            )


            errors = POPCOUNT[
                xor
            ].sum(
                axis=1
            )


            idx = int(
                np.argmin(
                    errors
                )
            )


            error_count = int(
                errors[
                    idx
                ]
            )


            corr = (
                64
                -
                error_count
            )


            if corr > best_corr:

                best_corr = (
                    corr
                )

                best_variant = (
                    variant
                )

                best_pos = (
                    bit_offset
                    +
                    idx
                    * 8
                )


    return (
        best_pos,
        best_variant,
        best_corr
    )


# ============================================================
# CORRELACIÓN LOCAL
# ============================================================

def correlation_at(
    hard,
    position
):

    block = hard[
        position:
        position + 64
    ]


    if len(block) < 64:

        return [
            -1
            for _ in range(
                8
            )
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
        for pattern in PATTERN_BITS
    ]


def find_local_sync(
    hard,
    expected_position
):

    start = max(
        0,
        expected_position
        -
        LOCAL_SYNC_RADIUS
    )

    end = min(
        len(hard)
        -
        64,
        expected_position
        +
        LOCAL_SYNC_RADIUS
    )


    if (
        start % 2
        != expected_position % 2
    ):

        start += 1


    best_position = (
        expected_position
    )

    best_correlations = None

    best_corr = -1


    for position in range(
        start,
        end + 1,
        2
    ):

        correlations = (
            correlation_at(
                hard,
                position
            )
        )


        local = max(
            correlations
        )


        if local > best_corr:

            best_corr = (
                local
            )

            best_position = (
                position
            )

            best_correlations = (
                correlations
            )


    return (
        best_position,
        best_correlations,
        best_corr
    )


# ============================================================
# ROTATE SOFT
# ============================================================

def rotate_soft(
    data,
    phase,
    iq_swap
):

    x = (
        data
        .astype(
            np.int16
        )
        .copy()
    )


    pairs = x.reshape(
        -1,
        2
    )


    if iq_swap:

        old_i = (
            pairs[
                :,
                0
            ]
            .copy()
        )

        old_q = (
            pairs[
                :,
                1
            ]
            .copy()
        )


        pairs[
            :,
            0
        ] = old_q

        pairs[
            :,
            1
        ] = old_i


    if phase == 1:

        old_i = (
            pairs[
                :,
                0
            ]
            .copy()
        )

        old_q = (
            pairs[
                :,
                1
            ]
            .copy()
        )


        pairs[
            :,
            0
        ] = old_q

        pairs[
            :,
            1
        ] = -old_i


    elif phase == 2:

        pairs *= -1


    elif phase == 3:

        old_i = (
            pairs[
                :,
                0
            ]
            .copy()
        )

        old_q = (
            pairs[
                :,
                1
            ]
            .copy()
        )


        pairs[
            :,
            0
        ] = -old_q

        pairs[
            :,
            1
        ] = old_i


    return np.clip(
        pairs.reshape(
            -1
        ),
        -127,
        127
    ).astype(
        np.int8
    )


# ============================================================
# VITERBI TRELLIS
# ============================================================

def parity(
    x
):

    return (
        x.bit_count()
        &
        1
    )


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
    |
    32
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
        &
        1
    )


    for predecessor, table in (
        (
            int(
                PRED0[
                    next_state
                ]
            ),
            SIGN0
        ),
        (
            int(
                PRED1[
                    next_state
                ]
            ),
            SIGN1
        )
    ):

        register = (
            (
                predecessor
                << 1
            )
            |
            bit
        )


        b0 = parity(
            register
            &
            POLYS[0]
        )

        b1 = parity(
            register
            &
            POLYS[1]
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
# VITERBI NUMBA
# ============================================================

@njit(cache=True)
def viterbi_decode_numba(
    soft,
    pred0,
    pred1,
    sign0,
    sign1
):

    steps = (
        len(soft)
        //
        2
    )


    metrics = np.zeros(
        64,
        dtype=np.float64
    )


    decisions = np.empty(
        (
            steps,
            64
        ),
        dtype=np.uint8
    )


    for t in range(
        steps
    ):

        r0 = float(
            soft[
                2 * t
            ]
        )

        r1 = float(
            soft[
                2 * t + 1
            ]
        )


        new_metrics = np.empty(
            64,
            dtype=np.float64
        )


        for state in range(
            64
        ):

            p0 = int(
                pred0[
                    state
                ]
            )

            p1 = int(
                pred1[
                    state
                ]
            )


            candidate0 = (
                metrics[
                    p0
                ]
                +
                r0
                *
                sign0[
                    state,
                    0
                ]
                +
                r1
                *
                sign0[
                    state,
                    1
                ]
            )


            candidate1 = (
                metrics[
                    p1
                ]
                +
                r0
                *
                sign1[
                    state,
                    0
                ]
                +
                r1
                *
                sign1[
                    state,
                    1
                ]
            )


            if candidate1 > candidate0:

                new_metrics[
                    state
                ] = (
                    candidate1
                )

                decisions[
                    t,
                    state
                ] = 1

            else:

                new_metrics[
                    state
                ] = (
                    candidate0
                )

                decisions[
                    t,
                    state
                ] = 0


        metrics = (
            new_metrics
        )


    state = 0

    best_metric = (
        metrics[0]
    )


    for s in range(
        1,
        64
    ):

        if (
            metrics[s]
            >
            best_metric
        ):

            best_metric = (
                metrics[s]
            )

            state = s


    decoded = np.empty(
        steps,
        dtype=np.uint8
    )


    for t in range(
        steps - 1,
        -1,
        -1
    ):

        decoded[
            t
        ] = (
            state
            &
            1
        )


        high = int(
            decisions[
                t,
                state
            ]
        )


        state = (
            (
                state
                >> 1
            )
            |
            (
                high
                << 5
            )
        )


    return decoded


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


    output[
        4:
    ] ^= (
        PN_FRAME
    )


    return output


# ============================================================
# REED SOLOMON
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


    if all(
        x == 0
        for x in syndromes[
            1:
        ]
    ):

        return (
            True,
            codeword.copy(),
            0
        )


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
            bytes(
                message
            )
            +
            bytes(
                ecc
            ),
            dtype=np.uint8
        ).copy()


        after = rs.rs_calc_syndromes(
            bytearray(
                corrected.tobytes()
            ),
            NSYM,
            fcr=FCR,
            generator=GENERATOR
        )


        ok = all(
            x == 0
            for x in after[
                1:
            ]
        )


        return (
            ok,
            corrected,
            len(
                errata_pos
            )
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
        output[
            4:
        ]
        .copy()
    )


    errors = []


    for branch in range(
        4
    ):

        codeword = area[
            branch::4
        ]


        (
            ok,
            corrected,
            error_count
        ) = decode_rs_codeword(
            codeword
        )


        errors.append(
            error_count
        )


        if not ok:

            return (
                False,
                frame,
                errors
            )


        area[
            branch::4
        ] = corrected


    output[
        :4
    ] = ASM


    output[
        4:
    ] = area


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
        >>
        6
    )


    spacecraft_id = int(
        (
            (
                int(
                    cadu[4]
                )
                &
                0x3F
            )
            << 2
        )
        |
        (
            int(
                cadu[5]
            )
            >> 6
        )
    )


    vcid = int(
        cadu[5]
        &
        0x3F
    )


    counter = (
        (
            int(
                cadu[6]
            )
            << 16
        )
        |
        (
            int(
                cadu[7]
            )
            << 8
        )
        |
        int(
            cadu[8]
        )
    )


    return (
        version,
        spacecraft_id,
        vcid,
        counter
    )


# ============================================================
# DECODIFICAR SOFT DE UN CHUNK
# ============================================================

def decode_soft_chunk(
    soft,
    chunk_index,
    usable_start_time
):

    hard = (
        soft > 0
    ).astype(
        np.uint8
    )


    (
        anchor_pos,
        anchor_variant,
        anchor_corr
    ) = find_global_anchor(
        hard
    )


    if (
        anchor_pos is None
        or anchor_corr
        < MIN_SYNC_CORRELATION
    ):

        return (
            [],
            {
                "anchor_corr":
                    (
                        -1
                        if anchor_pos is None
                        else anchor_corr
                    ),

                "anchor_pos":
                    (
                        ""
                        if anchor_pos is None
                        else anchor_pos
                    ),

                "candidates": 0,

                "valid": 0
            }
        )


    # --------------------------------------------------------
    # Construir retícula nominal
    # --------------------------------------------------------

    first = (
        anchor_pos
    )


    while (
        first
        -
        FRAME_SOFT_BITS
        >= 0
    ):

        first -= (
            FRAME_SOFT_BITS
        )


    positions = []

    expected = (
        first
    )


    while (
        expected
        +
        FRAME_SOFT_BITS
        <= len(
            soft
        )
    ):

        positions.append(
            expected
        )

        expected += (
            FRAME_SOFT_BITS
        )


    valid_frames = []

    last_valid_variant = None


    for frame_number, expected in enumerate(
        positions
    ):

        (
            position,
            correlations,
            local_best_corr
        ) = find_local_sync(
            hard,
            expected
        )


        if (
            correlations is None
            or local_best_corr
            < MIN_SYNC_CORRELATION
        ):

            continue


        ordered = list(
            np.argsort(
                correlations
            )[::-1]
        )


        trial_variants = []


        for variant in ordered[
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


        if (
            last_valid_variant
            is not None
            and
            last_valid_variant
            not in trial_variants
        ):

            trial_variants.append(
                last_valid_variant
            )


        raw_frame = soft[
            position:
            position
            +
            FRAME_SOFT_BITS
        ]


        if (
            len(
                raw_frame
            )
            != FRAME_SOFT_BITS
        ):

            continue


        accepted = None


        for variant in trial_variants:

            (
                phase,
                iq_swap
            ) = phase_swap_from_variant(
                variant
            )


            corrected_soft = rotate_soft(
                raw_frame,
                phase,
                iq_swap
            )


            decoded_bits = viterbi_decode_numba(
                corrected_soft,
                PRED0,
                PRED1,
                SIGN0,
                SIGN1
            )


            frame = np.packbits(
                decoded_bits
            )


            if (
                len(
                    frame
                )
                != FRAME_BYTES
            ):

                continue


            frame = derandomize(
                frame
            )


            if (
                frame[9]
                ==
                0xFF
            ):

                frame ^= (
                    0xFF
                )


            (
                rs_ok,
                corrected_frame,
                rs_errors
            ) = rs_correct_frame(
                frame
            )


            if not rs_ok:

                continue


            (
                version,
                spacecraft_id,
                vcid,
                counter
            ) = parse_vcdu(
                corrected_frame
            )


            if version > 1:

                continue


            accepted = {
                "chunk":
                    chunk_index,

                "frame_number":
                    frame_number,

                "soft_position":
                    position,

                "expected_position":
                    expected,

                "sync_delta":
                    position
                    -
                    expected,

                "sync_corr":
                    int(
                        correlations[
                            variant
                        ]
                    ),

                "variant":
                    VARIANTS[
                        variant
                    ][1],

                "variant_id":
                    variant,

                "rs_errors":
                    list(
                        rs_errors
                    ),

                "rs_total":
                    sum(
                        max(
                            0,
                            x
                        )
                        for x in rs_errors
                    ),

                "version":
                    version,

                "spacecraft_id":
                    spacecraft_id,

                "vcid":
                    vcid,

                "counter":
                    counter,

                "counter_hex":
                    f"0x{counter:06X}",

                "time_sec":
                    (
                        usable_start_time
                        +
                        position
                        /
                        (
                            2.0
                            *
                            SYMBOL_RATE
                        )
                    ),

                "frame":
                    corrected_frame.copy()
            }


            break


        if accepted is not None:

            valid_frames.append(
                accepted
            )


            last_valid_variant = (
                accepted[
                    "variant_id"
                ]
            )


    return (
        valid_frames,
        {
            "anchor_corr":
                anchor_corr,

            "anchor_pos":
                anchor_pos,

            "candidates":
                len(
                    positions
                ),

            "valid":
                len(
                    valid_frames
                )
        }
    )


# ============================================================
# DECODIFICAR UN CHUNK COMPLETO DESDE I/Q
# ============================================================

def process_chunk(
    wav_data,
    input_fs,
    start_sample,
    end_sample,
    chunk_index
):

    chunk_start_time = (
        start_sample
        /
        input_fs
    )

    chunk_end_time = (
        end_sample
        /
        input_fs
    )

    duration = (
        chunk_end_time
        -
        chunk_start_time
    )


    print()
    print(
        "================================================"
    )

    print(
        f"CHUNK {chunk_index:03d}"
    )

    print(
        "================================================"
    )

    print(
        f"Tiempo : "
        f"{chunk_start_time:8.3f}"
        f" -> "
        f"{chunk_end_time:8.3f} s"
    )

    print(
        f"Duración: "
        f"{duration:.3f} s"
    )


    # ========================================================
    # I/Q
    # ========================================================

    raw = np.asarray(
        wav_data[
            start_sample:
            end_sample
        ],
        dtype=np.float32
    )


    if (
        raw.ndim != 2
        or raw.shape[1] < 2
    ):

        raise RuntimeError(
            "El WAV debe tener dos canales I/Q."
        )


    iq = (
        raw[
            :,
            0
        ]
        +
        1j
        *
        raw[
            :,
            1
        ]
    ).astype(
        np.complex64
    )


    del raw


    iq -= np.mean(
        iq
    )


    rms = np.sqrt(
        np.mean(
            np.abs(
                iq
            ) ** 2
        )
    )


    if rms < 1e-12:

        return (
            [],
            {
                "chunk":
                    chunk_index,

                "start_sec":
                    chunk_start_time,

                "end_sec":
                    chunk_end_time,

                "duration":
                    duration,

                "cfo_hz":
                    0,

                "cfo_quality_db":
                    0,

                "symbols":
                    0,

                "soft_values":
                    0,

                "anchor_corr":
                    -1,

                "candidates":
                    0,

                "valid":
                    0
            }
        )


    iq /= (
        rms
    )


    # ========================================================
    # LOWPASS
    # ========================================================

    filtered = signal.lfilter(
        LOWPASS_FILTER,
        [1.0],
        iq
    ).astype(
        np.complex64
    )


    del iq


    # ========================================================
    # RESAMPLE 500 k -> 288 k
    # ========================================================

    resampled = signal.resample_poly(
        filtered,
        72,
        125
    ).astype(
        np.complex64
    )


    del filtered


    # ========================================================
    # OFFSET DE PORTADORA x^4
    # ========================================================

    (
        coarse_offset,
        offset_quality
    ) = estimate_qpsk_offset(
        resampled,
        OUTPUT_FS
    )


    print(
        f"CFO grueso : "
        f"{coarse_offset:+9.3f} Hz"
        f" | "
        f"x4={offset_quality:.2f} dB"
    )


    if abs(
        coarse_offset
    ) > 0.001:

        n = np.arange(
            len(
                resampled
            ),
            dtype=np.float32
        )


        correction = np.exp(
            -1j
            *
            2.0
            *
            np.pi
            *
            coarse_offset
            *
            n
            /
            OUTPUT_FS
        ).astype(
            np.complex64
        )


        resampled *= (
            correction
        )


        del (
            n,
            correction
        )


    # ========================================================
    # RRC
    # ========================================================

    rrc = signal.lfilter(
        RRC_FILTER,
        [1.0],
        resampled
    ).astype(
        np.complex64
    )


    del resampled


    rrc_rms = np.sqrt(
        np.mean(
            np.abs(
                rrc
            ) ** 2
        )
    )


    if rrc_rms > 1e-12:

        rrc /= (
            rrc_rms
        )


    # ========================================================
    # COSTAS
    # ========================================================

    costas, final_freq = (
        costas_qpsk_numba(
            rrc,
            COSTAS_BW,
            COSTAS_FREQ_LIMIT
        )
    )


    del rrc


    final_freq_hz = (
        final_freq
        *
        OUTPUT_FS
        /
        (
            2.0
            *
            np.pi
        )
    )


    # ========================================================
    # M&M
    # ========================================================

    symbols, final_omega = (
        mm_clock_recovery_numba(
            costas,
            SPS,
            CLOCK_GAIN_OMEGA,
            CLOCK_MU_INITIAL,
            CLOCK_GAIN_MU,
            CLOCK_OMEGA_REL_LIMIT
        )
    )


    del costas


    print(
        f"Símbolos M&M: "
        f"{len(symbols):,}"
        f" | "
        f"omega={final_omega:.7f}"
        f" | "
        f"Costas final={final_freq_hz:+.2f} Hz"
    )


    # ========================================================
    # QUITAR EXTREMOS DEL CHUNK
    # ========================================================

    discard_symbols = int(
        EDGE_DISCARD_SECONDS
        *
        SYMBOL_RATE
    )


    if (
        len(
            symbols
        )
        >
        2
        *
        discard_symbols
    ):

        symbols = symbols[
            discard_symbols:
            -
            discard_symbols
        ]


        usable_start_time = (
            chunk_start_time
            +
            EDGE_DISCARD_SECONDS
        )

    else:

        usable_start_time = (
            chunk_start_time
        )


    # ========================================================
    # NORMALIZACIÓN SOFT
    # ========================================================

    symbol_rms = np.sqrt(
        np.mean(
            np.abs(
                symbols
            ) ** 2
        )
    )


    if symbol_rms > 1e-12:

        symbols /= (
            symbol_rms
        )


    soft_float = np.empty(
        len(
            symbols
        )
        *
        2,
        dtype=np.float32
    )


    soft_float[
        0::2
    ] = (
        symbols.real
    )


    soft_float[
        1::2
    ] = (
        symbols.imag
    )


    del symbols


    p99 = np.percentile(
        np.abs(
            soft_float
        ),
        99
    )


    scale = (
        120.0
        /
        (
            p99
            +
            1e-12
        )
    )


    soft = np.clip(
        np.rint(
            soft_float
            *
            scale
        ),
        -127,
        127
    ).astype(
        np.int8
    )


    del soft_float


    # ========================================================
    # FRAMING + FEC
    # ========================================================

    frames, decoder_stats = (
        decode_soft_chunk(
            soft,
            chunk_index,
            usable_start_time
        )
    )


    print(
        f"Anchor      : "
        f"{decoder_stats['anchor_corr']}/64"
    )

    print(
        f"Candidatas  : "
        f"{decoder_stats['candidates']}"
    )

    print(
        f"CADUs válidas: "
        f"{decoder_stats['valid']}"
    )


    if frames:

        print(
            f"Contadores  : "
            f"{frames[0]['counter_hex']}"
            f" -> "
            f"{frames[-1]['counter_hex']}"
        )


    # ========================================================
    # GUARDAR CADUs DEL CHUNK
    # ========================================================

    chunk_cadu_file = (
        CHUNKS_DIR
        /
        f"chunk_{chunk_index:03d}_valid.cadu"
    )


    if frames:

        np.concatenate(
            [
                record[
                    "frame"
                ]
                for record in frames
            ]
        ).astype(
            np.uint8
        ).tofile(
            chunk_cadu_file
        )

    else:

        chunk_cadu_file.write_bytes(
            b""
        )


    stats = {
        "chunk":
            chunk_index,

        "start_sec":
            chunk_start_time,

        "end_sec":
            chunk_end_time,

        "duration":
            duration,

        "cfo_hz":
            coarse_offset,

        "cfo_quality_db":
            offset_quality,

        "costas_final_hz":
            final_freq_hz,

        "mm_omega":
            final_omega,

        "soft_values":
            len(
                soft
            ),

        "anchor_corr":
            decoder_stats[
                "anchor_corr"
            ],

        "candidates":
            decoder_stats[
                "candidates"
            ],

        "valid":
            decoder_stats[
                "valid"
            ]
    }


    del soft

    gc.collect()


    return (
        frames,
        stats
    )


# ============================================================
# MAIN
# ============================================================

print()
print(
    "================================================"
)

print(
    "   METEOR LRPT - FULL PASS PIPELINE"
)

print(
    "================================================"
)

print()

print(
    "Entrada:"
)

print(
    INPUT_WAV
)

print()


if not INPUT_WAV.exists():

    raise FileNotFoundError(
        INPUT_WAV
    )


# ============================================================
# WAV MEMORY-MAPPED
# ============================================================

input_fs, wav_data = wavfile.read(
    INPUT_WAV,
    mmap=True
)


if (
    input_fs
    != EXPECTED_INPUT_FS
):

    raise RuntimeError(
        f"Sample rate inesperado: {input_fs}"
    )


if (
    wav_data.ndim != 2
    or wav_data.shape[1] < 2
):

    raise RuntimeError(
        "Se esperaba WAV I/Q estéreo."
    )


total_samples = (
    wav_data.shape[0]
)

duration = (
    total_samples
    /
    input_fs
)


print(
    f"Sample rate : "
    f"{input_fs:,} Hz"
)

print(
    f"Muestras    : "
    f"{total_samples:,}"
)

print(
    f"Duración    : "
    f"{duration:.3f} s"
)

print(
    f"Canales WAV : "
    f"{wav_data.shape[1]}"
)

print()


# ============================================================
# CHUNKS
# ============================================================

chunk_samples = int(
    CHUNK_SECONDS
    *
    input_fs
)

hop_samples = int(
    HOP_SECONDS
    *
    input_fs
)


starts = list(
    range(
        0,
        total_samples,
        hop_samples
    )
)


print(
    f"Chunks programados: "
    f"{len(starts)}"
)

print()


# ============================================================
# PRIMERA LLAMADA NUMBA
#
# Compila kernels antes de comenzar la pasada.
# ============================================================

print(
    "Compilando kernels Numba..."
)

_dummy = np.ones(
    100,
    dtype=np.complex64
)

costas_qpsk_numba(
    _dummy,
    COSTAS_BW,
    COSTAS_FREQ_LIMIT
)

mm_clock_recovery_numba(
    _dummy,
    SPS,
    CLOCK_GAIN_OMEGA,
    CLOCK_MU_INITIAL,
    CLOCK_GAIN_MU,
    CLOCK_OMEGA_REL_LIMIT
)


_dummy_soft = np.ones(
    FRAME_SOFT_BITS,
    dtype=np.int8
)

viterbi_decode_numba(
    _dummy_soft,
    PRED0,
    PRED1,
    SIGN0,
    SIGN1
)

del (
    _dummy,
    _dummy_soft
)

print(
    "Numba listo."
)

print()


# ============================================================
# RECORRER PASADA
# ============================================================

all_records = []

chunk_stats = []

start_wall = time.time()


for chunk_index, start_sample in enumerate(
    starts
):

    end_sample = min(
        start_sample
        +
        chunk_samples,
        total_samples
    )


    chunk_duration = (
        (
            end_sample
            -
            start_sample
        )
        /
        input_fs
    )


    if (
        chunk_duration
        <
        MIN_CHUNK_SECONDS
    ):

        continue


    chunk_begin = (
        time.time()
    )


    try:

        (
            records,
            stats
        ) = process_chunk(
            wav_data,
            input_fs,
            start_sample,
            end_sample,
            chunk_index
        )


        all_records.extend(
            records
        )


        chunk_stats.append(
            stats
        )


    except Exception as exc:

        print()
        print(
            f"ERROR EN CHUNK "
            f"{chunk_index:03d}: "
            f"{exc}"
        )

        print()


        chunk_stats.append(
            {
                "chunk":
                    chunk_index,

                "start_sec":
                    start_sample
                    /
                    input_fs,

                "end_sec":
                    end_sample
                    /
                    input_fs,

                "duration":
                    chunk_duration,

                "cfo_hz":
                    "",

                "cfo_quality_db":
                    "",

                "costas_final_hz":
                    "",

                "mm_omega":
                    "",

                "soft_values":
                    "",

                "anchor_corr":
                    "",

                "candidates":
                    "",

                "valid":
                    0
            }
        )


    elapsed_chunk = (
        time.time()
        -
        chunk_begin
    )


    elapsed_total = (
        time.time()
        -
        start_wall
    )


    print(
        f"Tiempo chunk : "
        f"{elapsed_chunk:.1f} s"
    )

    print(
        f"Tiempo total : "
        f"{elapsed_total / 60:.1f} min"
    )


# ============================================================
# DEDUPLICAR CADUs
#
# Como hay solape entre chunks, veremos muchas CADUs dos veces.
#
# Key:
#   (VCID, VCDU counter)
#
# Elegimos:
#   1. menos correcciones RS
#   2. mayor correlación de sync
# ============================================================

print()
print(
    "================================================"
)

print(
    "              DEDUPLICANDO CADUs"
)

print(
    "================================================"
)

print()


best_by_counter = {}


for record in all_records:

    key = (
        record[
            "vcid"
        ],
        record[
            "counter"
        ]
    )


    rank = (
        -record[
            "rs_total"
        ],
        record[
            "sync_corr"
        ]
    )


    if (
        key
        not in best_by_counter
    ):

        record[
            "_rank"
        ] = rank

        best_by_counter[
            key
        ] = record


    else:

        previous = (
            best_by_counter[
                key
            ]
        )


        if (
            rank
            >
            previous[
                "_rank"
            ]
        ):

            record[
                "_rank"
            ] = rank

            best_by_counter[
                key
            ] = record


unique_records = list(
    best_by_counter.values()
)


# Orden temporal real de la grabación
unique_records.sort(
    key=lambda r: r[
        "time_sec"
    ]
)


for record in unique_records:

    record.pop(
        "_rank",
        None
    )


print(
    f"Antes de deduplicar : "
    f"{len(all_records)}"
)

print(
    f"CADUs únicas        : "
    f"{len(unique_records)}"
)


# ============================================================
# GUARDAR CADU COMPLETO
# ============================================================

if unique_records:

    np.concatenate(
        [
            record[
                "frame"
            ]
            for record in unique_records
        ]
    ).astype(
        np.uint8
    ).tofile(
        FINAL_CADU
    )

else:

    FINAL_CADU.write_bytes(
        b""
    )


# ============================================================
# CSV FRAMES
# ============================================================

frame_fields = [
    "time_sec",
    "chunk",
    "frame_number",
    "soft_position",
    "expected_position",
    "sync_delta",
    "sync_corr",
    "variant",
    "rs_errors",
    "rs_total",
    "version",
    "spacecraft_id",
    "vcid",
    "counter",
    "counter_hex"
]


with open(
    FINAL_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=frame_fields
    )

    writer.writeheader()


    for record in unique_records:

        row = {
            field:
                record[
                    field
                ]
            for field in frame_fields
        }


        row[
            "rs_errors"
        ] = ",".join(
            map(
                str,
                record[
                    "rs_errors"
                ]
            )
        )


        writer.writerow(
            row
        )


# ============================================================
# CSV CHUNKS
# ============================================================

chunk_fields = [
    "chunk",
    "start_sec",
    "end_sec",
    "duration",
    "cfo_hz",
    "cfo_quality_db",
    "costas_final_hz",
    "mm_omega",
    "soft_values",
    "anchor_corr",
    "candidates",
    "valid"
]


with open(
    CHUNKS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=chunk_fields
    )

    writer.writeheader()

    writer.writerows(
        chunk_stats
    )


# ============================================================
# ESTADÍSTICAS DE CONTINUIDAD
# ============================================================

vcid5 = [
    record
    for record in unique_records
    if record[
        "vcid"
    ] == 5
]


gaps = 0

missing_estimate = 0

longest_run = 0

current_run = 0

previous = None


for record in vcid5:

    counter = (
        record[
            "counter"
        ]
    )


    if previous is None:

        current_run = 1


    else:

        delta = (
            (
                counter
                -
                previous
            )
            &
            0xFFFFFF
        )


        if delta == 1:

            current_run += 1

        else:

            gaps += 1

            current_run = 1


            # Solo lo tratamos como hueco razonable si
            # el salto no es absurdo.
            if (
                1
                <
                delta
                <
                1000
            ):

                missing_estimate += (
                    delta
                    -
                    1
                )


    longest_run = max(
        longest_run,
        current_run
    )


    previous = (
        counter
    )


# ============================================================
# RESUMEN POR CHUNK
# ============================================================

print()
print(
    "================================================"
)

print(
    "               RESUMEN POR CHUNK"
)

print(
    "================================================"
)

print()


for stats in chunk_stats:

    print(
        f"{stats['chunk']:03d} | "
        f"{stats['start_sec']:7.1f}"
        f"-"
        f"{stats['end_sec']:7.1f}s | "
        f"sync={str(stats['anchor_corr']):>2} | "
        f"CADUs={stats['valid']:3d} | "
        f"CFO={str(stats['cfo_hz']):>9}"
    )


# ============================================================
# RESUMEN FINAL
# ============================================================

elapsed_total = (
    time.time()
    -
    start_wall
)


print()
print(
    "================================================"
)

print(
    "                  RESUMEN FINAL"
)

print(
    "================================================"
)

print()


print(
    f"Duración grabación       : "
    f"{duration:.3f} s"
)

print(
    f"Chunks procesados        : "
    f"{len(chunk_stats)}"
)

print(
    f"CADUs antes dedupe       : "
    f"{len(all_records)}"
)

print(
    f"CADUs únicas             : "
    f"{len(unique_records)}"
)

print(
    f"CADUs VCID 5             : "
    f"{len(vcid5)}"
)

print(
    f"Huecos contador          : "
    f"{gaps}"
)

print(
    f"CADUs faltantes estimadas: "
    f"{missing_estimate}"
)

print(
    f"Racha consecutiva máxima : "
    f"{longest_run}"
)


if vcid5:

    print()

    print(
        f"Primer contador          : "
        f"{vcid5[0]['counter_hex']}"
    )

    print(
        f"Último contador          : "
        f"{vcid5[-1]['counter_hex']}"
    )

    print(
        f"Primer tiempo            : "
        f"{vcid5[0]['time_sec']:.3f} s"
    )

    print(
        f"Último tiempo            : "
        f"{vcid5[-1]['time_sec']:.3f} s"
    )


print()

print(
    f"Tiempo procesamiento     : "
    f"{elapsed_total / 60:.2f} min"
)

print()

print(
    "CADU completo:"
)

print(
    FINAL_CADU
)

print()

print(
    "CSV frames:"
)

print(
    FINAL_CSV
)

print()

print(
    "CSV chunks:"
)

print(
    CHUNKS_CSV
)

print()

print(
    "Chunks individuales:"
)

print(
    CHUNKS_DIR
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