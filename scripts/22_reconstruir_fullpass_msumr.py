from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import csv

import numpy as np
from scipy.fft import idctn
from PIL import Image


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CADU_FILE = (
    ROOT
    / "outputs"
    / "decoded"
    / "fullpass"
    / "meteor_lrpt_full.cadu"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "images"
    / "msumr_fullpass"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PACKET_CSV = (
    OUTPUT_DIR
    / "fullpass_msumr_packets.csv"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CADU_SIZE = 1024

MPDU_DATA_SIZE = 882

HEADER_LENGTH = 6


# Un salto mayor que esto no lo tratamos como pérdida normal:
# probablemente sea una CADU falsa/outlier.
MAX_REASONABLE_VCDU_GAP = 512


# MSU-MR
SEGMENT_WIDTH = 112

SEGMENT_HEIGHT = 8

SEGMENTS_PER_LINE = 14

FULL_WIDTH = (
    SEGMENT_WIDTH
    * SEGMENTS_PER_LINE
)


# SatDump usa un límite equivalente para evitar
# imágenes absurdamente grandes por datos corruptos.
MAX_SEGMENT_SPAN = 20_000


# Rellenar huecos verticales pequeños únicamente
# para las versiones visuales *_filled.
FILL_MISSING = True

MAX_FILL_ROWS = 32


# RGB
ALIGN_STEP = 8

MAX_RGB_VERTICAL_SHIFT = 32

LOW_PERCENTILE = 1.0

HIGH_PERCENTILE = 99.0

GAMMA = 0.90


# ============================================================
# CUANTIZACIÓN
# ============================================================

BASE_QTABLE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
], dtype=np.float64)


# ============================================================
# ZIG-ZAG
# ============================================================

ZIGZAG = np.array([
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
], dtype=np.int32)


# ============================================================
# HUFFMAN JPEG LUMINANCIA
# ============================================================

DC_BITS = [
    0, 1, 5, 1,
    1, 1, 1, 1,
    1, 0, 0, 0,
    0, 0, 0, 0
]

DC_VALUES = list(
    range(12)
)


AC_BITS = [
    0, 2, 1, 3,
    3, 2, 4, 3,
    5, 5, 4, 4,
    0, 0, 1, 125
]


AC_VALUES = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]


# ============================================================
# HUFFMAN CANÓNICO
# ============================================================

def build_huffman_table(
    length_counts,
    symbols
):

    table = {}

    code = 0

    symbol_index = 0

    for length, count in enumerate(
        length_counts,
        start=1
    ):

        for _ in range(
            count
        ):

            table[
                (
                    length,
                    code
                )
            ] = symbols[
                symbol_index
            ]

            symbol_index += 1

            code += 1

        code <<= 1

    if symbol_index != len(
        symbols
    ):

        raise RuntimeError(
            "Tabla Huffman inconsistente."
        )

    return table


DC_TABLE = build_huffman_table(
    DC_BITS,
    DC_VALUES
)

AC_TABLE = build_huffman_table(
    AC_BITS,
    AC_VALUES
)


# ============================================================
# ESTRUCTURAS
# ============================================================

@dataclass
class VCDUHeader:

    version: int

    spacecraft_id: int

    vcid: int

    counter: int


@dataclass
class CCSDSHeader:

    version: int

    packet_type: int

    secondary_header: int

    apid: int

    sequence_flag: int

    sequence_count: int

    packet_length: int

    raw: bytes


# ============================================================
# BIT READER
# ============================================================

class BitReader:

    def __init__(
        self,
        data
    ):

        self.data = bytes(
            data
        )

        self.position = 0

        self.total_bits = (
            len(
                self.data
            )
            * 8
        )


    def read_bit(
        self
    ):

        if (
            self.position
            >= self.total_bits
        ):

            raise EOFError


        byte_index = (
            self.position
            // 8
        )

        bit_index = (
            7
            -
            (
                self.position
                % 8
            )
        )


        value = (
            self.data[
                byte_index
            ]
            >> bit_index
        ) & 1


        self.position += 1

        return value


    def read_bits(
        self,
        count
    ):

        value = 0

        for _ in range(
            count
        ):

            value = (
                value << 1
            ) | self.read_bit()

        return value


# ============================================================
# HUFFMAN
# ============================================================

def decode_huffman(
    reader,
    table
):

    code = 0

    for length in range(
        1,
        17
    ):

        code = (
            code << 1
        ) | reader.read_bit()


        key = (
            length,
            code
        )


        if key in table:

            return table[
                key
            ]


    raise ValueError(
        "Código Huffman desconocido."
    )


def receive_extend(
    reader,
    size
):

    if size == 0:

        return 0


    value = reader.read_bits(
        size
    )


    limit = (
        1
        << (
            size - 1
        )
    )


    if value < limit:

        value -= (
            (
                1
                << size
            )
            - 1
        )


    return value


# ============================================================
# CUANTIZACIÓN
# ============================================================

def get_quantization_table(
    qf
):

    qf = float(
        qf
    )


    if (
        qf >= 20
        and qf < 50
    ):

        scale = (
            5000.0
            / qf
        )

    else:

        scale = (
            200.0
            -
            2.0
            * qf
        )


    table = np.floor(
        (
            scale
            / 100.0
            * BASE_QTABLE
        )
        + 0.5
    )


    table[
        table < 1
    ] = 1


    return table


# ============================================================
# DCT BLOCK
# ============================================================

def decode_block(
    reader,
    last_dc,
    quant_table
):

    coefficients = np.zeros(
        64,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # DC
    # --------------------------------------------------------

    dc_size = decode_huffman(
        reader,
        DC_TABLE
    )


    dc_delta = receive_extend(
        reader,
        dc_size
    )


    dc_value = (
        last_dc
        +
        dc_delta
    )


    coefficients[0] = (
        dc_value
    )


    # --------------------------------------------------------
    # AC
    # --------------------------------------------------------

    index = 1


    while index < 64:

        symbol = decode_huffman(
            reader,
            AC_TABLE
        )


        # EOB
        if symbol == 0x00:

            break


        # ZRL
        if symbol == 0xF0:

            index += 16

            if index > 64:

                raise ValueError(
                    "ZRL inválido."
                )

            continue


        run = (
            symbol
            >> 4
        )

        size = (
            symbol
            & 0x0F
        )


        index += (
            run
        )


        if index >= 64:

            raise ValueError(
                "AC fuera de rango."
            )


        coefficients[
            index
        ] = receive_extend(
            reader,
            size
        )


        index += 1


    # --------------------------------------------------------
    # ZIGZAG + DEQUANT
    # --------------------------------------------------------

    natural = np.zeros(
        64,
        dtype=np.float64
    )


    for x in range(
        64
    ):

        natural[x] = (
            coefficients[
                ZIGZAG[x]
            ]
            *
            quant_table[x]
        )


    natural = natural.reshape(
        8,
        8
    )


    # --------------------------------------------------------
    # IDCT
    # --------------------------------------------------------

    spatial = idctn(
        natural,
        type=2,
        norm="ortho"
    )


    pixels = np.clip(
        np.rint(
            spatial
            + 128.0
        ),
        0,
        255
    ).astype(
        np.uint8
    )


    return (
        pixels,
        dc_value
    )


# ============================================================
# SEGMENTO MSU-MR
# ============================================================

def decode_segment(
    compressed_data,
    qf
):

    reader = BitReader(
        compressed_data
    )


    quant_table = (
        get_quantization_table(
            qf
        )
    )


    segment = np.zeros(
        (
            SEGMENT_HEIGHT,
            SEGMENT_WIDTH
        ),
        dtype=np.uint8
    )


    last_dc = 0

    decoded_blocks = 0


    for block_index in range(
        14
    ):

        try:

            (
                block,
                last_dc
            ) = decode_block(
                reader,
                last_dc,
                quant_table
            )


        except (
            EOFError,
            ValueError
        ):

            return (
                segment,
                decoded_blocks,
                True
            )


        x0 = (
            block_index
            * 8
        )

        x1 = (
            x0
            + 8
        )


        segment[
            :,
            x0:x1
        ] = block


        decoded_blocks += 1


    return (
        segment,
        decoded_blocks,
        False
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
                int(
                    cadu[4]
                )
                & 0x3F
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
        & 0x3F
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


    return VCDUHeader(
        version=version,
        spacecraft_id=spacecraft_id,
        vcid=vcid,
        counter=counter
    )


# ============================================================
# M-PDU
# ============================================================

def parse_mpdu(
    cadu
):

    fhp = (
        (
            int(
                cadu[12]
            )
            & 0x07
        )
        << 8
    ) | int(
        cadu[13]
    )


    data = cadu[
        14:
        14
        +
        MPDU_DATA_SIZE
    ]


    return (
        fhp,
        data
    )


# ============================================================
# CCSDS HEADER
# ============================================================

def parse_ccsds_header(
    raw
):

    if len(raw) < 6:

        return None


    b0 = int(
        raw[0]
    )

    b1 = int(
        raw[1]
    )

    b2 = int(
        raw[2]
    )

    b3 = int(
        raw[3]
    )

    b4 = int(
        raw[4]
    )

    b5 = int(
        raw[5]
    )


    version = (
        b0 >> 5
    )


    packet_type = (
        b0 >> 4
    ) & 1


    secondary_header = (
        b0 >> 3
    ) & 1


    apid = (
        (
            b0
            & 0x07
        )
        << 8
    ) | b1


    sequence_flag = (
        b2 >> 6
    )


    sequence_count = (
        (
            b2
            & 0x3F
        )
        << 8
    ) | b3


    packet_length = (
        b4 << 8
    ) | b5


    return CCSDSHeader(
        version=version,
        packet_type=packet_type,
        secondary_header=secondary_header,
        apid=apid,
        sequence_flag=sequence_flag,
        sequence_count=sequence_count,
        packet_length=packet_length,
        raw=bytes(
            raw[:6]
        )
    )


# ============================================================
# AOS DEMUX
# ============================================================

class AOSDemuxer:

    def __init__(
        self
    ):

        self.reset()


    def reset(
        self
    ):

        self.working_on_packet = False

        self.current_header = None

        self.current_payload = bytearray()

        self.current_payload_length = 0

        self.remaining_payload_length = 0

        self.total_packet_length = 0

        self.in_header = False

        self.header_buffer = bytearray(
            HEADER_LENGTH
        )

        self.in_header_buffer = 0


    def abort_packet(
        self
    ):

        self.working_on_packet = False

        self.current_header = None

        self.current_payload = bytearray()

        self.current_payload_length = 0

        self.remaining_payload_length = 0

        self.total_packet_length = 0


    def read_packet_header(
        self,
        raw_header
    ):

        header = parse_ccsds_header(
            raw_header
        )


        if (
            header is None
            or header.version != 0
        ):

            self.abort_packet()

            return False


        payload_length = (
            header.packet_length
            + 1
        )


        if (
            payload_length <= 0
            or payload_length > 65536
        ):

            self.abort_packet()

            return False


        self.working_on_packet = True

        self.current_header = header

        self.current_payload = bytearray()

        self.current_payload_length = (
            payload_length
        )

        self.remaining_payload_length = (
            payload_length
        )

        self.total_packet_length = (
            HEADER_LENGTH
            +
            payload_length
        )


        return True


    def push_payload(
        self,
        data
    ):

        if not self.working_on_packet:

            return


        length = min(
            len(data),
            self.remaining_payload_length
        )


        if length <= 0:

            return


        self.current_payload.extend(
            bytes(
                data[:length]
            )
        )


        self.remaining_payload_length -= (
            length
        )


    def finish_packet(
        self
    ):

        if (
            not self.working_on_packet
            or self.current_header is None
        ):

            return None


        packet = (
            self.current_header.raw
            +
            bytes(
                self.current_payload
            )
        )


        result = (
            self.current_header,
            packet
        )


        self.abort_packet()


        return result


    def process(
        self,
        cadu
    ):

        packets = []


        (
            fhp,
            data
        ) = parse_mpdu(
            cadu
        )


        if (
            fhp < 2047
            and fhp >= MPDU_DATA_SIZE
        ):

            return packets


        offset = 0


        # ----------------------------------------------------
        # HEADER CCSDS PARTIDO
        # ----------------------------------------------------

        if self.in_header:

            needed = (
                HEADER_LENGTH
                -
                self.in_header_buffer
            )


            take = min(
                needed,
                MPDU_DATA_SIZE
            )


            self.header_buffer[
                self.in_header_buffer:
                self.in_header_buffer
                +
                take
            ] = bytes(
                data[
                    :take
                ]
            )


            self.in_header_buffer += (
                take
            )


            offset = (
                take
            )


            if (
                self.in_header_buffer
                ==
                HEADER_LENGTH
            ):

                self.in_header = False


                self.read_packet_header(
                    self.header_buffer
                )


        # ----------------------------------------------------
        # CONTINUAR PAQUETE ANTERIOR
        # ----------------------------------------------------

        if (
            self.working_on_packet
            and
            self.remaining_payload_length > 0
        ):

            if fhp < 2047:

                available = max(
                    0,
                    (
                        fhp + 1
                    )
                    -
                    offset
                )

            else:

                available = (
                    MPDU_DATA_SIZE
                    -
                    offset
                )


            to_write = min(
                self.remaining_payload_length,
                available
            )


            if to_write > 0:

                self.push_payload(
                    data[
                        offset:
                        offset
                        +
                        to_write
                    ]
                )


            if (
                self.remaining_payload_length
                ==
                0
            ):

                packet = (
                    self.finish_packet()
                )


                if packet is not None:

                    packets.append(
                        packet
                    )


        # ----------------------------------------------------
        # SIN NUEVO HEADER
        # ----------------------------------------------------

        if fhp >= 2047:

            return packets


        # ----------------------------------------------------
        # NUEVOS PAQUETES
        # ----------------------------------------------------

        pos = fhp


        while pos < MPDU_DATA_SIZE:

            remaining = (
                MPDU_DATA_SIZE
                -
                pos
            )


            # Header dividido entre CADUs
            if remaining < HEADER_LENGTH:

                self.in_header = True

                self.in_header_buffer = (
                    remaining
                )


                self.header_buffer[:] = (
                    b"\x00"
                    *
                    HEADER_LENGTH
                )


                self.header_buffer[
                    :remaining
                ] = bytes(
                    data[
                        pos:
                    ]
                )


                break


            if not self.read_packet_header(
                data[
                    pos:
                    pos
                    +
                    HEADER_LENGTH
                ]
            ):

                break


            total_size = (
                self.total_packet_length
            )


            payload_start = (
                pos
                +
                HEADER_LENGTH
            )


            payload_available = (
                MPDU_DATA_SIZE
                -
                payload_start
            )


            if (
                payload_available
                >=
                self.current_payload_length
            ):

                self.push_payload(
                    data[
                        payload_start:
                        payload_start
                        +
                        self.current_payload_length
                    ]
                )


                packet = (
                    self.finish_packet()
                )


                if packet is not None:

                    packets.append(
                        packet
                    )


                pos += (
                    total_size
                )


            else:

                self.push_payload(
                    data[
                        payload_start:
                    ]
                )


                break


        return packets


# ============================================================
# CANALES
# ============================================================

channel_state = {
    channel: {
        "offset": None,
        "rollover": 0,
        "last_seq": 0,
        "first_segment": None,
        "last_segment": None,
        "segments": {},
        "partial": set()
    }
    for channel in range(
        1,
        7
    )
}


# ============================================================
# INSERTAR SEGMENTO
# ============================================================

def store_segment(
    channel,
    sequence,
    mcu_count,
    image,
    partial
):

    state = channel_state[
        channel
    ]


    # --------------------------------------------------------
    # ROLLOVER CCSDS 14 bits
    # --------------------------------------------------------

    if (
        state[
            "last_seq"
        ] > sequence
        and
        state[
            "last_seq"
        ] > 13926
        and
        sequence < 2458
    ):

        state[
            "rollover"
        ] += 16384


    # --------------------------------------------------------
    # OFFSET DE CICLO DE 43 PAQUETES
    # --------------------------------------------------------

    if state[
        "offset"
    ] is None:

        mcu_seq = (
            sequence
            +
            (
                16384
                if mcu_count > sequence
                else 0
            )
            -
            mcu_count
        )


        state[
            "offset"
        ] = (
            (
                mcu_seq
                +
                state[
                    "rollover"
                ]
            )
            % 43
        )


    # --------------------------------------------------------
    # ID ABSOLUTO DE SEGMENTO
    # --------------------------------------------------------

    segment_id = (
        (
            (
                sequence
                +
                state[
                    "rollover"
                ]
                -
                state[
                    "offset"
                ]
            )
            // 43
        )
        * 14
        +
        mcu_count
    )


    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    first = (
        segment_id
        if state[
            "first_segment"
        ] is None
        else min(
            state[
                "first_segment"
            ],
            segment_id
        )
    )


    last = (
        segment_id
        if state[
            "last_segment"
        ] is None
        else max(
            state[
                "last_segment"
            ],
            segment_id
        )
    )


    if (
        last
        -
        first
        >
        MAX_SEGMENT_SPAN
    ):

        return False


    # --------------------------------------------------------
    # DUPLICADOS:
    # preferir segmento completo a parcial
    # --------------------------------------------------------

    if segment_id in state[
        "segments"
    ]:

        old_partial = (
            segment_id
            in
            state[
                "partial"
            ]
        )


        if (
            old_partial
            and not partial
        ):

            state[
                "segments"
            ][
                segment_id
            ] = image


            state[
                "partial"
            ].discard(
                segment_id
            )


    else:

        state[
            "segments"
        ][
            segment_id
        ] = image


        if partial:

            state[
                "partial"
            ].add(
                segment_id
            )


    state[
        "first_segment"
    ] = (
        first
    )


    state[
        "last_segment"
    ] = (
        last
    )


    state[
        "last_seq"
    ] = (
        sequence
    )


    return True


# ============================================================
# RELLENO VERTICAL DE HUECOS
#
# Solo se usa para copias visuales.
# La versión RAW permanece intacta.
# ============================================================

def fill_vertical_gaps(
    image,
    valid_mask,
    max_gap
):

    output = (
        image
        .astype(
            np.float32
        )
        .copy()
    )


    height, width = (
        output.shape
    )


    for x in range(
        width
    ):

        good = (
            valid_mask[
                :,
                x
            ]
        )


        y = 0


        while y < height:

            if good[y]:

                y += 1

                continue


            start = y


            while (
                y < height
                and not good[y]
            ):

                y += 1


            end = y

            gap = (
                end
                -
                start
            )


            top = (
                start
                - 1
            )

            bottom = (
                end
            )


            if (
                gap <= max_gap
                and top >= 0
                and bottom < height
                and good[top]
                and good[bottom]
            ):

                top_value = (
                    output[
                        top,
                        x
                    ]
                )

                bottom_value = (
                    output[
                        bottom,
                        x
                    ]
                )


                for yy in range(
                    start,
                    end
                ):

                    fraction = (
                        (
                            yy
                            -
                            top
                        )
                        /
                        (
                            bottom
                            -
                            top
                        )
                    )


                    output[
                        yy,
                        x
                    ] = (
                        top_value
                        *
                        (
                            1.0
                            -
                            fraction
                        )
                        +
                        bottom_value
                        *
                        fraction
                    )


    return np.clip(
        np.rint(
            output
        ),
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# CONTRASTE
# ============================================================

def percentile_stretch(
    image,
    low=1.0,
    high=99.0
):

    x = (
        image
        .astype(
            np.float32
        )
    )


    valid = (
        x > 0
    )


    if not np.any(
        valid
    ):

        return np.zeros_like(
            image
        )


    p_low = np.percentile(
        x[
            valid
        ],
        low
    )


    p_high = np.percentile(
        x[
            valid
        ],
        high
    )


    if p_high <= p_low:

        return image.copy()


    x = (
        x
        -
        p_low
    ) / (
        p_high
        -
        p_low
    )


    x = np.clip(
        x,
        0.0,
        1.0
    )


    return np.rint(
        x
        *
        255.0
    ).astype(
        np.uint8
    )


def apply_gamma(
    image,
    gamma
):

    x = (
        image
        .astype(
            np.float32
        )
        /
        255.0
    )


    x = np.power(
        np.clip(
            x,
            0.0,
            1.0
        ),
        gamma
    )


    return np.rint(
        x
        *
        255.0
    ).astype(
        np.uint8
    )


# ============================================================
# CORRELACIÓN PARA RGB
# ============================================================

def normalized_correlation(
    a,
    b
):

    a = (
        a
        .astype(
            np.float64
        )
    )

    b = (
        b
        .astype(
            np.float64
        )
    )


    mask = (
        (a > 5)
        &
        (b > 5)
    )


    if np.count_nonzero(
        mask
    ) < 1000:

        return -1.0


    x = a[
        mask
    ]


    y = b[
        mask
    ]


    x -= (
        np.mean(
            x
        )
    )


    y -= (
        np.mean(
            y
        )
    )


    sx = np.std(
        x
    )

    sy = np.std(
        y
    )


    if (
        sx < 1e-12
        or sy < 1e-12
    ):

        return -1.0


    return float(
        np.mean(
            (
                x
                /
                sx
            )
            *
            (
                y
                /
                sy
            )
        )
    )


def find_vertical_shift(
    reference,
    image,
    max_shift=32,
    step=8
):

    best_score = -999.0

    best_move = 0


    for move in range(
        -max_shift,
        max_shift + 1,
        step
    ):

        if move > 0:

            ref_part = (
                reference[
                    move:
                ]
            )


            img_part = (
                image[
                    :-move
                ]
            )


        elif move < 0:

            amount = (
                -move
            )


            ref_part = (
                reference[
                    :-amount
                ]
            )


            img_part = (
                image[
                    amount:
                ]
            )


        else:

            ref_part = (
                reference
            )

            img_part = (
                image
            )


        height = min(
            ref_part.shape[0],
            img_part.shape[0]
        )


        if height <= 0:

            continue


        score = normalized_correlation(
            ref_part[
                :height
            ],
            img_part[
                :height
            ]
        )


        if score > best_score:

            best_score = (
                score
            )

            best_move = (
                move
            )


    return (
        best_move,
        best_score
    )


def shift_image_nan(
    image,
    move
):

    height, width = (
        image.shape
    )


    result = np.full(
        (
            height,
            width
        ),
        np.nan,
        dtype=np.float32
    )


    if move > 0:

        result[
            move:
        ] = image[
            :-move
        ]


    elif move < 0:

        amount = (
            -move
        )


        result[
            :-amount
        ] = image[
            amount:
        ]


    else:

        result[:] = (
            image
        )


    return result


# ============================================================
# CARGAR CADUs
# ============================================================

if not CADU_FILE.exists():

    raise FileNotFoundError(
        CADU_FILE
    )


raw = np.fromfile(
    CADU_FILE,
    dtype=np.uint8
)


if (
    len(raw)
    %
    CADU_SIZE
    != 0
):

    raise RuntimeError(
        "El archivo CADU no es múltiplo de 1024 bytes."
    )


total_cadus = (
    len(raw)
    //
    CADU_SIZE
)


print()
print(
    "================================================"
)

print(
    "       METEOR LRPT FULLPASS -> MSU-MR"
)

print(
    "================================================"
)

print()

print(
    f"CADUs entrada : "
    f"{total_cadus}"
)

print()


# ============================================================
# DEMULTIPLEX
# ============================================================

demuxer = AOSDemuxer()


previous_counter = None


accepted_cadus = 0

non_vcid5 = 0

duplicate_cadus = 0

counter_gaps = 0

missing_cadus = 0

outlier_cadus = 0


packet_count = 0

apid_counts = Counter()


segment_headers = 0

segments_decoded = 0

segments_partial = 0

segment_failures = 0


packet_rows = []


# ============================================================
# RECORRER CADUs
# ============================================================

for cadu_index in range(
    total_cadus
):

    cadu = raw[
        cadu_index
        *
        CADU_SIZE:
        (
            cadu_index
            +
            1
        )
        *
        CADU_SIZE
    ]


    vcdu = parse_vcdu(
        cadu
    )


    # --------------------------------------------------------
    # Solo MSU-MR VCID 5
    # --------------------------------------------------------

    if vcdu.vcid != 5:

        non_vcid5 += 1

        demuxer.reset()

        continue


    # --------------------------------------------------------
    # CONTROL DE CONTINUIDAD
    # --------------------------------------------------------

    if previous_counter is not None:

        delta = (
            (
                vcdu.counter
                -
                previous_counter
            )
            &
            0xFFFFFF
        )


        # Duplicado
        if delta == 0:

            duplicate_cadus += 1

            continue


        # Salto razonable
        if (
            1
            <
            delta
            <=
            MAX_REASONABLE_VCDU_GAP
        ):

            counter_gaps += 1

            missing_cadus += (
                delta
                -
                1
            )


            # NO unir paquetes CCSDS a través
            # de una CADU perdida.
            demuxer.reset()


        # Outlier absurdo
        elif delta > MAX_REASONABLE_VCDU_GAP:

            outlier_cadus += 1

            demuxer.reset()

            # No actualizamos previous_counter:
            # así la siguiente CADU normal se compara
            # contra la última trama buena.
            continue


    previous_counter = (
        vcdu.counter
    )


    accepted_cadus += 1


    # --------------------------------------------------------
    # AOS -> CCSDS
    # --------------------------------------------------------

    packets = demuxer.process(
        cadu
    )


    for (
        header,
        packet_bytes
    ) in packets:

        packet_count += 1


        apid_counts[
            header.apid
        ] += 1


        payload = np.frombuffer(
            packet_bytes[
                HEADER_LENGTH:
            ],
            dtype=np.uint8
        )


        channel = None


        if (
            64
            <= header.apid
            <= 69
        ):

            channel = (
                header.apid
                -
                63
            )


        # ----------------------------------------------------
        # REGISTRO
        # ----------------------------------------------------

        row = {
            "packet_index":
                packet_count - 1,

            "cadu_index":
                cadu_index,

            "vcdu_counter":
                vcdu.counter,

            "apid":
                header.apid,

            "channel":
                (
                    ""
                    if channel is None
                    else channel
                ),

            "sequence":
                header.sequence_count,

            "payload_bytes":
                len(payload),

            "mcun":
                "",

            "mcu_count":
                "",

            "qf":
                "",

            "decoded_blocks":
                "",

            "partial":
                ""
        }


        # ----------------------------------------------------
        # NO ES CANAL DE IMAGEN
        # ----------------------------------------------------

        if channel is None:

            packet_rows.append(
                row
            )

            continue


        # ----------------------------------------------------
        # HEADER MSU-MR
        # ----------------------------------------------------

        if len(payload) <= 14:

            segment_failures += 1

            packet_rows.append(
                row
            )

            continue


        mcun = int(
            payload[8]
        )


        qt = int(
            payload[9]
        )


        dc = (
            int(
                payload[10]
            )
            >> 4
        ) & 0x0F


        ac = (
            int(
                payload[10]
            )
            & 0x0F
        )


        qfm = (
            (
                int(
                    payload[11]
                )
                << 8
            )
            |
            int(
                payload[12]
            )
        )


        qf = int(
            payload[13]
        )


        valid_header = (
            qt == 0x00
            and dc == 0x00
            and ac == 0x00
            and qfm == 0xFFF0
        )


        if not valid_header:

            segment_failures += 1

            packet_rows.append(
                row
            )

            continue


        mcu_count = (
            mcun
            //
            14
        )


        if not (
            0
            <= mcu_count
            <= 13
        ):

            segment_failures += 1

            packet_rows.append(
                row
            )

            continue


        segment_headers += 1


        # ----------------------------------------------------
        # HUFFMAN + IDCT
        # ----------------------------------------------------

        (
            segment_image,
            decoded_blocks,
            partial
        ) = decode_segment(
            payload[
                14:
            ],
            qf
        )


        row[
            "mcun"
        ] = mcun


        row[
            "mcu_count"
        ] = mcu_count


        row[
            "qf"
        ] = qf


        row[
            "decoded_blocks"
        ] = decoded_blocks


        row[
            "partial"
        ] = int(
            partial
        )


        packet_rows.append(
            row
        )


        if decoded_blocks == 0:

            segment_failures += 1

            continue


        stored = store_segment(
            channel=channel,
            sequence=header.sequence_count,
            mcu_count=mcu_count,
            image=segment_image,
            partial=partial
        )


        if not stored:

            segment_failures += 1

            continue


        segments_decoded += 1


        if partial:

            segments_partial += 1


# ============================================================
# CSV
# ============================================================

packet_fields = [
    "packet_index",
    "cadu_index",
    "vcdu_counter",
    "apid",
    "channel",
    "sequence",
    "payload_bytes",
    "mcun",
    "mcu_count",
    "qf",
    "decoded_blocks",
    "partial"
]


with open(
    PACKET_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=packet_fields
    )

    writer.writeheader()

    writer.writerows(
        packet_rows
    )


# ============================================================
# INFORMACIÓN DE CANALES
# ============================================================

print()
print(
    "================================================"
)

print(
    "                 DEMUX RESULT"
)

print(
    "================================================"
)

print()

print(
    f"CADUs aceptadas       : "
    f"{accepted_cadus}"
)

print(
    f"CADUs no VCID 5       : "
    f"{non_vcid5}"
)

print(
    f"CADUs duplicadas      : "
    f"{duplicate_cadus}"
)

print(
    f"Outliers descartados  : "
    f"{outlier_cadus}"
)

print(
    f"Huecos VCDU           : "
    f"{counter_gaps}"
)

print(
    f"CADUs faltantes       : "
    f"{missing_cadus}"
)

print()

print(
    f"Paquetes CCSDS        : "
    f"{packet_count}"
)

print()


for apid in sorted(
    apid_counts
):

    label = ""


    if (
        64
        <= apid
        <= 69
    ):

        label = (
            f"canal {apid - 63}"
        )


    elif apid == 70:

        label = (
            "telemetría"
        )


    print(
        f"APID {apid:4d}: "
        f"{apid_counts[apid]:5d} "
        f"{label}"
    )


# ============================================================
# RECONSTRUIR CANALES EN COORDENADAS DE SEGMENT_ID
#
# Todos usan un sistema Y común.
# Esto evita el problema que teníamos en el experimento
# corto donde cada canal comenzaba en una línea distinta.
# ============================================================

all_segment_ids = []


for channel in range(
    1,
    7
):

    all_segment_ids.extend(
        channel_state[
            channel
        ][
            "segments"
        ].keys()
    )


if not all_segment_ids:

    raise RuntimeError(
        "No se recuperaron segmentos MSU-MR."
    )


global_first_id = min(
    all_segment_ids
)

global_last_id = max(
    all_segment_ids
)


global_first_line = (
    global_first_id
    -
    (
        global_first_id
        %
        14
    )
)


global_last_line = (
    global_last_id
    -
    (
        global_last_id
        %
        14
    )
)


line_starts = list(
    range(
        global_first_line,
        global_last_line
        +
        1,
        14
    )
)


full_height = (
    len(
        line_starts
    )
    *
    SEGMENT_HEIGHT
)


print()
print(
    "================================================"
)

print(
    "          RECONSTRUYENDO FULL PASS"
)

print(
    "================================================"
)

print()

print(
    f"Segment ID inicial : "
    f"{global_first_id}"
)

print(
    f"Segment ID final   : "
    f"{global_last_id}"
)

print(
    f"Líneas MSU-MR      : "
    f"{len(line_starts)}"
)

print(
    f"Canvas global      : "
    f"{FULL_WIDTH} x {full_height}"
)

print()


channel_images_raw = {}

channel_images_filled = {}

channel_masks = {}


# ============================================================
# CREAR IMAGEN GLOBAL POR CANAL
# ============================================================

for channel in range(
    1,
    7
):

    segments = channel_state[
        channel
    ][
        "segments"
    ]


    if not segments:

        print(
            f"Canal {channel}: "
            f"sin datos."
        )

        continue


    image = np.zeros(
        (
            full_height,
            FULL_WIDTH
        ),
        dtype=np.uint8
    )


    mask = np.zeros(
        (
            full_height,
            FULL_WIDTH
        ),
        dtype=bool
    )


    present = 0

    missing = 0


    for line_index, line_start in enumerate(
        line_starts
    ):

        y0 = (
            line_index
            *
            SEGMENT_HEIGHT
        )

        y1 = (
            y0
            +
            SEGMENT_HEIGHT
        )


        for segment_index in range(
            14
        ):

            segment_id = (
                line_start
                +
                segment_index
            )


            x0 = (
                segment_index
                *
                SEGMENT_WIDTH
            )

            x1 = (
                x0
                +
                SEGMENT_WIDTH
            )


            if segment_id in segments:

                image[
                    y0:y1,
                    x0:x1
                ] = segments[
                    segment_id
                ]


                mask[
                    y0:y1,
                    x0:x1
                ] = True


                present += 1


            else:

                missing += 1


    channel_images_raw[
        channel
    ] = image


    channel_masks[
        channel
    ] = mask


    # --------------------------------------------------------
    # GUARDAR RAW
    # --------------------------------------------------------

    raw_file = (
        OUTPUT_DIR
        /
        f"meteor_msumr_canal_{channel}_full_raw.png"
    )


    Image.fromarray(
        image,
        mode="L"
    ).save(
        raw_file
    )


    # --------------------------------------------------------
    # RELLENO VISUAL
    # --------------------------------------------------------

    if FILL_MISSING:

        filled = fill_vertical_gaps(
            image,
            mask,
            MAX_FILL_ROWS
        )

    else:

        filled = (
            image.copy()
        )


    channel_images_filled[
        channel
    ] = filled


    filled_file = (
        OUTPUT_DIR
        /
        f"meteor_msumr_canal_{channel}_full_filled.png"
    )


    Image.fromarray(
        filled,
        mode="L"
    ).save(
        filled_file
    )


    # --------------------------------------------------------
    # ENHANCED
    # --------------------------------------------------------

    enhanced = percentile_stretch(
        filled,
        LOW_PERCENTILE,
        HIGH_PERCENTILE
    )


    enhanced = apply_gamma(
        enhanced,
        GAMMA
    )


    enhanced_file = (
        OUTPUT_DIR
        /
        f"meteor_msumr_canal_{channel}_full_enhanced.png"
    )


    Image.fromarray(
        enhanced,
        mode="L"
    ).save(
        enhanced_file
    )


    print(
        f"Canal {channel}:"
    )

    print(
        f"  segmentos únicos : "
        f"{len(segments)}"
    )

    print(
        f"  segmentos canvas : "
        f"{present}"
    )

    print(
        f"  huecos canvas     : "
        f"{missing}"
    )

    print(
        f"  offset protocolo  : "
        f"{channel_state[channel]['offset']}"
    )

    print(
        f"  tamaño            : "
        f"{FULL_WIDTH} x {full_height}"
    )

    print()


# ============================================================
# RGB 1 / 2 / 3
# ============================================================

required_rgb = (
    1,
    2,
    3
)


if all(
    channel in channel_images_filled
    for channel in required_rgb
):

    c1 = channel_images_filled[1]

    c2 = channel_images_filled[2]

    c3 = channel_images_filled[3]


    # --------------------------------------------------------
    # Afinamiento adicional por correlación.
    #
    # El segment_id ya nos da alineamiento de protocolo.
    # Esto solo permite corregir ± unas pocas líneas.
    # --------------------------------------------------------

    shift1, corr1 = find_vertical_shift(
        c2,
        c1,
        max_shift=MAX_RGB_VERTICAL_SHIFT,
        step=ALIGN_STEP
    )


    shift2 = 0

    corr2 = 1.0


    shift3, corr3 = find_vertical_shift(
        c2,
        c3,
        max_shift=MAX_RGB_VERTICAL_SHIFT,
        step=ALIGN_STEP
    )


    print()
    print(
        "================================================"
    )

    print(
        "                RGB ALIGNMENT"
    )

    print(
        "================================================"
    )

    print()

    print(
        f"Canal 1 -> "
        f"{shift1:+d} px "
        f"| corr={corr1:.4f}"
    )

    print(
        f"Canal 2 -> "
        f"{shift2:+d} px "
        f"| corr={corr2:.4f}"
    )

    print(
        f"Canal 3 -> "
        f"{shift3:+d} px "
        f"| corr={corr3:.4f}"
    )


    aligned = {
        1:
            shift_image_nan(
                c1,
                shift1
            ),

        2:
            shift_image_nan(
                c2,
                shift2
            ),

        3:
            shift_image_nan(
                c3,
                shift3
            )
    }


    valid_rows = np.ones(
        full_height,
        dtype=bool
    )


    for channel in (
        1,
        2,
        3
    ):

        valid_rows &= np.all(
            ~np.isnan(
                aligned[
                    channel
                ]
            ),
            axis=1
        )


    row_indices = np.where(
        valid_rows
    )[0]


    if len(
        row_indices
    ) > 0:

        y0 = int(
            row_indices[0]
        )


        y1 = int(
            row_indices[-1]
            +
            1
        )


        for channel in (
            1,
            2,
            3
        ):

            aligned[
                channel
            ] = aligned[
                channel
            ][
                y0:y1
            ].astype(
                np.uint8
            )


        c1a = aligned[1]

        c2a = aligned[2]

        c3a = aligned[3]


        print()

        print(
            f"RGB crop: "
            f"y={y0}:{y1}"
        )

        print(
            f"RGB size: "
            f"{FULL_WIDTH} x "
            f"{y1 - y0}"
        )


        # ====================================================
        # GUARDAR ALINEADOS
        # ====================================================

        for channel, image in (
            (1, c1a),
            (2, c2a),
            (3, c3a)
        ):

            path = (
                OUTPUT_DIR
                /
                f"meteor_msumr_canal_{channel}_rgb_aligned.png"
            )


            Image.fromarray(
                image,
                mode="L"
            ).save(
                path
            )


        # ====================================================
        # RAW RGB
        # ====================================================

        rgb123_raw = np.dstack(
            [
                c1a,
                c2a,
                c3a
            ]
        )


        rgb321_raw = np.dstack(
            [
                c3a,
                c2a,
                c1a
            ]
        )


        # ====================================================
        # STRETCH
        # ====================================================

        c1e = apply_gamma(
            percentile_stretch(
                c1a,
                LOW_PERCENTILE,
                HIGH_PERCENTILE
            ),
            GAMMA
        )


        c2e = apply_gamma(
            percentile_stretch(
                c2a,
                LOW_PERCENTILE,
                HIGH_PERCENTILE
            ),
            GAMMA
        )


        c3e = apply_gamma(
            percentile_stretch(
                c3a,
                LOW_PERCENTILE,
                HIGH_PERCENTILE
            ),
            GAMMA
        )


        rgb123_enhanced = np.dstack(
            [
                c1e,
                c2e,
                c3e
            ]
        )


        rgb321_enhanced = np.dstack(
            [
                c3e,
                c2e,
                c1e
            ]
        )


        # ====================================================
        # GUARDAR RGB
        # ====================================================

        rgb_outputs = {
            "meteor_fullpass_rgb_123_raw.png":
                rgb123_raw,

            "meteor_fullpass_rgb_123_enhanced.png":
                rgb123_enhanced,

            "meteor_fullpass_rgb_321_raw.png":
                rgb321_raw,

            "meteor_fullpass_rgb_321_enhanced.png":
                rgb321_enhanced
        }


        for filename, array in rgb_outputs.items():

            Image.fromarray(
                array,
                mode="RGB"
            ).save(
                OUTPUT_DIR
                /
                filename
            )


        # ====================================================
        # PREVIEW 50 %
        # ====================================================

        for filename in (
            "meteor_fullpass_rgb_123_enhanced.png",
            "meteor_fullpass_rgb_321_enhanced.png"
        ):

            path = (
                OUTPUT_DIR
                /
                filename
            )


            image = Image.open(
                path
            )


            preview = image.resize(
                (
                    image.width
                    //
                    2,

                    max(
                        1,
                        image.height
                        //
                        2
                    )
                ),
                resample=Image.Resampling.LANCZOS
            )


            preview.save(
                OUTPUT_DIR
                /
                filename.replace(
                    ".png",
                    "_preview_50pct.png"
                )
            )


# ============================================================
# RESUMEN
# ============================================================

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
    f"CADUs archivo          : "
    f"{total_cadus}"
)

print(
    f"CADUs aceptadas VCID5  : "
    f"{accepted_cadus}"
)

print(
    f"CADUs no VCID5         : "
    f"{non_vcid5}"
)

print(
    f"Outliers descartados   : "
    f"{outlier_cadus}"
)

print(
    f"Huecos VCDU            : "
    f"{counter_gaps}"
)

print(
    f"CADUs faltantes        : "
    f"{missing_cadus}"
)

print()

print(
    f"Paquetes CCSDS         : "
    f"{packet_count}"
)

print(
    f"Headers MSU-MR válidos : "
    f"{segment_headers}"
)

print(
    f"Segmentos decodificados: "
    f"{segments_decoded}"
)

print(
    f"Segmentos parciales    : "
    f"{segments_partial}"
)

print(
    f"Fallos segmento        : "
    f"{segment_failures}"
)

print()

for channel in range(
    1,
    7
):

    state = channel_state[
        channel
    ]


    print(
        f"Canal {channel}: "
        f"{len(state['segments'])} segmentos | "
        f"offset={state['offset']} | "
        f"parciales={len(state['partial'])}"
    )


print()

print(
    f"Canvas fullpass        : "
    f"{FULL_WIDTH} x "
    f"{full_height}"
)

print()

print(
    "Archivos:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "CSV paquetes:"
)

print(
    PACKET_CSV
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