from pathlib import Path
from collections import defaultdict
import math

import numpy as np
from scipy.fft import idctn
from PIL import Image


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PACKET_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "msumr_packets"
    / "packets"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "images"
    / "msumr_10s"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

CCSDS_HEADER_SIZE = 6

# Cada segmento MSU-MR:
#
# 14 bloques horizontales
# 8x8 píxeles cada bloque
#
# ancho = 14 * 8 = 112
# alto  = 8
SEGMENT_WIDTH = 112
SEGMENT_HEIGHT = 8

# Una línea completa:
#
# 14 segmentos * 112 px = 1568 px
LINE_WIDTH = 1568


# ============================================================
# TABLA DE CUANTIZACIÓN
#
# Misma tabla usada por SatDump / JPEG luminancia.
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
#
# Mismo mapa usado por SatDump.
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
#
# SatDump tiene estas tablas escritas código por código.
# Aquí usamos la representación canónica compacta equivalente.
# ============================================================

DC_BITS = [
    0,  # longitud 1
    1,  # longitud 2
    5,  # longitud 3
    1,  # longitud 4
    1,  # longitud 5
    1,  # longitud 6
    1,  # longitud 7
    1,  # longitud 8
    1,  # longitud 9
    0,
    0,
    0,
    0,
    0,
    0,
    0
]

DC_VALUES = list(
    range(12)
)


AC_BITS = [
    0,
    2,
    1,
    3,
    3,
    2,
    4,
    3,
    5,
    5,
    4,
    4,
    0,
    0,
    1,
    125
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
# CONSTRUIR TABLAS HUFFMAN CANÓNICAS
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

            if symbol_index >= len(
                symbols
            ):

                raise RuntimeError(
                    "Tabla Huffman incorrecta."
                )

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
            "No se utilizaron todos los símbolos Huffman."
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
            len(self.data)
            * 8
        )


    def remaining(
        self
    ):

        return (
            self.total_bits
            - self.position
        )


    def read_bit(
        self
    ):

        if (
            self.position
            >= self.total_bits
        ):

            raise EOFError(
                "Fin del flujo Huffman."
            )


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


# ============================================================
# EXTENSIÓN DE SIGNO JPEG
#
# Equivalente al getValue() de SatDump.
# ============================================================

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
        1 << (
            size - 1
        )
    )


    if value < limit:

        value -= (
            (
                1 << size
            )
            - 1
        )


    return value


# ============================================================
# TABLA DE CUANTIZACIÓN SEGÚN QF
#
# Replica GetQuantizationTable() de SatDump.
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


    return table.astype(
        np.float64
    )


# ============================================================
# DECODIFICAR BLOQUE DCT 8x8
# ============================================================

def decode_block(
    reader,
    last_dc,
    quant_table
):

    # --------------------------------------------------------
    # Coeficientes en orden zig-zag
    # --------------------------------------------------------

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
        + dc_delta
    )

    coefficients[
        0
    ] = dc_value


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


        # ZRL:
        # 16 ceros
        if symbol == 0xF0:

            index += 16

            if index > 64:

                raise ValueError(
                    "ZRL fuera del bloque."
                )

            continue


        run = (
            symbol >> 4
        )

        size = (
            symbol & 0x0F
        )


        index += run


        if index >= 64:

            raise ValueError(
                "Run-length AC fuera del bloque."
            )


        value = receive_extend(
            reader,
            size
        )


        coefficients[
            index
        ] = value

        index += 1


    # ========================================================
    # DE-ZIGZAG + DEQUANTIZACIÓN
    #
    # SatDump:
    #
    # idctBlock[x] =
    #     block[Zigzag[x]] * qTable[x]
    # ========================================================

    natural = np.zeros(
        64,
        dtype=np.float64
    )


    for x in range(
        64
    ):

        natural[
            x
        ] = (
            coefficients[
                ZIGZAG[x]
            ]
            * quant_table[
                x
            ]
        )


    natural = natural.reshape(
        8,
        8
    )


    # ========================================================
    # IDCT
    #
    # SciPy con norm="ortho" implementa la misma IDCT
    # JPEG matemática. SatDump utiliza una versión entera
    # optimizada del mismo proceso.
    # ========================================================

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
# DECODIFICAR UN SEGMENTO MSU-MR
#
# Resultado:
#
# 8 x 112 píxeles
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

            # SatDump también permite segmentos parciales.
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
# CCSDS HEADER
# ============================================================

def parse_ccsds_packet(
    packet
):

    if len(packet) < 6:

        return None


    b0 = int(
        packet[0]
    )

    b1 = int(
        packet[1]
    )

    b2 = int(
        packet[2]
    )

    b3 = int(
        packet[3]
    )

    b4 = int(
        packet[4]
    )

    b5 = int(
        packet[5]
    )


    apid = (
        (
            b0
            & 0x07
        )
        << 8
    ) | b1


    sequence = (
        (
            b2
            & 0x3F
        )
        << 8
    ) | b3


    payload_length = (
        (
            b4 << 8
        )
        |
        b5
    ) + 1


    payload = packet[
        6:
        6 + payload_length
    ]


    return (
        apid,
        sequence,
        payload
    )


# ============================================================
# ESTADO DE CADA CANAL
#
# Copia el concepto que usa MSUMRReader:
#
# offset + rollover + sequence + mcu_count
# ============================================================

channel_state = {
    channel: {
        "offset": None,
        "rollover": 0,
        "last_seq": 0,
        "segments": {},
        "partial": set()
    }
    for channel in range(
        1,
        7
    )
}


# ============================================================
# CONTADORES
# ============================================================

packets_seen = 0

segments_seen = 0

segments_decoded = 0

partial_segments = 0

failed_headers = 0

first_preview_saved = False


# ============================================================
# PROCESAR PAQUETES
# ============================================================

print()
print(
    "================================================"
)

print(
    "     DECODIFICANDO SEGMENTOS MSU-MR"
)

print(
    "================================================"
)

print()


packet_files = sorted(
    PACKET_DIR.glob(
        "packet_*.bin"
    )
)


for packet_file in packet_files:

    packet = np.fromfile(
        packet_file,
        dtype=np.uint8
    )


    parsed = parse_ccsds_packet(
        packet
    )


    if parsed is None:

        continue


    (
        apid,
        sequence,
        payload
    ) = parsed


    packets_seen += 1


    # --------------------------------------------------------
    # Solo canales MSU-MR
    # --------------------------------------------------------

    if not (
        64
        <= apid
        <= 69
    ):

        continue


    channel = (
        apid
        - 63
    )


    if len(payload) <= 14:

        failed_headers += 1

        continue


    # ========================================================
    # HEADER DEL SEGMENTO
    # ========================================================

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

        failed_headers += 1

        continue


    mcu_count = (
        mcun
        // 14
    )


    if not (
        0
        <= mcu_count
        <= 13
    ):

        failed_headers += 1

        continue


    segments_seen += 1


    # ========================================================
    # DESCOMPRESIÓN
    # ========================================================

    compressed_data = (
        payload[
            14:
        ]
    )


    (
        segment_image,
        decoded_blocks,
        partial
    ) = decode_segment(
        compressed_data,
        qf
    )


    if decoded_blocks == 0:

        print(
            f"ERROR | "
            f"CH={channel} "
            f"SEQ={sequence} "
            f"MCU={mcu_count} "
            f"QF={qf}"
        )

        continue


    segments_decoded += 1


    if partial:

        partial_segments += 1


    # ========================================================
    # CALCULAR ID DEL SEGMENTO
    #
    # Misma lógica utilizada por SatDump.
    # ========================================================

    state = channel_state[
        channel
    ]


    # --------------------------------------------------------
    # Rollover contador CCSDS de 14 bits
    # --------------------------------------------------------

    if (
        state[
            "last_seq"
        ] > sequence
        and state[
            "last_seq"
        ] > 13926
        and sequence < 2458
    ):

        state[
            "rollover"
        ] += 16384


    # --------------------------------------------------------
    # Calcular offset de la rueda de 43 paquetes
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


    state[
        "segments"
    ][
        segment_id
    ] = (
        segment_image
    )


    if partial:

        state[
            "partial"
        ].add(
            segment_id
        )


    state[
        "last_seq"
    ] = (
        sequence
    )


    # ========================================================
    # GUARDAR UN SEGMENTO DE EJEMPLO
    # ========================================================

    if (
        not first_preview_saved
        and channel == 2
        and mcu_count == 0
    ):

        preview_file = (
            OUTPUT_DIR
            / "segmento_preview_canal_2.png"
        )


        Image.fromarray(
            segment_image,
            mode="L"
        ).save(
            preview_file
        )


        first_preview_saved = True


# ============================================================
# CONSTRUIR IMAGEN POR CANAL
# ============================================================

print()
print(
    "================================================"
)

print(
    "          RECONSTRUYENDO CANALES"
)

print(
    "================================================"
)

print()


generated_images = []


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
            f"Canal {channel}: sin datos."
        )

        continue


    ids = sorted(
        segments.keys()
    )


    first_id = (
        ids[0]
    )

    last_id = (
        ids[-1]
    )


    # Primer segmento de línea
    first_line = (
        first_id
        -
        (
            first_id
            % 14
        )
    )


    last_line = (
        last_id
        -
        (
            last_id
            % 14
        )
    )


    line_starts = list(
        range(
            first_line,
            last_line + 1,
            14
        )
    )


    height = (
        len(
            line_starts
        )
        * SEGMENT_HEIGHT
    )


    image = np.zeros(
        (
            height,
            LINE_WIDTH
        ),
        dtype=np.uint8
    )


    missing_segments = 0

    present_segments = 0


    for line_index, line_start in enumerate(
        line_starts
    ):

        y0 = (
            line_index
            * SEGMENT_HEIGHT
        )

        y1 = (
            y0
            + SEGMENT_HEIGHT
        )


        for segment_index in range(
            14
        ):

            segment_id = (
                line_start
                + segment_index
            )


            x0 = (
                segment_index
                * SEGMENT_WIDTH
            )

            x1 = (
                x0
                + SEGMENT_WIDTH
            )


            if segment_id in segments:

                image[
                    y0:y1,
                    x0:x1
                ] = segments[
                    segment_id
                ]

                present_segments += 1


            else:

                missing_segments += 1


    output_file = (
        OUTPUT_DIR
        /
        f"meteor_msumr_canal_{channel}.png"
    )


    Image.fromarray(
        image,
        mode="L"
    ).save(
        output_file
    )


    generated_images.append(
        (
            channel,
            image,
            output_file
        )
    )


    print(
        f"Canal {channel}:"
    )

    print(
        f"  dimensión          : "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print(
        f"  segmentos presentes: "
        f"{present_segments}"
    )

    print(
        f"  segmentos faltantes: "
        f"{missing_segments}"
    )

    print(
        f"  PNG                : "
        f"{output_file}"
    )

    print()


# ============================================================
# PREVIEW AMPLIADO
#
# Las imágenes de 10 s serán muy bajas, aprox. 64 px.
# Creamos una versión ampliada SIN interpolación para verla.
# ============================================================

for (
    channel,
    image,
    original_file
) in generated_images:

    if image.size == 0:

        continue


    scale = 4


    preview = Image.fromarray(
        image,
        mode="L"
    ).resize(
        (
            image.shape[1],
            image.shape[0]
            * scale
        ),
        resample=Image.Resampling.NEAREST
    )


    preview_file = (
        OUTPUT_DIR
        /
        f"meteor_msumr_canal_{channel}_preview_4x.png"
    )


    preview.save(
        preview_file
    )


# ============================================================
# RESUMEN
# ============================================================

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
    f"Paquetes examinados       : "
    f"{packets_seen}"
)

print(
    f"Segmentos encontrados     : "
    f"{segments_seen}"
)

print(
    f"Segmentos decodificados   : "
    f"{segments_decoded}"
)

print(
    f"Segmentos parciales       : "
    f"{partial_segments}"
)

print(
    f"Headers inválidos         : "
    f"{failed_headers}"
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
        f"offset={state['offset']}"
    )


print()
print(
    "Imágenes guardadas en:"
)

print(
    OUTPUT_DIR
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