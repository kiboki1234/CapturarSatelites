from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
import csv
import numpy as np


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CADU_FILE = (
    ROOT
    / "outputs"
    / "decoded"
    / "full_decode"
    / "meteor_lrpt_valid_10s.cadu"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "msumr_packets"
)

PACKET_DIR = (
    OUTPUT_DIR
    / "packets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PACKET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_FILE = (
    OUTPUT_DIR
    / "msumr_packets.csv"
)


# ============================================================
# CONSTANTES
# ============================================================

CADU_SIZE = 1024

# Igual que SatDump:
#
# Demuxer(882, true)
MPDU_DATA_SIZE = 882

INSERT_ZONE_SIZE = 2

HEADER_LENGTH = 6


# ============================================================
# ESTRUCTURAS
# ============================================================

@dataclass
class VCDUHeader:
    version: int
    spacecraft_id: int
    vcid: int
    counter: int
    replay_flag: int


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
# VCDU
# ============================================================

def parse_vcdu(cadu):

    version = int(
        cadu[4] >> 6
    )

    spacecraft_id = int(
        (
            (
                int(cadu[4])
                & 0x3F
            )
            << 2
        )
        |
        (
            int(cadu[5])
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

    replay_flag = int(
        cadu[9] >> 7
    )

    return VCDUHeader(
        version=version,
        spacecraft_id=spacecraft_id,
        vcid=vcid,
        counter=counter,
        replay_flag=replay_flag
    )


# ============================================================
# M-PDU
#
# Insert zone de 2 bytes:
#
# CADU:
#
# 0..3    ASM
# 4..9    VCDU header
# 10..11  insert zone
# 12..13  MPDU header / FHP
# 14..    MPDU data
# ============================================================

def parse_mpdu(cadu):

    fhp = (
        (
            int(cadu[12])
            & 0x07
        )
        << 8
    ) | int(
        cadu[13]
    )

    data = (
        cadu[
            14:
            14 + MPDU_DATA_SIZE
        ]
        .copy()
    )

    return (
        fhp,
        data
    )


# ============================================================
# CCSDS PRIMARY HEADER
# ============================================================

def parse_ccsds_header(raw):

    if len(raw) < 6:
        return None

    b0 = int(raw[0])
    b1 = int(raw[1])
    b2 = int(raw[2])
    b3 = int(raw[3])
    b4 = int(raw[4])
    b5 = int(raw[5])

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
        raw=bytes(raw[:6])
    )


# ============================================================
# DEMUXER AOS
#
# Traducción de la lógica del Demuxer de SatDump.
#
# Mantiene paquetes que cruzan de una CADU a la siguiente.
# ============================================================

class AOSDemuxer:

    def __init__(self):

        self.reset()

        self.total_packets = 0


    def reset(self):

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


    # --------------------------------------------------------
    # Comenzar nuevo paquete
    # --------------------------------------------------------

    def read_packet_header(
        self,
        raw_header
    ):

        header = parse_ccsds_header(
            raw_header
        )

        if header is None:

            self.abort_packet()

            return False


        # Validaciones básicas
        if header.version != 0:

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

        self.total_packet_length = (
            payload_length
            + HEADER_LENGTH
        )

        self.remaining_payload_length = (
            payload_length
        )

        return True


    # --------------------------------------------------------
    # Añadir payload
    # --------------------------------------------------------

    def push_payload(
        self,
        data
    ):

        if not self.working_on_packet:

            return


        n = min(
            len(data),
            self.remaining_payload_length
        )


        if n <= 0:

            return


        self.current_payload.extend(
            bytes(
                data[:n]
            )
        )

        self.remaining_payload_length -= (
            n
        )


    # --------------------------------------------------------
    # Terminar paquete
    # --------------------------------------------------------

    def finish_packet(self):

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


        self.working_on_packet = False

        self.current_header = None

        self.current_payload = bytearray()

        self.current_payload_length = 0

        self.remaining_payload_length = 0

        self.total_packet_length = 0

        self.total_packets += 1


        return result


    # --------------------------------------------------------
    # Cancelar paquete parcial
    # --------------------------------------------------------

    def abort_packet(self):

        self.working_on_packet = False

        self.current_header = None

        self.current_payload = bytearray()

        self.current_payload_length = 0

        self.remaining_payload_length = 0

        self.total_packet_length = 0


    # --------------------------------------------------------
    # Procesar una CADU
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # FHP inválido
        # ----------------------------------------------------

        if (
            fhp < 2047
            and fhp >= MPDU_DATA_SIZE
        ):

            return (
                packets,
                fhp
            )


        offset = 0


        # ====================================================
        # HEADER CCSDS QUE EMPEZÓ AL FINAL DE CADU ANTERIOR
        # ====================================================

        if self.in_header:

            needed = (
                HEADER_LENGTH
                - self.in_header_buffer
            )


            take = min(
                needed,
                MPDU_DATA_SIZE
            )


            self.header_buffer[
                self.in_header_buffer:
                self.in_header_buffer + take
            ] = bytes(
                data[:take]
            )


            self.in_header_buffer += (
                take
            )

            offset = take


            if (
                self.in_header_buffer
                == HEADER_LENGTH
            ):

                self.in_header = False

                self.read_packet_header(
                    self.header_buffer
                )


        # ====================================================
        # CONTINUACIÓN DE PAQUETE DE CADU ANTERIOR
        # ====================================================

        if (
            self.working_on_packet
            and
            self.remaining_payload_length > 0
        ):

            if fhp < 2047:

                # Antes del primer nuevo header hay datos
                # pertenecientes al paquete anterior.
                available = max(
                    0,
                    (
                        fhp + 1
                    )
                    - offset
                )

            else:

                available = (
                    MPDU_DATA_SIZE
                    - offset
                )


            to_write = min(
                self.remaining_payload_length,
                available
            )


            if to_write > 0:

                self.push_payload(
                    data[
                        offset:
                        offset + to_write
                    ]
                )


            if (
                self.remaining_payload_length
                == 0
            ):

                packet = (
                    self.finish_packet()
                )

                if packet is not None:

                    packets.append(
                        packet
                    )


        # ====================================================
        # NO HAY NUEVO HEADER
        # ====================================================

        if fhp >= 2047:

            return (
                packets,
                fhp
            )


        # ====================================================
        # BUSCAR PAQUETES DESDE FIRST HEADER POINTER
        # ====================================================

        pos = fhp


        while (
            pos < MPDU_DATA_SIZE
        ):

            remaining = (
                MPDU_DATA_SIZE
                - pos
            )


            # ------------------------------------------------
            # HEADER PARCIAL AL FINAL DE CADU
            # ------------------------------------------------

            if (
                remaining
                < HEADER_LENGTH
            ):

                self.in_header = True

                self.in_header_buffer = (
                    remaining
                )

                self.header_buffer[:] = (
                    b"\x00"
                    * HEADER_LENGTH
                )

                self.header_buffer[
                    :remaining
                ] = bytes(
                    data[
                        pos:
                    ]
                )

                break


            # ------------------------------------------------
            # LEER HEADER
            # ------------------------------------------------

            if not self.read_packet_header(
                data[
                    pos:
                    pos + HEADER_LENGTH
                ]
            ):

                break


            total_size = (
                self.total_packet_length
            )


            payload_start = (
                pos
                + HEADER_LENGTH
            )


            payload_available = (
                MPDU_DATA_SIZE
                - payload_start
            )


            # ------------------------------------------------
            # PAQUETE COMPLETO EN ESTA CADU
            # ------------------------------------------------

            if (
                payload_available
                >= self.current_payload_length
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


            # ------------------------------------------------
            # PAQUETE CONTINÚA EN SIGUIENTE CADU
            # ------------------------------------------------

            else:

                self.push_payload(
                    data[
                        payload_start:
                    ]
                )

                break


        return (
            packets,
            fhp
        )


# ============================================================
# ANALIZAR HEADER MSU-MR
# ============================================================

def parse_msumr_segment(
    payload
):

    # SatDump necesita al menos 14 bytes de header
    if len(payload) < 14:

        return None


    day_time = (
        (
            int(payload[0])
            << 8
        )
        |
        int(payload[1])
    )


    ms_time = (
        (
            int(payload[2])
            << 24
        )
        |
        (
            int(payload[3])
            << 16
        )
        |
        (
            int(payload[4])
            << 8
        )
        |
        int(payload[5])
    )


    us_time = (
        (
            int(payload[6])
            << 8
        )
        |
        int(payload[7])
    )


    mcun = int(
        payload[8]
    )

    qt = int(
        payload[9]
    )

    dc = (
        int(payload[10])
        >> 4
    ) & 0x0F

    ac = (
        int(payload[10])
        & 0x0F
    )

    qfm = (
        (
            int(payload[11])
            << 8
        )
        |
        int(payload[12])
    )

    qf = int(
        payload[13]
    )


    header_valid = (
        qt == 0x00
        and dc == 0x00
        and ac == 0x00
        and qfm == 0xFFF0
    )


    return {
        "day_time": day_time,
        "ms_time": ms_time,
        "us_time": us_time,

        "mcun": mcun,

        # SatDump:
        # mcu_count = MCUN / 14
        "segment_index": (
            mcun // 14
        ),

        "qt": qt,
        "dc": dc,
        "ac": ac,
        "qfm": qfm,
        "qf": qf,

        "header_valid": (
            header_valid
        ),

        # datos comprimidos después
        # del header de 14 bytes
        "compressed_bytes": (
            len(payload)
            - 14
        )
    }


# ============================================================
# CARGAR CADUs
# ============================================================

raw = np.fromfile(
    CADU_FILE,
    dtype=np.uint8
)


if (
    len(raw)
    % CADU_SIZE
    != 0
):

    raise RuntimeError(
        "El archivo no contiene CADUs "
        "de 1024 bytes completas."
    )


num_cadus = (
    len(raw)
    // CADU_SIZE
)


print()
print(
    "================================================"
)

print(
    "      DEMUX CCSDS / MSU-MR - METEOR LRPT"
)

print(
    "================================================"
)

print()

print(
    f"CADUs disponibles : "
    f"{num_cadus}"
)

print()


# ============================================================
# PROCESAMIENTO
# ============================================================

demuxer = AOSDemuxer()

rows = []

apid_counts = Counter()

channel_counts = Counter()

valid_segment_counts = Counter()

sequence_by_apid = defaultdict(
    list
)


previous_vcdu_counter = None

global_packet_index = 0

vcdu_gaps = 0


for cadu_index in range(
    num_cadus
):

    cadu = raw[
        cadu_index
        * CADU_SIZE:
        (
            cadu_index
            + 1
        )
        * CADU_SIZE
    ]


    vcdu = parse_vcdu(
        cadu
    )


    # --------------------------------------------------------
    # CONTROL DE CONTINUIDAD
    # --------------------------------------------------------

    if (
        previous_vcdu_counter
        is not None
    ):

        gap = (
            (
                vcdu.counter
                - previous_vcdu_counter
            )
            & 0xFFFFFF
        )


        if gap != 1:

            vcdu_gaps += 1

            print(
                f"ADVERTENCIA: salto VCDU "
                f"{previous_vcdu_counter:06X} -> "
                f"{vcdu.counter:06X}"
            )

            # No podemos unir fragmentos a través
            # de una CADU perdida.
            demuxer.reset()


    previous_vcdu_counter = (
        vcdu.counter
    )


    # --------------------------------------------------------
    # MSU-MR usa VCID 5
    # --------------------------------------------------------

    if vcdu.vcid != 5:

        continue


    packets, fhp = (
        demuxer.process(
            cadu
        )
    )


    print(
        f"CADU {cadu_index:03d} | "
        f"CNT=0x{vcdu.counter:06X} | "
        f"FHP={fhp:4d} | "
        f"packets={len(packets)}"
    )


    # ========================================================
    # PAQUETES CCSDS
    # ========================================================

    for (
        header,
        packet_bytes
    ) in packets:

        payload = np.frombuffer(
            packet_bytes[
                HEADER_LENGTH:
            ],
            dtype=np.uint8
        )


        apid = (
            header.apid
        )


        apid_counts[
            apid
        ] += 1


        sequence_by_apid[
            apid
        ].append(
            header.sequence_count
        )


        # ----------------------------------------------------
        # APID 64..69 = canales MSU-MR 1..6
        # ----------------------------------------------------

        if (
            64
            <= apid
            <= 69
        ):

            channel = (
                apid
                - 63
            )

        else:

            channel = None


        segment = None


        if channel is not None:

            channel_counts[
                channel
            ] += 1

            segment = (
                parse_msumr_segment(
                    payload
                )
            )


            if (
                segment is not None
                and segment[
                    "header_valid"
                ]
            ):

                valid_segment_counts[
                    channel
                ] += 1


        # ----------------------------------------------------
        # GUARDAR PAQUETE
        # ----------------------------------------------------

        filename = (
            PACKET_DIR
            /
            (
                f"packet_{global_packet_index:05d}"
                f"_apid_{apid}"
                f"_seq_{header.sequence_count:05d}.bin"
            )
        )


        np.frombuffer(
            packet_bytes,
            dtype=np.uint8
        ).tofile(
            filename
        )


        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        row = {
            "packet_index":
                global_packet_index,

            "cadu_index":
                cadu_index,

            "vcdu_counter":
                vcdu.counter,

            "apid":
                apid,

            "channel":
                (
                    ""
                    if channel is None
                    else channel
                ),

            "sequence_count":
                header.sequence_count,

            "sequence_flag":
                header.sequence_flag,

            "packet_length_field":
                header.packet_length,

            "payload_bytes":
                len(payload),

            "day_time":
                "",

            "ms_time":
                "",

            "us_time":
                "",

            "mcun":
                "",

            "segment_index":
                "",

            "qt":
                "",

            "dc":
                "",

            "ac":
                "",

            "qfm":
                "",

            "qf":
                "",

            "segment_header_valid":
                "",

            "compressed_bytes":
                ""
        }


        if segment is not None:

            row.update(
                {
                    "day_time":
                        segment[
                            "day_time"
                        ],

                    "ms_time":
                        segment[
                            "ms_time"
                        ],

                    "us_time":
                        segment[
                            "us_time"
                        ],

                    "mcun":
                        segment[
                            "mcun"
                        ],

                    "segment_index":
                        segment[
                            "segment_index"
                        ],

                    "qt":
                        segment[
                            "qt"
                        ],

                    "dc":
                        segment[
                            "dc"
                        ],

                    "ac":
                        segment[
                            "ac"
                        ],

                    "qfm":
                        (
                            f"0x"
                            f"{segment['qfm']:04X}"
                        ),

                    "qf":
                        segment[
                            "qf"
                        ],

                    "segment_header_valid":
                        int(
                            segment[
                                "header_valid"
                            ]
                        ),

                    "compressed_bytes":
                        segment[
                            "compressed_bytes"
                        ]
                }
            )


        rows.append(
            row
        )


        global_packet_index += 1


# ============================================================
# CSV
# ============================================================

fieldnames = [
    "packet_index",
    "cadu_index",
    "vcdu_counter",

    "apid",
    "channel",

    "sequence_count",
    "sequence_flag",

    "packet_length_field",
    "payload_bytes",

    "day_time",
    "ms_time",
    "us_time",

    "mcun",
    "segment_index",

    "qt",
    "dc",
    "ac",
    "qfm",
    "qf",

    "segment_header_valid",

    "compressed_bytes"
]


with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# RESUMEN APID
# ============================================================

print()
print(
    "================================================"
)

print(
    "                   APIDs"
)

print(
    "================================================"
)

print()


for apid in sorted(
    apid_counts
):

    if (
        64
        <= apid
        <= 69
    ):

        channel = (
            apid
            - 63
        )

        label = (
            f"MSU-MR canal {channel}"
        )

    else:

        label = "otro"


    print(
        f"APID {apid:4d} | "
        f"{apid_counts[apid]:4d} paquetes | "
        f"{label}"
    )


# ============================================================
# CONTINUIDAD DE SECUENCIA CCSDS
# ============================================================

print()
print(
    "================================================"
)

print(
    "            CONTINUIDAD CCSDS"
)

print(
    "================================================"
)

print()


for apid in sorted(
    sequence_by_apid
):

    seqs = (
        sequence_by_apid[
            apid
        ]
    )


    gaps = 0


    for i in range(
        1,
        len(seqs)
    ):

        expected = (
            seqs[i - 1]
            + 1
        ) & 0x3FFF


        if (
            seqs[i]
            != expected
        ):

            gaps += 1


    print(
        f"APID {apid:4d} | "
        f"paquetes={len(seqs):4d} | "
        f"seq inicial={seqs[0]:5d} | "
        f"seq final={seqs[-1]:5d} | "
        f"saltos={gaps}"
    )


# ============================================================
# SEGMENTOS MSU-MR
# ============================================================

print()
print(
    "================================================"
)

print(
    "               SEGMENTOS MSU-MR"
)

print(
    "================================================"
)

print()


for channel in range(
    1,
    7
):

    packets = (
        channel_counts[
            channel
        ]
    )

    valid = (
        valid_segment_counts[
            channel
        ]
    )


    print(
        f"Canal {channel} | "
        f"paquetes={packets:4d} | "
        f"headers válidos={valid:4d}"
    )


# ============================================================
# PRIMEROS SEGMENTOS
# ============================================================

print()
print(
    "================================================"
)

print(
    "          PRIMEROS SEGMENTOS VÁLIDOS"
)

print(
    "================================================"
)

print()


shown = 0


for row in rows:

    if (
        row[
            "segment_header_valid"
        ]
        == 1
    ):

        print(
            f"PKT={row['packet_index']:4d} | "
            f"APID={row['apid']} | "
            f"CH={row['channel']} | "
            f"SEQ={row['sequence_count']:5d} | "
            f"MCUN={row['mcun']:3d} | "
            f"SEG={row['segment_index']:2d} | "
            f"QF={row['qf']:3d} | "
            f"comp={row['compressed_bytes']:4d} bytes"
        )


        shown += 1


        if shown >= 20:

            break


# ============================================================
# RESUMEN FINAL
# ============================================================

total_msumr = sum(
    channel_counts.values()
)

total_valid_segments = sum(
    valid_segment_counts.values()
)


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
    f"CADUs procesadas          : "
    f"{num_cadus}"
)

print(
    f"Saltos de VCDU            : "
    f"{vcdu_gaps}"
)

print(
    f"Paquetes CCSDS completos  : "
    f"{global_packet_index}"
)

print(
    f"Paquetes MSU-MR 64..69    : "
    f"{total_msumr}"
)

print(
    f"Segment headers válidos   : "
    f"{total_valid_segments}"
)

print()

print(
    "CSV:"
)

print(
    CSV_FILE
)

print()

print(
    "Paquetes:"
)

print(
    PACKET_DIR
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