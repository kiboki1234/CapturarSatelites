from pathlib import Path
from dataclasses import dataclass
import numpy as np


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CADU_FILE = (
    ROOT
    / "outputs"
    / "decoded"
    / "reed_solomon"
    / "meteor_lrpt_valid.cadu"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "ccsds_packets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURACION METEOR LRPT
# ============================================================

CADU_SIZE = 1024

# SatDump:
# Demuxer(882, true)
MPDU_DATA_SIZE = 882

HAS_INSERT_ZONE = True
INSERT_ZONE_SIZE = 2

# Con zona de inserción:
#
# VCDU header : bytes 4..9
# Insert zone : bytes 10..11
# MPDU header : bytes 12..13
# MPDU data   : byte 14...
#
MPDU_HEADER_OFFSET = 12
MPDU_DATA_OFFSET = 14

CCSDS_HEADER_SIZE = 6


# ============================================================
# ESTRUCTURAS
# ============================================================

@dataclass
class VCDU:
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


# ============================================================
# PARSER VCDU
# ============================================================

def parse_vcdu(cadu):

    version = (
        cadu[4] >> 6
    )

    spacecraft_id = (
        ((cadu[4] & 0x3F) << 2)
        | (cadu[5] >> 6)
    )

    vcid = (
        cadu[5] & 0x3F
    )

    counter = (
        (int(cadu[6]) << 16)
        | (int(cadu[7]) << 8)
        | int(cadu[8])
    )

    replay_flag = (
        cadu[9] >> 7
    )

    return VCDU(
        version,
        spacecraft_id,
        vcid,
        counter,
        replay_flag
    )


# ============================================================
# FIRST HEADER POINTER
# ============================================================

def parse_first_header_pointer(cadu):

    offset = MPDU_HEADER_OFFSET

    return (
        ((int(cadu[offset]) & 0x07) << 8)
        | int(cadu[offset + 1])
    )


# ============================================================
# CCSDS PRIMARY HEADER
# ============================================================

def parse_ccsds_header(data):

    if len(data) < 6:
        return None

    version = (
        data[0] >> 5
    )

    packet_type = (
        (data[0] >> 4) & 1
    )

    secondary_header = (
        (data[0] >> 3) & 1
    )

    apid = (
        ((int(data[0]) & 0x07) << 8)
        | int(data[1])
    )

    sequence_flag = (
        data[2] >> 6
    )

    sequence_count = (
        ((int(data[2]) & 0x3F) << 8)
        | int(data[3])
    )

    packet_length = (
        (int(data[4]) << 8)
        | int(data[5])
    )

    return CCSDSHeader(
        version,
        packet_type,
        secondary_header,
        apid,
        sequence_flag,
        sequence_count,
        packet_length
    )


# ============================================================
# FORMATO HEX
# ============================================================

def hex_string(data, n=32):

    return " ".join(
        f"{int(x):02X}"
        for x in data[:n]
    )


# ============================================================
# DEMUXER SIMPLE
#
# Replica la lógica necesaria para nuestras CADUs consecutivas.
# ============================================================

class Demuxer:

    def __init__(self):

        self.partial_packet = bytearray()
        self.expected_packet_size = None

        self.packet_index = 0


    def save_packet(self, packet_bytes):

        header = parse_ccsds_header(
            packet_bytes[:6]
        )

        if header is None:
            return


        filename = (
            OUTPUT_DIR
            / (
                f"packet_{self.packet_index:04d}"
                f"_apid_{header.apid}.bin"
            )
        )

        np.frombuffer(
            bytes(packet_bytes),
            dtype=np.uint8
        ).tofile(
            filename
        )

        print(
            f"  PACKET {self.packet_index:03d} | "
            f"APID={header.apid:4d} | "
            f"SEQ={header.sequence_count:5d} | "
            f"flag={header.sequence_flag} | "
            f"payload={header.packet_length + 1:4d} | "
            f"total={len(packet_bytes):4d}"
        )

        print(
            f"             HDR: "
            f"{hex_string(packet_bytes, 16)}"
        )

        self.packet_index += 1


    def process_cadu(self, cadu):

        fhp = parse_first_header_pointer(
            cadu
        )

        data = cadu[
            MPDU_DATA_OFFSET:
            MPDU_DATA_OFFSET + MPDU_DATA_SIZE
        ]


        # ----------------------------------------------------
        # Si hay un paquete empezado en la CADU anterior,
        # los primeros bytes pertenecen a su continuación.
        # ----------------------------------------------------

        if self.partial_packet:

            if fhp < 2047:

                continuation = data[
                    :fhp
                ]

            else:

                continuation = data


            self.partial_packet.extend(
                continuation.tobytes()
            )


            if self.expected_packet_size is not None:

                if (
                    len(self.partial_packet)
                    >= self.expected_packet_size
                ):

                    packet = (
                        self.partial_packet[
                            :self.expected_packet_size
                        ]
                    )

                    self.save_packet(
                        packet
                    )

                    self.partial_packet = bytearray()
                    self.expected_packet_size = None


        # ----------------------------------------------------
        # 2047 = no comienza paquete nuevo aquí
        # ----------------------------------------------------

        if fhp >= 2047:

            return


        if fhp >= MPDU_DATA_SIZE:

            print(
                f"  FHP inválido: {fhp}"
            )

            return


        # ----------------------------------------------------
        # Recorrer paquetes desde el FHP
        # ----------------------------------------------------

        pos = fhp


        while pos < MPDU_DATA_SIZE:

            remaining = (
                MPDU_DATA_SIZE - pos
            )


            # Header parcial
            if remaining < CCSDS_HEADER_SIZE:

                self.partial_packet = bytearray(
                    data[pos:].tobytes()
                )

                self.expected_packet_size = None

                return


            header = parse_ccsds_header(
                data[
                    pos:
                    pos + 6
                ]
            )


            # Payload = packet_length + 1
            total_size = (
                CCSDS_HEADER_SIZE
                + header.packet_length
                + 1
            )


            # ------------------------------------------------
            # VALIDACIONES BÁSICAS
            # ------------------------------------------------

            if (
                header.version > 0
                or total_size <= 6
                or total_size > 65542
            ):

                print(
                    f"  Header no plausible en offset {pos}: "
                    f"version={header.version}, "
                    f"APID={header.apid}, "
                    f"size={total_size}"
                )

                return


            # ------------------------------------------------
            # Paquete completo dentro de esta CADU
            # ------------------------------------------------

            if pos + total_size <= MPDU_DATA_SIZE:

                packet = data[
                    pos:
                    pos + total_size
                ]

                self.save_packet(
                    packet.tobytes()
                )

                pos += total_size


            # ------------------------------------------------
            # Paquete continúa en la CADU siguiente
            # ------------------------------------------------

            else:

                self.partial_packet = bytearray(
                    data[pos:].tobytes()
                )

                self.expected_packet_size = (
                    total_size
                )

                return


# ============================================================
# CARGAR CADUS
# ============================================================

raw = np.fromfile(
    CADU_FILE,
    dtype=np.uint8
)


if len(raw) % CADU_SIZE != 0:

    raise RuntimeError(
        "El archivo CADU no tiene múltiplos de 1024 bytes."
    )


num_cadus = (
    len(raw)
    // CADU_SIZE
)


print()
print("================================================")
print("          ANALISIS CCSDS / METEOR LRPT")
print("================================================")
print()

print(
    f"CADUs disponibles : {num_cadus}"
)

print()


demuxer = Demuxer()


# ============================================================
# PROCESAR
# ============================================================

for i in range(num_cadus):

    cadu = raw[
        i * CADU_SIZE:
        (i + 1) * CADU_SIZE
    ]

    vcdu = parse_vcdu(
        cadu
    )

    fhp = parse_first_header_pointer(
        cadu
    )


    print(
        "------------------------------------------------"
    )

    print(
        f"CADU {i}"
    )

    print(
        f"Version        : {vcdu.version}"
    )

    print(
        f"Spacecraft ID  : {vcdu.spacecraft_id}"
    )

    print(
        f"VCID           : {vcdu.vcid}"
    )

    print(
        f"VCDU counter   : "
        f"0x{vcdu.counter:06X} "
        f"({vcdu.counter})"
    )

    print(
        f"Replay flag    : {vcdu.replay_flag}"
    )

    print(
        f"First Header Pointer : {fhp}"
    )

    print()


    if vcdu.vcid != 5:

        print(
            "VCID distinto de 5; SatDump no lo "
            "usaría para MSU-MR."
        )

        print()

        continue


    demuxer.process_cadu(
        cadu
    )

    print()


# ============================================================
# RESUMEN
# ============================================================

print()
print("================================================")
print("                   RESUMEN")
print("================================================")
print()

print(
    f"CADUs procesadas  : {num_cadus}"
)

print(
    f"Paquetes completos: {demuxer.packet_index}"
)

if demuxer.partial_packet:

    print(
        f"Paquete parcial   : "
        f"{len(demuxer.partial_packet)} bytes"
    )

print()

print(
    "Paquetes guardados en:"
)

print(
    OUTPUT_DIR
)