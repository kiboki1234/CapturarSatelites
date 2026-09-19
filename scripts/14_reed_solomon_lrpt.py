from pathlib import Path
import numpy as np
import reedsolo as rs


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "derandomized"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "decoded"
    / "reed_solomon"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CADU_FILE = (
    OUTPUT_DIR
    / "meteor_lrpt_valid.cadu"
)


# ============================================================
# PARAMETROS CCSDS RS
# ============================================================

N = 255
K = 223
NSYM = 32

INTERLEAVE = 4

PRIMITIVE_POLY = 0x187
FCR = 112

# ------------------------------------------------------------
# IMPORTANTE
#
# SatDump usa raíces:
#
# alpha ^ (11 * (i + 112))
#
# reedsolo construye:
#
# generator ^ (i + fcr)
#
# Si generator = alpha^11, ambas expresiones coinciden.
#
# En GF(256) usando 0x187:
#
# alpha^11 = 0xAD
# ------------------------------------------------------------

GENERATOR = 0xAD


# ============================================================
# INICIALIZAR GF(256)
# ============================================================

rs.init_tables(
    prim=PRIMITIVE_POLY,
    generator=GENERATOR,
    c_exp=8
)


# ============================================================
# FUNCIONES
# ============================================================

def hex_string(data, n=32):

    return " ".join(
        f"{int(x):02X}"
        for x in data[:n]
    )


def decode_codeword(codeword):

    cw = bytearray(
        codeword.tobytes()
    )

    # --------------------------------------------------------
    # Síndromes ANTES de corregir
    # --------------------------------------------------------

    syndromes = rs.rs_calc_syndromes(
        cw,
        NSYM,
        fcr=FCR,
        generator=GENERATOR
    )

    nonzero_before = sum(
        s != 0
        for s in syndromes[1:]
    )

    clean_before = (
        nonzero_before == 0
    )


    # --------------------------------------------------------
    # Si ya está limpio, no necesitamos corregir
    # --------------------------------------------------------

    if clean_before:

        return {
            "success": True,
            "corrected": np.array(
                list(cw),
                dtype=np.uint8
            ),
            "errors": 0,
            "syndromes_before": 0,
            "syndromes_after": 0
        }


    # --------------------------------------------------------
    # REED-SOLOMON
    # --------------------------------------------------------

    try:

        result = rs.rs_correct_msg(
            cw,
            NSYM,
            fcr=FCR,
            generator=GENERATOR
        )

        # Versiones modernas:
        # message, ecc, errata_pos

        if len(result) == 3:

            message, ecc, errata_pos = result

        else:

            message, ecc = result
            errata_pos = []


        corrected_bytes = (
            bytes(message)
            + bytes(ecc)
        )

        corrected = np.frombuffer(
            corrected_bytes,
            dtype=np.uint8
        ).copy()


        # ----------------------------------------------------
        # VALIDAR DESPUÉS
        # ----------------------------------------------------

        syndromes_after = rs.rs_calc_syndromes(
            bytearray(corrected.tobytes()),
            NSYM,
            fcr=FCR,
            generator=GENERATOR
        )

        nonzero_after = sum(
            s != 0
            for s in syndromes_after[1:]
        )


        return {
            "success": nonzero_after == 0,
            "corrected": corrected,
            "errors": len(errata_pos),
            "syndromes_before": nonzero_before,
            "syndromes_after": nonzero_after
        }


    except rs.ReedSolomonError:

        return {
            "success": False,
            "corrected": codeword.copy(),
            "errors": -1,
            "syndromes_before": nonzero_before,
            "syndromes_after": -1
        }


# ============================================================
# PROCESAMIENTO
# ============================================================

print()
print("================================================")
print("       REED-SOLOMON METEOR LRPT / CCSDS")
print("================================================")
print()

print(f"RS                     : ({N}, {K})")
print(f"Paridad                : {NSYM} bytes")
print(f"Interleave             : {INTERLEAVE}")
print(f"Primitive polynomial   : 0x{PRIMITIVE_POLY:X}")
print(f"First root             : {FCR}")
print(f"Generator alpha^11     : 0x{GENERATOR:02X}")

print()


valid_frames = []
total_frames = 0


for frame_file in sorted(
    INPUT_DIR.glob("derand_*.bin")
):

    total_frames += 1

    frame = np.fromfile(
        frame_file,
        dtype=np.uint8
    )


    print()
    print("================================================")
    print(frame_file.name)
    print("================================================")


    if len(frame) != 1024:

        print(
            f"ERROR: tamaño {len(frame)}, "
            "se esperaban 1024 bytes."
        )

        continue


    # --------------------------------------------------------
    # ASM
    # --------------------------------------------------------

    print()

    print(
        "ASM:"
    )

    print(
        hex_string(
            frame[:4],
            4
        )
    )


    # --------------------------------------------------------
    # Los 1020 bytes protegidos por RS
    # --------------------------------------------------------

    rs_area = frame[
        4:
    ].copy()


    if len(rs_area) != 1020:

        print(
            "ERROR: área RS incorrecta."
        )

        continue


    corrected_area = rs_area.copy()

    all_ok = True
    errors_per_branch = []


    # ========================================================
    # 4 CODEWORDS INTERLEAVED
    #
    # Rama 0:
    #   bytes 0,4,8,12,...
    #
    # Rama 1:
    #   bytes 1,5,9,13,...
    #
    # etc.
    # ========================================================

    print()
    print(
        "Ramas Reed-Solomon:"
    )

    print()


    for branch in range(INTERLEAVE):

        codeword = rs_area[
            branch::INTERLEAVE
        ]


        if len(codeword) != 255:

            print(
                f"RS{branch}: longitud incorrecta "
                f"{len(codeword)}"
            )

            all_ok = False
            continue


        result = decode_codeword(
            codeword
        )


        success = result[
            "success"
        ]

        errors = result[
            "errors"
        ]

        synd_before = result[
            "syndromes_before"
        ]

        synd_after = result[
            "syndromes_after"
        ]


        if success:

            estado = "OK"

        else:

            estado = "FALLO"
            all_ok = False


        print(
            f"RS{branch} | "
            f"síndromes antes={synd_before:2d} | "
            f"errores={errors:3d} | "
            f"síndromes después={synd_after:2d} | "
            f"{estado}"
        )


        errors_per_branch.append(
            errors
        )


        # ----------------------------------------------------
        # REINTERLEAVE
        # ----------------------------------------------------

        corrected_area[
            branch::INTERLEAVE
        ] = result[
            "corrected"
        ]


    # ========================================================
    # RECONSTRUIR CADU
    # ========================================================

    corrected_frame = np.empty(
        1024,
        dtype=np.uint8
    )

    corrected_frame[:4] = [
        0x1D,
        0xCF,
        0xFC,
        0x1D
    ]

    corrected_frame[
        4:
    ] = corrected_area


    # ========================================================
    # CONTADOR VCDU
    # ========================================================

    vcdu_counter = int.from_bytes(
        corrected_frame[
            6:9
        ].tobytes(),
        byteorder="big"
    )


    print()

    print(
        f"VCDU counter : "
        f"0x{vcdu_counter:06X} "
        f"({vcdu_counter})"
    )


    print()

    print(
        "Primeros 32 bytes después de RS:"
    )

    print(
        hex_string(
            corrected_frame,
            32
        )
    )


    # ========================================================
    # GUARDAR
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / frame_file.name.replace(
            "derand_",
            "rs_"
        )
    )

    corrected_frame.tofile(
        output_file
    )


    print()

    print(
        "Guardado:"
    )

    print(
        output_file
    )


    if all_ok:

        valid_frames.append(
            corrected_frame
        )

        print()

        print(
            ">>> CADU VALIDA <<<"
        )

    else:

        print()

        print(
            ">>> CADU DESCARTADA POR RS <<<"
        )


# ============================================================
# GUARDAR CADUS VALIDAS JUNTAS
# ============================================================

print()
print("================================================")
print("                   RESUMEN")
print("================================================")
print()

print(
    f"Tramas procesadas : {total_frames}"
)

print(
    f"CADUs válidas     : {len(valid_frames)}"
)


if valid_frames:

    all_valid = np.concatenate(
        valid_frames
    )

    all_valid.tofile(
        CADU_FILE
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
        f"Tamaño: "
        f"{len(all_valid):,} bytes"
    )


print()
print("================================================")
print("FIN")
print("================================================")