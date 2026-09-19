# Taller Satélite

Guía para instalar y ejecutar el proyecto de procesamiento de señales de
satélites METEOR. Los scripts inspeccionan una grabación I/Q, demodulan LRPT,
recuperan paquetes CCSDS y reconstruyen imágenes MSU-MR.

La numeración de los archivos indica el orden normal del procesamiento. Una
persona que clone o copie este proyecto puede seguir esta guía sin instalar
las dependencias manualmente una por una.

## Requisitos

- Windows 10/11.
- Python 3.10 o superior.
- Una grabación I/Q WAV en
  `data/raw/meteor_137100/meteor_137100_iq.wav`.

## Empezar desde cero

1. Descarga o clona este proyecto.
2. Abre PowerShell en la carpeta raíz, la que contiene `scripts`, `data` y
  `README.md`.
3. Crea y activa el entorno virtual siguiendo la sección siguiente.
4. Instala las versiones congeladas con `requirements.txt`.
5. Ejecuta uno de los recorridos documentados más abajo.

No es necesario ejecutar los scripts desde la carpeta `scripts`: todos calculan
la raíz del proyecto automáticamente. Sí es necesario conservar la estructura
de carpetas y colocar la grabación en la ruta indicada.

Los scripts actuales usan esa ruta de entrada y no reciben argumentos por
línea de comandos. Las rutas de salida se crean dentro de `outputs/`.

## Crear el entorno virtual

Abre PowerShell en la raíz del proyecto:

```powershell
cd C:\Users\andres\Downloads\TallerSatelite
py -3 -m venv .venv
```

Activa el entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Si el entorno está activo, el prompt de PowerShell mostrará `(.venv)`.

## Instalar dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` contiene las versiones congeladas de las
dependencias del entorno virtual.

Comprueba que las dependencias principales se pueden importar:

```powershell
python -c "import numpy, matplotlib, scipy, reedsolo, PIL, numba; print('Entorno correcto')"
```

También puedes comprobar la sintaxis de todos los scripts:

```powershell
python -m compileall scripts
```

## Ejecutar el pipeline clásico de 10 segundos

Ejecuta los comandos desde la raíz del proyecto y en este orden:

```powershell
python scripts\01_inspeccionar_iq.py
python scripts\02_espectro_iq.py
python scripts\03_waterfall_iq.py
python scripts\04_espectro_promedio.py
python scripts\05_extraer_canal_lrpt.py
python scripts\06_constelacion_bruta.py
python scripts\07_correccion_frecuencia.py
python scripts\08_sincronizacion_simbolos.py
python scripts\09_bits_desde_constelacion.py
python scripts\10_buscar_sync_lrpt.py
python scripts\11_validar_tramas_lrpt.py
python scripts\12_viterbi_lrpt.py
python scripts\13_derandomizar_ccsds.py
python scripts\14_reed_solomon_lrpt.py
python scripts\15_extraer_paquetes_ccsds.py
```

Este recorrido genera datos intermedios en `outputs/decoded/`, gráficos en
`outputs/plots/` y `outputs/spectra/`, y CADUs válidas en
`outputs/decoded/reed_solomon/`.

### Qué hace cada script

| Script | Función | Necesita previamente |
| --- | --- | --- |
| `01_inspeccionar_iq.py` | Muestra los metadatos y las primeras muestras I/Q. | La grabación WAV |
| `02_espectro_iq.py` | Calcula y guarda un espectro FFT. | La grabación WAV |
| `03_waterfall_iq.py` | Genera un waterfall de la señal. | La grabación WAV |
| `04_espectro_promedio.py` | Calcula el espectro promedio. | La grabación WAV |
| `05_extraer_canal_lrpt.py` | Filtra y remuestrea el canal LRPT a 288 kS/s. | La grabación WAV |
| `06_constelacion_bruta.py` | Grafica la constelación sin corregir. | `lrpt_channel_288ksps.npy` |
| `07_correccion_frecuencia.py` | Estima y corrige el desplazamiento de frecuencia. | `lrpt_channel_288ksps.npy` |
| `08_sincronizacion_simbolos.py` | Aplica RRC y prepara la sincronización de símbolos. | La señal corregida |
| `09_bits_desde_constelacion.py` | Realiza una primera decisión QPSK y guarda bits. | La señal corregida |
| `10_buscar_sync_lrpt.py` | Busca la palabra de sincronización LRPT. | La señal corregida |
| `11_validar_tramas_lrpt.py` | Comprueba la separación y la correlación de tramas. | Símbolos blandos LRPT |
| `12_viterbi_lrpt.py` | Decodifica el código convolucional CCSDS. | Tramas LRPT válidas |
| `13_derandomizar_ccsds.py` | Elimina el aleatorizador CCSDS. | Tramas Viterbi |
| `14_reed_solomon_lrpt.py` | Corrige errores Reed-Solomon y genera CADUs. | Tramas derandomizadas |
| `15_extraer_paquetes_ccsds.py` | Extrae paquetes CCSDS desde las CADUs. | CADUs corregidas |

Los scripts `01` a `05` son análisis y preparación. Los scripts `06` a `15`
forman la cadena de demodulación y decodificación; por eso sus archivos de
salida son los prerrequisitos de los pasos siguientes.

## Demodulación adaptativa y MSU-MR

Los scripts `16` a `20` forman otro recorrido de 10 segundos. El script `17`
debe ejecutarse antes que el `16`, porque produce los símbolos adaptativos que
este necesita:

```powershell
python scripts\17_demodulacion_adaptativa_lrpt.py
python scripts\16_decodificar_todas_cadus.py
python scripts\18_extraer_paquetes_msumr.py
python scripts\19_reconstruir_imagen_msumr.py
python scripts\20_composicion_rgb_msumr.py
```

La imagen RGB final y los canales reconstruidos se guardan en
`outputs/images/msumr_10s/`.

En este recorrido, `17_demodulacion_adaptativa_lrpt.py` reemplaza la cadena
manual de sincronización para producir símbolos blandos con recuperación
adaptativa. Después, `16` decodifica las CADUs y `18` a `20` extraen paquetes,
reconstruyen los canales y generan la composición RGB.

| Script | Función | Necesita previamente |
| --- | --- | --- |
| `16_decodificar_todas_cadus.py` | Decodifica todas las CADUs con los símbolos adaptativos. | `lrpt_soft_symbols_adaptive.npy` |
| `17_demodulacion_adaptativa_lrpt.py` | Hace recuperación adaptativa de portadora y reloj. | La señal corregida |
| `18_extraer_paquetes_msumr.py` | Extrae los paquetes de imagen MSU-MR. | El CADU de 10 segundos |
| `19_reconstruir_imagen_msumr.py` | Reconstruye los tres canales de imagen. | Los paquetes MSU-MR |
| `20_composicion_rgb_msumr.py` | Alinea los canales y crea la composición RGB. | Los tres canales reconstruidos |

## Procesar la grabación completa

Para procesar el pase completo, usa `21` y después `22`:

```powershell
python scripts\21_pipeline_fullpass_lrpt.py
python scripts\22_reconstruir_fullpass_msumr.py
```

Este recorrido crea el CADU completo en
`outputs/decoded/fullpass/meteor_lrpt_full.cadu` y las imágenes del pase en
`outputs/images/msumr_fullpass/`.

`21_pipeline_fullpass_lrpt.py` es una alternativa al recorrido de 10 segundos:
procesa toda la grabación por bloques. No es necesario ejecutar antes los
scripts `01` a `20`; únicamente debe existir la grabación WAV en la ruta de
entrada. El script `22` depende del CADU completo producido por `21`.

| Script | Función | Necesita previamente |
| --- | --- | --- |
| `21_pipeline_fullpass_lrpt.py` | Demodula y decodifica toda la grabación por bloques. | La grabación WAV |
| `22_reconstruir_fullpass_msumr.py` | Reconstruye las imágenes MSU-MR del pase completo. | El CADU producido por `21` |

El script `21` puede tardar varios minutos y requiere bastante memoria y
espacio en disco, porque procesa la grabación por bloques y conserva resultados
intermedios en `outputs/decoded/fullpass/chunks/`.

## Estructura resumida

```text
data/raw/          Grabaciones I/Q de entrada
scripts/            Procesamiento numerado
outputs/decoded/    Señales, símbolos, paquetes y CADUs
outputs/images/     Imágenes MSU-MR reconstruidas
outputs/plots/      Constelaciones
outputs/spectra/    Espectros y waterfalls
```

## Solución de problemas

### No se puede activar `.venv`

Ejecuta la política de ejecución solo para la sesión actual de PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Falta un módulo

Comprueba que el entorno está activo y reinstala las dependencias:

```powershell
python -m pip install -r requirements.txt
```

### No existe el archivo WAV

Verifica que la grabación esté exactamente en:

```text
data/raw/meteor_137100/meteor_137100_iq.wav
```

Si vas a procesar otra grabación, cambia las constantes de ruta y los
parámetros de frecuencia de los scripts que la utilizan.