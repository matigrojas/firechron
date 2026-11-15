"""
Limpieza de CSVs de NASA FIRMS (MODIS + VIIRS, archive y NRT) -> Parquet unificado.

Uso:
    python limpieza_firms.py --input data/raw/*.csv --output data/parquet

Qué hace:
  1. Detecta sensor/satélite y tipo de producto (archive / nrt) de cada archivo.
  2. Normaliza columnas para que MODIS y VIIRS tengan el mismo esquema.
  3. Lleva la confianza a una escala común: low / nominal / high.
  4. Arma fecha-hora UTC, año y mes.
  5. Asigna una celda de grilla de ~1 km (0.01°) para agregar después.
  6. Evita duplicados archive vs NRT: del NRT sólo se queda con fechas
     posteriores a la última fecha del archive del mismo satélite.
  7. Escribe un dataset Parquet particionado por satélite y año.

Lee los CSV por bloques (chunks), así que aguanta archivos grandes.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 1_000_000          # filas por bloque
GRID_DEG = 0.01            # ~1 km (en lon es algo menos a latitudes altas)

# Columnas finales (mismo esquema para todos los sensores)
COLS_OUT = [
    "latitude", "longitude", "datetime_utc", "acq_date", "year", "month",
    "satellite", "sensor", "product", "conf_raw", "conf_class",
    "daynight", "type", "frp", "bt_mir", "bt_tir", "scan", "track",
    "cell_r", "cell_c",
]

# Código de satélite en el CSV -> nombre normalizado
SAT_MAP = {
    "terra": "Terra", "t": "Terra",
    "aqua": "Aqua", "a": "Aqua",
    "n": "SNPP", "npp": "SNPP", "snpp": "SNPP",
    "n20": "NOAA20", "1": "NOAA20", "j1": "NOAA20", "noaa-20": "NOAA20",
    "n21": "NOAA21", "2": "NOAA21", "j2": "NOAA21", "noaa-21": "NOAA21",
}

# Pista desde el nombre de archivo (fallback)
FILE_HINTS = [
    (r"M-C61|MODIS", "MODIS", None),
    (r"SV-C2", "VIIRS", "SNPP"),
    (r"J1V-C2", "VIIRS", "NOAA20"),
    (r"J2V-C2", "VIIRS", "NOAA21"),
]


def info_archivo(path):
    name = os.path.basename(path)
    product = "nrt" if "nrt" in name.lower() else "archive"
    sensor, sat = None, None
    for pat, s, st in FILE_HINTS:
        if re.search(pat, name, re.IGNORECASE):
            sensor, sat = s, st
            break
    return product, sensor, sat


def conf_modis(x):
    # MODIS: 0-100.  <30 low, 30-79 nominal, >=80 high (User Guide C6)
    x = pd.to_numeric(x, errors="coerce")
    return pd.cut(x, [-1, 29, 79, 100], labels=["low", "nominal", "high"]).astype(str)


def conf_viirs(x):
    m = {"l": "low", "low": "low", "n": "nominal", "nominal": "nominal",
         "h": "high", "high": "high"}
    return x.astype(str).str.strip().str.lower().map(m)


def limpiar_chunk(df, product, sensor_hint, sat_hint):
    # Nombres de bandas: VIIRS a veces viene como bright_ti4/ti5, otras con
    # nombres "estilo MODIS" (brightness/bright_t31). No son comparables entre
    # sensores: se guardan con nombres neutros.
    df = df.rename(columns={
        "brightness": "bt_mir", "bright_ti4": "bt_mir",
        "bright_t31": "bt_tir", "bright_ti5": "bt_tir",
    })

    # Sensor
    if "instrument" in df.columns:
        sensor = df["instrument"].astype(str).str.upper().str.strip()
    else:
        sensor = pd.Series(sensor_hint, index=df.index)
    df["sensor"] = sensor.fillna(sensor_hint)

    # Satélite
    sat = df["satellite"].astype(str).str.strip().str.lower().map(SAT_MAP)
    if sat_hint:
        sat = sat.fillna(sat_hint)
    df["satellite"] = sat

    # Confianza
    df["conf_raw"] = df["confidence"].astype(str)
    es_modis = df["sensor"].eq("MODIS")
    df["conf_class"] = None
    df.loc[es_modis, "conf_class"] = conf_modis(df.loc[es_modis, "confidence"])
    df.loc[~es_modis, "conf_class"] = conf_viirs(df.loc[~es_modis, "confidence"])

    # Fecha y hora (acq_time viene como HHMM en UTC; puede perder el 0 inicial)
    hhmm = df["acq_time"].astype(str).str.split(".").str[0].str.zfill(4)
    df["datetime_utc"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + hhmm, format="%Y-%m-%d %H%M", utc=True
    )
    df["acq_date"] = df["datetime_utc"].dt.date
    df["year"] = df["datetime_utc"].dt.year.astype("int16")
    df["month"] = df["datetime_utc"].dt.month.astype("int8")

    # type (sólo existe en archive): 0 veg., 1 volcán, 2 fuente fija, 3 offshore
    df["type"] = pd.to_numeric(df["type"], errors="coerce").astype("Int8") \
        if "type" in df.columns else pd.array([pd.NA] * len(df), dtype="Int8")

    df["product"] = product
    df["daynight"] = df["daynight"].astype(str).str.upper()

    # Grilla ~1 km
    df["cell_r"] = np.floor(df["latitude"] / GRID_DEG).astype("int32")
    df["cell_c"] = np.floor(df["longitude"] / GRID_DEG).astype("int32")

    for c in COLS_OUT:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS_OUT]


def procesar(inputs, outdir):
    files = sorted({f for pat in inputs for f in glob.glob(pat)})
    if not files:
        raise SystemExit("No se encontraron CSVs.")
    # archive primero, para saber hasta qué fecha llega cada satélite
    files.sort(key=lambda f: info_archivo(f)[0] != "archive")

    os.makedirs(outdir, exist_ok=True)
    max_archive = {}          # satélite -> última fecha del archive
    resumen = []

    for f in files:
        product, sensor_hint, sat_hint = info_archivo(f)
        n_in = n_out = 0
        for i, chunk in enumerate(pd.read_csv(f, chunksize=CHUNK,
                                              dtype={"acq_time": str})):
            n_in += len(chunk)
            df = limpiar_chunk(chunk, product, sensor_hint, sat_hint)

            if product == "archive":
                for sat, d in df.groupby("satellite")["acq_date"].max().items():
                    max_archive[sat] = max(d, max_archive.get(sat, d))
            else:
                lim = df["satellite"].map(max_archive)
                df = df[lim.isna() | (df["acq_date"] > lim)]

            if df.empty:
                continue
            n_out += len(df)
            tabla = pa.Table.from_pandas(df, preserve_index=False)
            base = re.sub(r"\W", "_", os.path.splitext(os.path.basename(f))[0])
            pq.write_to_dataset(
                tabla, root_path=outdir,
                partition_cols=["satellite", "year"],
                basename_template=f"{base}_{i}_{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
            )
        resumen.append((os.path.basename(f), product, n_in, n_out))

    print("\nArchivo | producto | filas leídas | filas escritas")
    for r in resumen:
        print(" | ".join(map(str, r)))
    print(f"\nDataset Parquet en: {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True,
                    help="CSV(s) o patrón, ej. data/raw/*.csv")
    ap.add_argument("--output", default="data/parquet")
    a = ap.parse_args()
    procesar(a.input, a.output)
