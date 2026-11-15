"""
Pasa los NetCDF de ERA5-Land (0,1°) a la grilla del modelo (0,5°) y a una tabla mensual.

Salida: parquet con una fila por celda de 0,5° y mes:
  r5, c5, year, month,
  t2m_c   temperatura media (°C)
  rh      humedad relativa (%)
  vpd     déficit de presión de vapor (kPa)
  ws      velocidad del viento (m/s)
  pr_mm   precipitación del mes (mm)
  sm      humedad del suelo, capa 0–7 cm (m³/m³)

Uso:
    python pronostico/era5_a_grilla.py --input Sudamerica/ERAS5-monthly --output data/era5_05.parquet
  (--input puede ser una carpeta, se leen todos los .nc adentro, o un archivo .nc)
"""
import argparse
import glob
import math

import numpy as np
import pandas as pd
import xarray as xr

CELL = 0.5


def sat_vp(tc):
    """Presión de vapor de saturación (kPa), Tetens."""
    return 0.6108 * np.exp(17.27 * tc / (tc + 237.3))


def procesar_archivo(path):
    ds = xr.open_dataset(path)
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    lat, lon = ds["latitude"].values, ds["longitude"].values
    r = np.floor(lat / CELL).astype(int)
    c = np.floor(lon / CELL).astype(int)
    R, C = np.meshgrid(r, c, indexing="ij")
    r0, c0 = R.min(), C.min()
    nc = C.max() - c0 + 1
    key = ((R - r0) * nc + (C - c0)).ravel()
    nkey = key.max() + 1

    filas = []
    tiempos = pd.to_datetime(ds[tdim].values)
    for i, t in enumerate(tiempos):
        if i % 24 == 0:
            print(f"  {t:%Y-%m} …", flush=True)
        g = {v: ds[v].isel({tdim: i}).values for v in ["t2m", "d2m", "tp", "u10", "v10", "swvl1"]}
        tc = g["t2m"] - 273.15
        td = g["d2m"] - 273.15
        es, ea = sat_vp(tc), sat_vp(td)
        dias = pd.Period(t, "M").days_in_month
        campos = {
            "t2m_c": tc,
            "rh": np.clip(100 * ea / es, 0, 100),
            "vpd": np.maximum(es - ea, 0),
            "ws": np.hypot(g["u10"], g["v10"]),
            "pr_mm": g["tp"] * 1000 * dias,  # ERA5-Land mensual: m/día promedio
            "sm": g["swvl1"],
        }
        valid = np.isfinite(tc).ravel()
        k = key[valid]
        n = np.bincount(k, minlength=nkey)
        out = {"r5": (np.arange(nkey) // nc) + r0, "c5": (np.arange(nkey) % nc) + c0}
        for name, arr in campos.items():
            s = np.bincount(k, weights=arr.ravel()[valid], minlength=nkey)
            out[name] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        df = pd.DataFrame(out)
        df = df[n > 0]
        df["year"], df["month"] = t.year, t.month
        filas.append(df)
    return pd.concat(filas)


def main(inp, out):
    files = [inp] if inp.endswith(".nc") else sorted(glob.glob(f"{inp}/**/*.nc", recursive=True))
    if not files:
        raise SystemExit(f"No hay archivos .nc en {inp}")
    print("Archivos:", *files, sep="\n  ")
    df = pd.concat([procesar_archivo(f) for f in files], ignore_index=True)
    # si dos archivos repiten meses, quedarse con uno
    df = df.drop_duplicates(["r5", "c5", "year", "month"], keep="last")
    for col in ["r5", "c5"]:
        df[col] = df[col].astype("int16")
    df.to_parquet(out, index=False)
    print(f"{out}: {len(df):,} filas, {df.year.min()}–{df.year.max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Sudamerica/ERAS5-monthly")
    ap.add_argument("--output", default="data/era5_05.parquet")
    a = ap.parse_args()
    main(a.input, a.output)
