"""
Descarga ERA5-Land mensual (Copernicus CDS) para Sudamérica, un archivo por año.

Requisitos (una sola vez):
  1. Cuenta en https://cds.climate.copernicus.eu y aceptar la licencia del dataset
     "ERA5-Land monthly averaged data from 1950 to present" (pestaña Download, al pie).
  2. Archivo ~/.cdsapirc con:
        url: https://cds.climate.copernicus.eu/api
        key: TU-API-KEY
  3. pip install cdsapi

Uso (se puede cortar y relanzar: saltea los años ya bajados):
    python pronostico/descargar_era5.py --out data/era5 --desde 2002 --hasta 2026
"""
import argparse
import datetime as dt
import os
import time

import cdsapi

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "volumetric_soil_water_layer_1",
]
AREA = [13, -82, -56, -34]  # N, O, S, E  (Sudamérica)


def bajar(c, year, months, target):
    c.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{m:02d}" for m in months],
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": AREA,
        },
        target,
    )


def main(out, desde, hasta):
    os.makedirs(out, exist_ok=True)
    c = cdsapi.Client()
    hoy = dt.date.today()
    for year in range(desde, hasta + 1):
        target = os.path.join(out, f"era5land_{year}.nc")
        if os.path.exists(target) and os.path.getsize(target) > 0:
            print(f"{year}: ya existe, salteo")
            continue
        # año en curso: hasta el mes anterior (los últimos meses pueden no estar publicados aún)
        last = 12 if year < hoy.year else max(1, hoy.month - 1)
        for intento in range(3):
            try:
                print(f"{year}: pidiendo meses 1–{last} …", flush=True)
                bajar(c, year, range(1, last + 1), target + ".part")
                os.replace(target + ".part", target)
                print(f"{year}: OK ({os.path.getsize(target) / 1e6:.0f} MB)", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"{year}: error ({e}); reintento", flush=True)
                if year == hoy.year and last > 1:
                    last -= 1  # quizás el último mes todavía no está publicado
                time.sleep(30)
        else:
            print(f"{year}: falló 3 veces, sigo con el próximo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/era5")
    ap.add_argument("--desde", type=int, default=2002)
    ap.add_argument("--hasta", type=int, default=dt.date.today().year)
    a = ap.parse_args()
    main(a.out, a.desde, a.hasta)
