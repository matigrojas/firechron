"""
Genera el archivo liviano que usa la app a partir de armonizado.parquet.

Agrega las celdas-día de 1 km a una grilla de 0,1° (~11 km) por día:
  g_r, g_c : índice de celda de 0,1°  (lat ≈ (g_r + 0.5) / 10)
  date     : fecha (UTC)
  focos    : suma de `peso` = focos armonizados (escala VIIRS S-NPP)
  crudo    : celdas-día sin armonizar (para mostrar el "antes")

Uso:
    python app/preparar_datos_app.py --input data/salida/armonizado.parquet \
                                     --output app/data/app_data.parquet
"""
import argparse
import os

import duckdb


def main(inp, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          SELECT CAST(floor(cell_r / 10) AS SMALLINT) AS g_r,
                 CAST(floor(cell_c / 10) AS SMALLINT) AS g_c,
                 acq_date AS date,
                 CAST(sum(peso) AS FLOAT) AS focos,
                 CAST(count(*) AS INTEGER) AS crudo
          FROM read_parquet('{inp}')
          GROUP BY ALL
          ORDER BY g_r, g_c, date
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 200000)
    """)
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    print(f"{out}: {n:,} filas, {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/salida/armonizado.parquet")
    ap.add_argument("--output", default="app/data/app_data.parquet")
    a = ap.parse_args()
    main(a.input, a.output)
