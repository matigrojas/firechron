"""
Armonización MODIS <-> VIIRS de focos FIRMS.

Entrada : dataset Parquet generado por limpieza_firms.py
Salidas : (en --output)
  celda_dia/        una fila por celda 1 km x día x satélite (sin armonizar)
  armonizado/       serie armonizada lista para la app (columna `peso`)
  calibracion.csv   factores por región de calibración
  validacion.png    MODIS vs VIIRS mensual, antes y después de calibrar
  metricas.txt      métricas de validación (período independiente)

Uso:
    python armonizar_firms.py --input data/parquet --output data/armonizado_out

Idea:
  1. Filtrar: confianza nominal/high, sólo incendios de vegetación (type 0 o
     NRT sin type) y sacar celdas de fuentes fijas (type 1, 2 o 3 en el archive).
  2. Unidad común: "celda-día" = celda de ~1 km con al menos un foco ese día.
     Así VIIRS (375 m) no cuenta muchas veces el mismo fuego.
  3. Escala de referencia: VIIRS S-NPP (es la que continúa a futuro).
  4. Factores por región (bloques de REG_DEG grados), con el solapamiento:
       k_modis  = celdas-día S-NPP / celdas-día MODIS (Terra+Aqua)
       k_terra  = celdas-día S-NPP / celdas-día sólo Terra (para 2000-2002,
                  cuando Aqua todavía no volaba)
       k_n20    = celdas-día S-NPP / celdas-día NOAA-20 (para después del
                  fin de S-NPP)
     Si una región tiene pocos datos, se usa el factor global.
  5. Serie final:
       antes de VIIRS_START          -> MODIS (o Terra) x k
       VIIRS_START .. fin de S-NPP   -> S-NPP, peso 1
       después del fin de S-NPP      -> NOAA-20 x k_n20
     Contar focos armonizados = SUM(peso).
"""

import argparse
import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------- parámetros ----------------
REG_DEG = 1.0                  # tamaño de la región de calibración (grados)
CELL_DEG = 0.01                # tiene que coincidir con GRID_DEG de la limpieza
AQUA_START = "2002-07-04"
VIIRS_START = "2012-01-20"
CAL = ("2012-02-01", "2017-12-31")   # período para calcular factores
VAL = ("2018-01-01", "2020-12-31")   # período para validar (independiente)
# Hasta 2020: después Terra y Aqua empiezan a derivar en su hora de pasada.
N20_OVERLAP_START = "2018-04-01"
MIN_EVENTOS = 200              # mínimo de celdas-día para confiar en una región
# --------------------------------------------


def main(inp, out):
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    # Límite de memoria: si no alcanza, DuckDB usa disco temporal
    con.execute(f"SET memory_limit='{os.environ.get('DUCKDB_MEM', '2GB')}'")
    con.execute(f"SET temp_directory='{os.environ.get('DUCKDB_TMP', '.duckdb_tmp')}'")
    f = REG_DEG / CELL_DEG

    con.execute(f"""
        CREATE VIEW raw AS
        SELECT * FROM read_parquet('{inp}/**/*.parquet', hive_partitioning=true)
    """)

    # 1. Celdas de fuentes fijas (volcanes, industria, offshore)
    con.execute("""
        CREATE TABLE fijas AS
        SELECT DISTINCT cell_r, cell_c FROM raw WHERE type IN (1, 2, 3)
    """)

    # 2. Celda-día por satélite, ya filtrado
    con.execute(f"""
        CREATE TABLE celda_dia AS
        SELECT r.cell_r, r.cell_c, r.acq_date, r.satellite,
               floor(r.cell_r / {f})::INT AS reg_r,
               floor(r.cell_c / {f})::INT AS reg_c,
               count(*) AS n_focos
        FROM raw r
        ANTI JOIN fijas x ON r.cell_r = x.cell_r AND r.cell_c = x.cell_c
        WHERE r.conf_class IN ('nominal', 'high')
          AND (r.type IS NULL OR r.type = 0)
          AND r.satellite IN ('Terra', 'Aqua', 'SNPP', 'NOAA20')
        GROUP BY ALL
    """)

    # MODIS combinado: celda-día con detección de Terra o Aqua (sin duplicar)
    con.execute("""
        CREATE VIEW modis_cd AS
        SELECT cell_r, cell_c, acq_date, reg_r, reg_c,
               bool_or(satellite = 'Terra') AS tiene_terra
        FROM celda_dia WHERE satellite IN ('Terra', 'Aqua')
        GROUP BY ALL
    """)

    snpp_fin = con.execute(
        "SELECT max(acq_date) FROM celda_dia WHERE satellite = 'SNPP'"
    ).fetchone()[0]
    print(f"Último día con S-NPP: {snpp_fin}")

    # 3. Factores por región
    def factor(nombre, sql_ref, sql_otro, d0, d1):
        con.execute(f"""
            CREATE TABLE {nombre} AS
            WITH a AS (SELECT reg_r, reg_c, count(*) n FROM ({sql_ref})
                       WHERE acq_date BETWEEN '{d0}' AND '{d1}' GROUP BY ALL),
                 b AS (SELECT reg_r, reg_c, count(*) n FROM ({sql_otro})
                       WHERE acq_date BETWEEN '{d0}' AND '{d1}' GROUP BY ALL)
            SELECT coalesce(a.reg_r, b.reg_r) reg_r,
                   coalesce(a.reg_c, b.reg_c) reg_c,
                   coalesce(a.n, 0) n_ref, coalesce(b.n, 0) n_otro
            FROM a FULL JOIN b USING (reg_r, reg_c)
        """)
        glob = con.execute(
            f"SELECT sum(n_ref)::DOUBLE / nullif(sum(n_otro), 0) FROM {nombre}"
        ).fetchone()[0] or 1.0
        con.execute(f"""
            ALTER TABLE {nombre} ADD COLUMN k DOUBLE;
            UPDATE {nombre} SET k = CASE
                WHEN n_ref >= {MIN_EVENTOS} AND n_otro >= {MIN_EVENTOS}
                THEN n_ref::DOUBLE / n_otro ELSE {glob} END;
        """)
        return glob

    snpp = "SELECT * FROM celda_dia WHERE satellite = 'SNPP'"
    g_modis = factor("k_modis", snpp, "SELECT * FROM modis_cd", *CAL)
    # Terra sola: misma comparación pero contando sólo celdas-día con Terra
    g_terra = factor("k_terra", snpp,
                     "SELECT * FROM modis_cd WHERE tiene_terra", *CAL)
    g_n20 = factor("k_n20", snpp,
                   "SELECT * FROM celda_dia WHERE satellite = 'NOAA20'",
                   N20_OVERLAP_START, str(snpp_fin))
    print(f"Factores globales: MODIS->VIIRS {g_modis:.2f} | "
          f"Terra->VIIRS {g_terra:.2f} | NOAA20->SNPP {g_n20:.2f}")

    con.execute(f"""
        COPY (
          SELECT m.reg_r * {REG_DEG} AS lat_min, m.reg_c * {REG_DEG} AS lon_min,
                 m.n_ref AS snpp_cal, m.n_otro AS modis_cal, m.k AS k_modis,
                 t.k AS k_terra, n.k AS k_n20
          FROM k_modis m
          LEFT JOIN k_terra t USING (reg_r, reg_c)
          LEFT JOIN k_n20 n USING (reg_r, reg_c)
          ORDER BY 1, 2
        ) TO '{out}/calibracion.csv' (HEADER)
    """)

    # 4. Serie armonizada
    con.execute(f"""
        CREATE TABLE armonizado AS
        -- MODIS antes de VIIRS
        SELECT c.cell_r, c.cell_c, c.acq_date,
               CASE WHEN c.acq_date < '{AQUA_START}' THEN 'Terra' ELSE 'MODIS' END AS fuente,
               CASE WHEN c.acq_date < '{AQUA_START}'
                    THEN coalesce(t.k, {g_terra}) ELSE coalesce(m.k, {g_modis}) END AS peso
        FROM modis_cd c
        LEFT JOIN k_modis m USING (reg_r, reg_c)
        LEFT JOIN k_terra t USING (reg_r, reg_c)
        WHERE c.acq_date < '{VIIRS_START}'
        UNION ALL
        -- S-NPP mientras exista
        SELECT cell_r, cell_c, acq_date, 'SNPP', 1.0
        FROM celda_dia
        WHERE satellite = 'SNPP' AND acq_date >= '{VIIRS_START}'
        UNION ALL
        -- NOAA-20 después del fin de S-NPP
        SELECT c.cell_r, c.cell_c, c.acq_date, 'NOAA20', coalesce(n.k, {g_n20})
        FROM celda_dia c LEFT JOIN k_n20 n USING (reg_r, reg_c)
        WHERE c.satellite = 'NOAA20' AND c.acq_date > '{snpp_fin}'
    """)

    con.execute(f"""
        COPY (
          SELECT cell_r, cell_c,
                 (cell_r + 0.5) * {CELL_DEG} AS lat, (cell_c + 0.5) * {CELL_DEG} AS lon,
                 acq_date, year(acq_date) AS year, month(acq_date) AS month,
                 fuente, peso
          FROM armonizado
        ) TO '{out}/armonizado' (FORMAT parquet, PARTITION_BY (year), OVERWRITE_OR_IGNORE)
    """)
    con.execute(f"""
        COPY (SELECT *, year(acq_date) AS year FROM celda_dia)
        TO '{out}/celda_dia' (FORMAT parquet, PARTITION_BY (year), OVERWRITE_OR_IGNORE)
    """)

    # 5. Validación en período independiente (VAL), mensual por región
    df = con.execute(f"""
        WITH s AS (SELECT reg_r, reg_c, date_trunc('month', acq_date) mes, count(*) n
                   FROM celda_dia WHERE satellite = 'SNPP'
                   AND acq_date BETWEEN '{VAL[0]}' AND '{VAL[1]}' GROUP BY ALL),
             m AS (SELECT reg_r, reg_c, date_trunc('month', acq_date) mes, count(*) n
                   FROM modis_cd
                   WHERE acq_date BETWEEN '{VAL[0]}' AND '{VAL[1]}' GROUP BY ALL)
        SELECT coalesce(s.n, 0) AS viirs, coalesce(m.n, 0) AS modis,
               coalesce(m.n, 0) * coalesce(k.k, {g_modis}) AS modis_arm
        FROM s FULL JOIN m USING (reg_r, reg_c, mes)
        LEFT JOIN k_modis k USING (reg_r, reg_c)
    """).df()

    def metr(obs, est):
        obs, est = np.log1p(obs), np.log1p(est)
        r2 = 1 - np.sum((obs - est) ** 2) / np.sum((obs - obs.mean()) ** 2)
        return r2, np.mean(est - obs)

    txt = []
    if len(df):
        r2a, ba = metr(df.viirs, df.modis)
        r2d, bd = metr(df.viirs, df.modis_arm)
        tot = df[["viirs", "modis", "modis_arm"]].sum()
        txt = [
            f"Validación {VAL[0]} a {VAL[1]} ({len(df)} región-mes)",
            f"Totales: VIIRS {tot.viirs:.0f} | MODIS crudo {tot.modis:.0f} | "
            f"MODIS armonizado {tot.modis_arm:.0f}",
            f"R² (log): antes {r2a:.3f} | después {r2d:.3f}",
            f"Sesgo medio (log): antes {ba:+.3f} | después {bd:+.3f}",
        ]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for a, col, tit in [(ax[0], "modis", "Antes (MODIS crudo)"),
                            (ax[1], "modis_arm", "Después (MODIS armonizado)")]:
            a.scatter(df[col] + 1, df.viirs + 1, s=6, alpha=0.4)
            lim = [1, max(df.viirs.max(), df[col].max()) + 1]
            a.plot(lim, lim, "k--", lw=1)
            a.set(xscale="log", yscale="log", title=tit,
                  xlabel="MODIS (celdas-día + 1)", ylabel="VIIRS S-NPP (celdas-día + 1)")
        fig.suptitle(f"Validación mensual por región, {VAL[0][:4]}–{VAL[1][:4]}")
        fig.tight_layout()
        fig.savefig(f"{out}/validacion.png", dpi=150)

    with open(f"{out}/metricas.txt", "w") as fh:
        fh.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    print(f"\nListo. Salidas en {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/parquet")
    ap.add_argument("--output", default="data/salida")
    a = ap.parse_args()
    main(a.input, a.output)
