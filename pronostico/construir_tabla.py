"""
Arma la tabla de entrenamiento del pronóstico mensual de anomalías de fuego.

Una fila = una celda de 0,5° × un mes objetivo M. Con anticipación L (--anticipacion, 1 a 3),
todo lo que entra como variable está disponible al terminar el mes M-L: el pronóstico se emite
L meses antes. Ej.: L=3 → al terminar junio se pronostica septiembre.

Objetivo  y = 1 si los focos armonizados de M superan la mediana histórica de ese mes
          en esa celda (mediana calculada SOLO con los años de entrenamiento).

Variables:
  fuego    : focos de M-L, M-L-1, M-L-2 (lag1..lag3 = los 3 meses más recientes disponibles) y M-12
             (mismo mes del año anterior), en log y anomalía; suma de los 8 vecinos en M-L;
             climatología del mes objetivo (mediana, P75, fracción de años con fuego)
  clima    : (si está data/era5_05.parquet) anomalías de M-L de temperatura, humedad relativa,
             déficit de vapor, viento y humedad del suelo; lluvia acumulada 1, 3 y 6 meses
             como cociente logarítmico contra lo normal
  ENSO     : índice ONI (si está data/oni.txt o se puede bajar de NOAA)
  posición : lat, lon, mes (seno/coseno)

Uso:
    python pronostico/construir_tabla.py --anticipacion 3
    (la salida por defecto es data/pronostico/L3/tabla.parquet)
    (opciones: --fuego app/data/app_data.parquet --clima data/era5_05.parquet
               --oni data/oni.txt --salida data/pronostico/tabla.parquet)
"""
import argparse
import datetime as dt
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN = (2003, 2019)          # años para climatologías y entrenamiento
START = (2002, 7)             # desde que vuelan Terra + Aqua
MIN_FOCOS_TRAIN = 50          # celda "activa": al menos esto en 2003–2019
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
CLIMA_VARS = ["t2m_c", "rh", "vpd", "ws", "sm"]


def month_index(y, m):
    return (y - START[0]) * 12 + (m - START[1])


def leer_fuego(path):
    t = pq.read_table(path, columns=["g_r", "g_c", "date", "focos"])
    days = t.column("date").cast(pa.int32()).to_numpy()
    d64 = days.astype("datetime64[D]")
    m64 = d64.astype("datetime64[M]")
    y = m64.astype("datetime64[Y]").astype(int) + 1970
    m = m64.astype(int) % 12 + 1
    last = d64.max()
    # último mes completo: el del último día si ese día cierra el mes; si no, el anterior
    mes_cerrado = (last + np.timedelta64(1, "D")).astype("datetime64[M]") > last.astype("datetime64[M]")
    last_full = last.astype("datetime64[M]") - np.timedelta64(0 if mes_cerrado else 1, "M")
    df = pd.DataFrame({"r5": np.floor(t.column("g_r").to_numpy() / 5).astype(int),
                       "c5": np.floor(t.column("g_c").to_numpy() / 5).astype(int),
                       "t": month_index(y, m), "focos": t.column("focos").to_numpy()})
    df = df[df.t >= 0].groupby(["r5", "c5", "t"], as_index=False).focos.sum()
    ly = int(str(last_full)[:4]); lm = int(str(last_full)[5:7])
    return df, month_index(ly, lm)


def leer_oni(path):
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            urllib.request.urlretrieve(ONI_URL, path)
            print(f"ONI descargado de NOAA → {path}")
        except Exception as e:  # noqa: BLE001
            print(f"(Sin ONI: no se pudo bajar {ONI_URL}: {e})")
            return None
    seas = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
            "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 4 and p[0] in seas:
                try:
                    out[(int(p[1]), seas[p[0]])] = float(p[3])  # (año, mes central) → anomalía
                except ValueError:
                    pass
    return out


def main(a):
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    fire, T_last = leer_fuego(a.fuego)
    T = T_last + 1                     # meses completos 0..T_last
    print(f"Fuego: {START[0]}-{START[1]:02d} → mes {T_last} (último completo); celdas con fuego: "
          f"{fire.groupby(['r5', 'c5']).ngroups:,}")

    # grilla densa [t, i, j]
    r0, r1 = fire.r5.min() - 1, fire.r5.max() + 1
    c0, c1 = fire.c5.min() - 1, fire.c5.max() + 1
    NR, NC = r1 - r0 + 1, c1 - c0 + 1
    L = a.anticipacion
    F = np.zeros((T + L, NR, NC), np.float32)          # +L: meses futuros (sin dato)
    ff = fire[fire.t < T]
    F[ff.t.values, ff.r5.values - r0, ff.c5.values - c0] = ff.focos.values
    ts = np.arange(T + L)
    years = START[0] + (ts + START[1] - 1) // 12
    months = (ts + START[1] - 1) % 12 + 1
    trn = (years >= TRAIN[0]) & (years <= TRAIN[1])

    # climatologías del fuego (sólo años de entrenamiento)
    med = np.zeros((13, NR, NC), np.float32); p75 = med.copy(); frac = med.copy()
    for m in range(1, 13):
        sel = trn & (months == m)
        med[m] = np.median(F[sel], axis=0)
        p75[m] = np.percentile(F[sel], 75, axis=0)
        frac[m] = (F[sel] > 0).mean(axis=0)
    activa = F[trn].sum(axis=0) >= MIN_FOCOS_TRAIN
    ii, jj = np.nonzero(activa)
    print(f"Celdas activas (≥{MIN_FOCOS_TRAIN} focos en {TRAIN[0]}–{TRAIN[1]}): {len(ii):,}")

    LF = np.log1p(F)
    Lmed = np.log1p(med)
    ANOM = LF - Lmed[months]                            # anomalía log de cada mes
    # suma de los 8 vecinos
    P = np.pad(F, ((0, 0), (1, 1), (1, 1)))
    VEC = sum(P[:, 1 + di:1 + di + NR, 1 + dj:1 + dj + NC] for di in (-1, 0, 1) for dj in (-1, 0, 1)
              if (di, dj) != (0, 0))

    # clima
    W = {}
    if a.clima and os.path.exists(a.clima):
        cl = pd.read_parquet(a.clima)
        cl["t"] = month_index(cl.year, cl.month)
        cl = cl[(cl.t >= 0) & (cl.t < T + L) & cl.r5.between(r0, r1) & cl.c5.between(c0, c1)]
        for v in CLIMA_VARS + ["pr_mm"]:
            arr = np.full((T + L, NR, NC), np.nan, np.float32)
            arr[cl.t.values, cl.r5.values - r0, cl.c5.values - c0] = cl[v].values
            W[v] = arr
        # lluvia acumulada 1, 3, 6 meses
        pr = W.pop("pr_mm")
        for k in (1, 3, 6):
            acc = np.full_like(pr, np.nan)
            for t in range(k - 1, T + L):
                acc[t] = pr[t - k + 1:t + 1].sum(axis=0)
            W[f"pr{k}"] = acc
        n_meses = int(np.isfinite(W["t2m_c"]).any(axis=(1, 2)).sum())
        print(f"Clima ERA5-Land: {n_meses} meses con datos")
    else:
        print("(Sin clima: no encontré", a.clima, ")")
    # anomalías del clima contra su climatología de entrenamiento
    WA = {}
    for v, arr in W.items():
        clim = np.full((13, NR, NC), np.nan, np.float32)
        for m in range(1, 13):
            sel = trn & (months == m)
            clim[m] = np.nanmean(arr[sel], axis=0)
        if v.startswith("pr"):
            WA[v] = np.log((arr + 1) / (clim[months] + 1))   # cociente log vs. lo normal
        else:
            WA[v] = arr - clim[months]

    oni = leer_oni(a.oni) if a.oni else None

    # filas: meses objetivo desde START+12 hasta el mes a pronosticar (T+L-1)
    filas = []
    for t in range(max(12, L + 2), T + L):
        if T <= t < T + L - 1:
            continue          # meses intermedios: no hay datos de M-L todavía para la anticipación pedida
        y, m = int(years[t]), int(months[t])
        d = {
            "r5": ii + r0, "c5": jj + c0, "year": np.full(len(ii), y), "month": np.full(len(ii), m),
            "lat": (ii + r0 + 0.5) * 0.5, "lon": (jj + c0 + 0.5) * 0.5,
            "mes_sin": np.full(len(ii), np.sin(2 * np.pi * m / 12)),
            "mes_cos": np.full(len(ii), np.cos(2 * np.pi * m / 12)),
            "clim_med": Lmed[m][ii, jj], "clim_p75": np.log1p(p75[m][ii, jj]), "clim_frac": frac[m][ii, jj],
            "lag1": LF[t - L][ii, jj], "lag2": LF[t - L - 1][ii, jj], "lag3": LF[t - L - 2][ii, jj],
            "lag12": LF[t - 12][ii, jj],
            "anom1": ANOM[t - L][ii, jj], "anom2": ANOM[t - L - 1][ii, jj], "anom3": ANOM[t - L - 2][ii, jj],
            "anom12": ANOM[t - 12][ii, jj],
            "vecinos1": np.log1p(VEC[t - L][ii, jj]),
        }
        d["anom_1a3"] = (d["anom1"] + d["anom2"] + d["anom3"]) / 3
        for v, arr in WA.items():
            d[f"c_{v}"] = arr[t - L][ii, jj]
        if oni is not None:
            # ONI de la estación centrada en M-L-1 (ya publicada al terminar M-L)
            tt = t - L - 1
            d["oni"] = np.full(len(ii), oni.get((int(years[tt]), int(months[tt])), np.nan))
        if t < T:
            focos = F[t][ii, jj]
            d["focos"] = focos
            d["y"] = (focos > med[m][ii, jj]).astype(np.int8)
        else:
            d["focos"] = np.full(len(ii), np.nan)
            d["y"] = np.full(len(ii), -1, np.int8)        # mes a pronosticar
        filas.append(pd.DataFrame(d))
    df = pd.concat(filas, ignore_index=True)
    df["split"] = np.select([df.y < 0, df.year <= TRAIN[1], df.year <= 2022, df.year <= 2025],
                            ["pronostico", "train", "val", "test"], "extra")
    df.to_parquet(a.salida, index=False)
    print(f"\nTabla: {a.salida} — {len(df):,} filas, {df.shape[1]} columnas")
    print(df.groupby("split").agg(filas=("y", "size"), positivos=("y", lambda s: (s == 1).mean())).round(3))
    fy, fm = int(years[T + L - 1]), int(months[T + L - 1])
    ly, lm = int(years[T - 1]), int(months[T - 1])
    print(f"Anticipación: {L} mes(es). Con datos hasta {ly}-{lm:02d} se pronostica {fy}-{fm:02d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuego", default="app/data/app_data.parquet")
    ap.add_argument("--clima", default="data/era5_05.parquet")
    ap.add_argument("--oni", default="data/oni.txt")
    ap.add_argument("--anticipacion", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    if a.salida is None:
        a.salida = f"data/pronostico/L{a.anticipacion}/tabla.parquet"
    main(a)
