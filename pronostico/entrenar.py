"""
Entrena y evalúa el pronóstico mensual "¿el mes que viene tendrá más fuego de lo normal?".

Modelos (todos dan una probabilidad):
  clim        referencia 1: frecuencia histórica de "por encima de la mediana" en esa celda y mes
  pers        referencia 2: persistencia (¿el mes anterior estuvo por encima de lo normal?)
  lgb_fuego   LightGBM sólo con variables de fuego
  lgb_clima   LightGBM con fuego + clima ERA5-Land + ENSO
  mlp         red neuronal (PyTorch) con las mismas variables que lgb_clima

Separación temporal: entrenamiento 2003–2019 · validación 2020–2022 · test 2023–2025.
Métricas: AUC, Brier y Brier Skill Score contra la climatología (BSS > 0 = mejor que la referencia).

Uso:
    python pronostico/entrenar.py            (lee data/pronostico/tabla.parquet)
    python pronostico/entrenar.py --sin-mlp  (sólo referencias y LightGBM)
"""
import argparse
import json
import os
import time

import subprocess
import sys

# La red (PyTorch) se entrena en un proceso aparte (entrenar_mlp.py): en Mac, LightGBM y
# PyTorch traen cada uno su propia librería OpenMP y juntos en un mismo proceso se cuelgan
# o dan "segmentation fault".
import lightgbm as lgb  # noqa: E402
import matplotlib
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NO_FEAT = {"r5", "c5", "year", "month", "focos", "y", "split"}


def metricas(y, p, p_ref):
    b = brier_score_loss(y, p)
    return {"AUC": roc_auc_score(y, p), "Brier": b, "BSS": 1 - b / brier_score_loss(y, p_ref),
            "Acierto": float(((p >= 0.5) == y).mean())}


def entrenar_lgb(tr, va, feats, nombre, out):
    params = dict(objective="binary", metric=["auc", "binary_logloss"], learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=42)
    dtr = lgb.Dataset(tr[feats], tr.y)
    dva = lgb.Dataset(va[feats], va.y, reference=dtr)
    t0 = time.time()
    m = lgb.train(params, dtr, 3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False), lgb.log_evaluation(0)])
    print(f"  {nombre}: {m.best_iteration} árboles, {time.time() - t0:.0f} s")
    m.save_model(os.path.join(out, f"{nombre}.txt"))
    return m


def main(a):
    out = os.path.dirname(a.tabla)
    df = pd.read_parquet(a.tabla)
    feats_all = [c for c in df.columns if c not in NO_FEAT]
    feats_fuego = [c for c in feats_all if not c.startswith("c_") and c != "oni"]
    hay_clima = any(c.startswith("c_") for c in feats_all)
    print(f"Variables: {len(feats_fuego)} de fuego, {len(feats_all) - len(feats_fuego)} de clima/ENSO")

    tr, va = df[df.split == "train"], df[df.split == "val"]
    # referencias
    clim = tr.groupby(["r5", "c5", "month"]).y.mean().rename("p_clim")
    pers = tr.assign(arriba=tr.anom1 > 0).groupby(["month", "arriba"]).y.mean().rename("p_pers")
    df = df.join(clim, on=["r5", "c5", "month"])
    df["p_clim"] = df.p_clim.fillna(tr.y.mean())
    df = df.assign(arriba=df.anom1 > 0).join(pers, on=["month", "arriba"]).drop(columns="arriba")

    print("Entrenando…")
    m_f = entrenar_lgb(tr, va, feats_fuego, "lgb_fuego", out)
    df["p_lgb_fuego"] = m_f.predict(df[feats_fuego], num_iteration=m_f.best_iteration)
    modelos = ["p_clim", "p_pers", "p_lgb_fuego"]
    if hay_clima:
        tr_c = tr[tr[[c for c in feats_all if c.startswith("c_")]].notna().all(axis=1)]
        m_c = entrenar_lgb(tr_c, va, feats_all, "lgb_clima", out)
        df["p_lgb_clima"] = m_c.predict(df[feats_all], num_iteration=m_c.best_iteration)
        modelos.append("p_lgb_clima")
        imp = pd.Series(m_c.feature_importance("gain"), index=feats_all).sort_values(ascending=False)
        if not a.sin_mlp:
            print("  mlp: entrenando en un proceso aparte (entrenar_mlp.py)…", flush=True)
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entrenar_mlp.py")
            r = subprocess.run([sys.executable, script, "--tabla", a.tabla, "--dispositivo", a.dispositivo])
            pm = os.path.join(out, "pred_mlp.parquet")
            if r.returncode == 0 and os.path.exists(pm):
                df = df.merge(pd.read_parquet(pm), on=["r5", "c5", "year", "month"], how="left")
                modelos.append("p_mlp")
            else:
                print("  (la red falló; sigo sin ella)")
    else:
        imp = pd.Series(m_f.feature_importance("gain"), index=feats_fuego).sort_values(ascending=False)
    (imp / imp.sum()).rename("importancia").to_csv(os.path.join(out, "importancias.csv"))

    # métricas
    filas = []
    ccols = [c for c in feats_all if c.startswith("c_")]
    for sp in ["val", "test", "extra"]:
        d = df[df.split == sp]
        if d.empty:
            continue
        # 2026: los modelos con clima se evalúan sólo en los meses que ya tienen datos de ERA5
        d_clima = d[d[ccols].notna().all(axis=1)] if ccols else d
        for mo in modelos:
            if mo in ("p_lgb_clima", "p_mlp"):
                if len(d_clima) < 0.9 * len(d):
                    continue
                dd = d_clima
            else:
                dd = d
            filas.append({"split": sp, "modelo": mo[2:], **metricas(dd.y, dd[mo], dd.p_clim)})
    met = pd.DataFrame(filas)
    met.to_csv(os.path.join(out, "metricas.csv"), index=False)
    print("\nMétricas (val 2020–22, test 2023–25, extra = 2026, el test más reciente):")
    print(met.round(3).to_string(index=False))

    # pronóstico del próximo mes: con clima si hay datos del mes anterior, si no sólo fuego
    fc = df[df.split == "pronostico"].copy()
    # se usa el LightGBM que mejor anduvo en validación (si hay clima para el mes de emisión)
    usa_clima = hay_clima and fc[[c for c in feats_all if c.startswith("c_")]].notna().all(axis=1).mean() > 0.9
    mv = met[met.split == "val"].set_index("modelo").AUC
    if usa_clima and mv.get("lgb_clima", 0) > mv.get("lgb_fuego", 0):
        fc["prob"], fc["modelo"] = fc.p_lgb_clima, "lgb_clima"
    else:
        fc["prob"], fc["modelo"] = fc.p_lgb_fuego, "lgb_fuego"
    cols = ["r5", "c5", "year", "month", "lat", "lon", "split", "y", "focos"] + modelos
    df[df.split.isin(["val", "test", "extra"])][cols].to_parquet(os.path.join(out, "predicciones.parquet"), index=False)
    fc[["r5", "c5", "year", "month", "lat", "lon", "prob", "modelo", "p_clim"]].to_parquet(
        os.path.join(out, "pronostico.parquet"), index=False)
    fy, fm = int(fc.year.iloc[0]), int(fc.month.iloc[0])
    json.dump({"mes": f"{fy}-{fm:02d}", "modelo": fc.modelo.iloc[0]}, open(os.path.join(out, "pronostico.json"), "w"))
    print(f"\nPronóstico {fy}-{fm:02d} ({fc.modelo.iloc[0]}): {len(fc):,} celdas, "
          f"{(fc.prob >= 0.5).mean():.0%} con prob ≥ 50 %")

    # figuras
    t = met[met.split == "test"].set_index("modelo")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    t.AUC.plot.bar(ax=ax[0], color="#3B7DD8"); ax[0].set_ylim(0.5, 1); ax[0].set_title("AUC (test 2023–2025)")
    t.BSS.plot.bar(ax=ax[1], color="#E07B39"); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Brier Skill Score vs. climatología")
    for a_ in ax:
        a_.set_xlabel(""); a_.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(out, "metricas.png"), dpi=150)

    top = (imp / imp.sum()).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    top.plot.barh(ax=ax, color="#3B7DD8"); ax.set_title("Importancia de variables (LightGBM)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "importancias.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 7))
    sc = ax.scatter(fc.lon, fc.lat, c=fc.prob, s=6, cmap="RdBu_r", vmin=0, vmax=1, marker="s")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Prob. de fuego por encima de lo normal")
    ax.set_aspect("equal"); ax.set_title(f"Pronóstico {fy}-{fm:02d}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pronostico.png"), dpi=150)
    print(f"Figuras y resultados en {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabla", default="data/pronostico/tabla.parquet")
    ap.add_argument("--sin-mlp", action="store_true")
    ap.add_argument("--dispositivo", default="cpu", help="cpu | mps | cuda | auto (para la red)")
    main(ap.parse_args())
