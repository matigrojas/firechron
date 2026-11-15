"""
Red neuronal (perceptrón multicapa, PyTorch) del pronóstico mensual de anomalías de fuego.
La llama entrenar.py en un proceso aparte; también se puede correr sola:

    python pronostico/entrenar_mlp.py --dispositivo cpu

Arquitectura: entradas (variables estandarizadas + indicador de clima faltante)
  → 128 ReLU → Dropout 0,2 → 64 ReLU → Dropout 0,2 → 1 (sigmoide = probabilidad).
Pérdida BCE, AdamW (lr 1e-3, weight decay 1e-4), lotes de 4096, parada temprana por AUC
en validación (2020–2022) con paciencia 5.
Guarda data/pronostico/pred_mlp.parquet (probabilidad para cada fila) y mlp.pt.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

NO_FEAT = {"r5", "c5", "year", "month", "focos", "y", "split"}


def auc(y, p):
    """AUC por rangos (sin scikit-learn, para no mezclar librerías OpenMP)."""
    y = np.asarray(y); r = pd.Series(p).rank().to_numpy()
    n1 = (y == 1).sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main(a):
    out = os.path.dirname(a.tabla)
    df = pd.read_parquet(a.tabla)
    feats = [c for c in df.columns if c not in NO_FEAT]
    ccols = [c for c in feats if c.startswith("c_")]
    tr = df[(df.split == "train") & df[ccols].notna().all(axis=1)]
    va = df[df.split == "val"]
    if a.dispositivo == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = a.dispositivo
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)

    def prep(d):
        x = ((d[feats] - mu) / sd).astype("float32")
        falta = x.isna().any(axis=1).astype("float32").to_numpy()[:, None]
        return np.hstack([x.fillna(0).to_numpy(), falta]).astype("float32")

    Xtr = torch.tensor(prep(tr)); ytr = torch.tensor(tr.y.to_numpy().astype("float32"))
    Xva = torch.tensor(prep(va)).to(dev)
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, pac = -1, None, 0
    print(f"    red en {dev}: {len(tr):,} filas de entrenamiento, {Xtr.shape[1]} entradas", flush=True)
    for ep in range(a.epocas):
        net.train()
        perm = torch.randperm(len(Xtr))
        for k in range(0, len(perm), 4096):
            idx = perm[k:k + 4096]
            opt.zero_grad()
            lossf(net(Xtr[idx].to(dev)).squeeze(1), ytr[idx].to(dev)).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pva = torch.sigmoid(net(Xva).squeeze(1)).cpu().numpy()
        s = auc(va.y.to_numpy(), pva)
        print(f"    época {ep + 1:2d}  AUC val {s:.4f}", flush=True)
        if s > best + 1e-4:
            best, pac = s, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            pac += 1
            if pac >= 5:
                break
    net.load_state_dict(best_state)
    torch.save({"state": best_state, "feats": feats, "mu": mu.to_dict(), "sd": sd.to_dict()},
               os.path.join(out, "mlp.pt"))
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(prep(df)).to(dev)).squeeze(1)).cpu().numpy()
    df[["r5", "c5", "year", "month"]].assign(p_mlp=p).to_parquet(os.path.join(out, "pred_mlp.parquet"), index=False)
    print(f"    red lista (mejor AUC val {best:.4f})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabla", default="data/pronostico/tabla.parquet")
    ap.add_argument("--dispositivo", default="cpu", help="cpu | mps | cuda | auto")
    ap.add_argument("--epocas", type=int, default=40)
    main(ap.parse_args())
