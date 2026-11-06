"""Phase 4 - PyTorch Autoencoder (unsupervised anomaly detection). Runs on CPU, GPU optional.

Idea: train a network to rebuild NORMAL sensor patterns. Normal rows are rebuilt with small
error; abnormal rows (patterns it never learned) are rebuilt badly -> large reconstruction error
= anomaly score.

Uses the SAME processed files / split / evaluation code as Isolation Forest and Random Forest:
    python -m src.train_autoencoder            # full run (6 configurations)
    python -m src.train_autoencoder --quick    # 2 configs, 5 epochs (smoke test)

Outputs (all named autoencoder_*):  models/autoencoder_best.pt, models/autoencoder_config.json,
                                    reports/phase3/autoencoder_*_results.json, figures/*
"""
import argparse
import copy
import json
import logging
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config as C
from src.evaluate import FIG_DIR, evaluate_model, load_experiment_data, save_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ae")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurations compared (chosen by VALIDATION PR-AUC). They vary hidden sizes, latent size,
# learning rate, batch size, optimiser and loss - the knobs the project brief asks to explore.
CONFIGS = [
    {"name": "small_latent8",  "hidden": [32],     "latent": 8,  "lr": 1e-3, "batch": 256, "optimizer": "adam",  "loss": "mse"},
    {"name": "mid_latent8",    "hidden": [64, 32], "latent": 8,  "lr": 1e-3, "batch": 256, "optimizer": "adam",  "loss": "mse"},
    {"name": "mid_latent4",    "hidden": [64, 32], "latent": 4,  "lr": 1e-3, "batch": 256, "optimizer": "adam",  "loss": "mse"},
    {"name": "mid_latent16",   "hidden": [64, 32], "latent": 16, "lr": 1e-3, "batch": 256, "optimizer": "adam",  "loss": "mse"},
    {"name": "mid_huber",      "hidden": [64, 32], "latent": 8,  "lr": 1e-3, "batch": 256, "optimizer": "adam",  "loss": "huber"},
    {"name": "mid_adamw_b128", "hidden": [64, 32], "latent": 8,  "lr": 5e-4, "batch": 128, "optimizer": "adamw", "loss": "mse"},
]
MAX_EPOCHS, PATIENCE = 60, 8


def set_seed(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class AutoEncoder(nn.Module):
    """input -> encoder (hidden layers) -> latent bottleneck -> decoder (mirror) -> reconstruction."""
    def __init__(self, n_in, hidden, latent):
        super().__init__()
        enc, d = [], n_in
        for h in hidden:
            enc += [nn.Linear(d, h), nn.ReLU()]; d = h
        enc += [nn.Linear(d, latent)]
        dec, d = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(d, h), nn.ReLU()]; d = h
        dec += [nn.Linear(d, n_in)]
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_loss(name):
    return nn.MSELoss() if name == "mse" else nn.HuberLoss(delta=1.0)


def make_optim(name, params, lr):
    return (torch.optim.Adam if name == "adam" else torch.optim.AdamW)(params, lr=lr)


@torch.no_grad()
def recon_error(model, X: np.ndarray, batch=4096):
    """Per-row mean squared reconstruction error (ALWAYS MSE for scoring, whatever the training loss)."""
    model.eval(); out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(DEVICE)
        out.append(((model(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


def train_one(cfg, Xtr_n, Xva_n, n_in, max_epochs):
    """Train on normal train rows; early-stop on loss over normal VALIDATION rows."""
    set_seed()
    model = AutoEncoder(n_in, cfg["hidden"], cfg["latent"]).to(DEVICE)
    opt, crit = make_optim(cfg["optimizer"], model.parameters(), cfg["lr"]), make_loss(cfg["loss"])
    g = torch.Generator().manual_seed(C.SEED)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_n)), batch_size=cfg["batch"], shuffle=True, generator=g)
    xva = torch.from_numpy(Xva_n).to(DEVICE)
    hist, best_loss, best_state, bad = [], float("inf"), None, 0
    for ep in range(1, max_epochs + 1):
        model.train(); tot = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE); opt.zero_grad()
            loss = crit(model(xb), xb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vloss = crit(model(xva), xva).item()
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr_n), "val_loss": vloss})
        if vloss < best_loss - 1e-6:
            best_loss, best_state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError('Validation loss was never finite - check for NaN/inf in the scaled features')
    model.load_state_dict(best_state)
    return model, hist, best_loss


def main(feature_set="detrended", quick=False):
    t_all = time.perf_counter()
    data = load_experiment_data(feature_set)
    (_, Xtr, ytr), (_, Xva, yva) = data["train"], data["val"]
    feats = list(Xtr.columns)

    # scaler: fit on NORMAL TRAIN rows only; reused for val/test/inference
    scaler = StandardScaler().fit(Xtr[ytr == 0])
    sc = lambda X: scaler.transform(X[feats]).astype(np.float32)
    Xtr_n, Xva_n = sc(Xtr[ytr == 0]), sc(Xva[yva == 0])
    Xva_all = sc(Xva)
    log.info("device=%s | train normal rows=%d | val normal rows=%d | features=%d",
             DEVICE, len(Xtr_n), len(Xva_n), len(feats))

    cfgs = CONFIGS[:2] if quick else CONFIGS
    runs = []
    for cfg in cfgs:
        t0 = time.perf_counter()
        model, hist, best_loss = train_one(cfg, Xtr_n, Xva_n, len(feats), 5 if quick else MAX_EPOCHS)
        ap = average_precision_score(yva, recon_error(model, Xva_all))
        log.info("%-15s epochs=%2d best_val_loss=%.4f  val PR-AUC=%.4f  (%.0fs)",
                 cfg["name"], len(hist), best_loss, ap, time.perf_counter() - t0)
        runs.append({"cfg": cfg, "model": model, "hist": hist, "val_pr_auc": ap, "seconds": time.perf_counter() - t0})
    best = max(runs, key=lambda r: r["val_pr_auc"])
    log.info("BEST config: %s", best["cfg"]["name"])

    # evaluate best with the shared evaluation code (threshold from validation, test scored once)
    model = best["model"]
    score_fn = lambda X: recon_error(model, sc(X))
    res, scores = evaluate_model("autoencoder_" + feature_set, score_fn, data)

    # --- save weights + everything needed to reload (Drive -> VS Code) ---
    C.MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), C.MODELS_DIR / "autoencoder_best.pt")
    config = {"feature_set": feature_set, "feature_names": feats, "n_in": len(feats), **best["cfg"],
              "threshold": res["threshold"], "score_direction": "higher_is_more_anomalous",
              "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
              "seed": C.SEED, "torch_version": torch.__version__}
    (C.MODELS_DIR / "autoencoder_config.json").write_text(json.dumps(config, indent=2))

    # --- plots: loss curves + reconstruction-error distribution (log scale) ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    h = pd.DataFrame(best["hist"])
    ax[0].plot(h.epoch, h.train_loss, label="train (normal)"); ax[0].plot(h.epoch, h.val_loss, label="validation (normal)")
    ax[0].set(title=f"Loss curves - {best['cfg']['name']}", xlabel="epoch", ylabel=best["cfg"]["loss"]); ax[0].legend()
    yte = data["test"][2].values
    bins = np.logspace(np.log10(max(scores["test"].min(), 1e-4)), np.log10(scores["test"].max()), 60)
    ax[1].hist(scores["test"][yte == 0], bins=bins, alpha=.6, label="normal"); ax[1].hist(scores["test"][yte == 1], bins=bins, alpha=.6, label="anomaly")
    ax[1].axvline(res["threshold"], c="k", ls="--", label="threshold"); ax[1].set_xscale("log")
    ax[1].set(title="Reconstruction error (test)", xlabel="MSE (log scale)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "autoencoder_loss_and_error.png", dpi=120); plt.close(fig)

    save_results("autoencoder_" + feature_set, {
        "run": "autoencoder_" + feature_set, "model": "autoencoder", "feature_set": feature_set, "holdout": None,
        "seed": C.SEED, "best_config": best["cfg"], "selection_metric": "val_pr_auc",
        "all_configs": [{"cfg": r["cfg"], "val_pr_auc": r["val_pr_auc"], "epochs": len(r["hist"])} for r in runs],
        "train_rows": int(len(Xtr_n)), "n_features": len(feats), "feature_names": feats,
        "train_seconds": time.perf_counter() - t_all, "loss_history": best["hist"], "device": str(DEVICE), **res})

    t, v = res["metrics"]["test"], res["metrics"]["val"]
    print(f"\n[autoencoder_{feature_set}] best={best['cfg']['name']}\n  val  PR-AUC={v['pr_auc']:.3f} F1={v['f1']:.3f}  thr={res['threshold']:.4f}"
          f"\n  test PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} P={t['precision']:.3f} "
          f"R={t['recall']:.3f} F1={t['f1']:.3f} FAR={t['false_alarm_rate']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="detrended", choices=["detrended", "full"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.feature_set, a.quick)