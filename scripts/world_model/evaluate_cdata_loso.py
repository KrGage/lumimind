"""Leakage-controlled, subject-level LOSO evaluation of C-data transitions.

Run: python scripts/world_model/evaluate_cdata_loso.py
The generated report contains raw-scale MAE; no held-out subject is used to fit
scalers, generators, model parameters, or baselines.
"""
from __future__ import annotations

import json
import mimetypes
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

# The restricted Windows runner cannot read the MIME registry used by openpyxl.
# XLSX MIME types are unnecessary for reading the source workbook.
if hasattr(mimetypes, "_mimetypes_read_windows_registry"):
    mimetypes._mimetypes_read_windows_registry = lambda add_type: None

from cdata_loader import GROUPS, STATE_KEYS, build_dataset, build_transitions

HERE = Path(__file__).resolve().parent
REPORT = HERE / "cdata_loso_phase1_report.json"
DIM = len(STATE_KEYS)
PA, NA = STATE_KEYS.index("PA"), STATE_KEYS.index("NA")
SEED = 20261026


def fold_dataset(source, train_ids):
    """Fit state moments on training subjects only; action moments are fixed design values."""
    mean, std = {}, {}
    for key in STATE_KEYS:
        values = [v for sid in train_ids for v in source.measures.get(key, {}).get(sid, {}).values()
                  if v is not None and np.isfinite(v)]
        a = np.asarray(values, dtype=float)
        mean[key] = float(a.mean()) if len(a) else 0.0
        sd = float(a.std()) if len(a) else 1.0
        std[key] = sd if sd > 1e-6 else 1.0
    return replace(source, subjects=list(train_ids), m_mean=mean, m_std=std)


def features(ds, sid, group, step):
    current = ds.state_raw(sid, group, step + 1)
    previous = ds.state_raw(sid, group, step) if step else np.full(DIM, np.nan)
    mu = np.array([ds.m_mean[k] for k in STATE_KEYS])
    sd = np.array([ds.m_std[k] for k in STATE_KEYS])
    def encode(raw):
        mask = np.isfinite(raw).astype(np.float32)
        z = np.where(mask, (raw - mu) / sd, 0).astype(np.float32)
        return z, mask
    c, cm = encode(current)
    p, pm = encode(previous)
    return np.concatenate([c, cm, p, pm, ds.action_vec(group), [float(step)] ]).astype(np.float32)


def real_rows(ds):
    rows = []
    for tr in build_transitions(ds):
        sid, g, step = tr["sid"], tr["g"], tr["t_from"] - 1
        before, after = tr["raw_from"], tr["raw_to"]
        mask = (np.isfinite(before) & np.isfinite(after)).astype(np.float32)
        delta = np.where(mask, tr["s_to"] - tr["s_from"], 0).astype(np.float32)
        rows.append(dict(sid=sid, group=g, step=step, x=features(ds, sid, g, step),
                         delta=delta, valid=mask, before=before, after=after))
    return rows


def fit_ridges(rows):
    x = np.stack([r["x"] for r in rows])
    y = np.stack([r["delta"] for r in rows])
    mask = np.stack([r["valid"] for r in rows]).astype(bool)
    models, residuals = [], []
    for d in range(DIM):
        if mask[:, d].sum() < 3:
            models.append(None)
            residuals.append(np.array([0.0]))
            continue
        model = Ridge(alpha=10.0).fit(x[mask[:, d]], y[mask[:, d], d])
        models.append(model)
        residuals.append(y[mask[:, d], d] - model.predict(x[mask[:, d]]))
    return models, residuals


def predict_ridges(models, x):
    x = np.asarray(x).reshape(1, -1)
    return np.array([m.predict(x)[0] if m is not None else 0.0 for m in models], np.float32)


def mock_rows(ds, real, models, residuals, n_subjects=30, seed=SEED):
    """Resample training baselines and training missingness, then roll out fold-fitted dynamics."""
    rng = np.random.default_rng(seed)
    mu = np.array([ds.m_mean[k] for k in STATE_KEYS]); sd = np.array([ds.m_std[k] for k in STATE_KEYS])
    lookup = {(r["sid"], r["group"], r["step"]): r for r in real}
    origins = [(sid, g) for sid in ds.subjects for g in GROUPS
               if (sid, g, 0) in lookup and (sid, g, 1) in lookup]
    output = []
    for n in range(n_subjects):
        for g in GROUPS:
            sid0, g0 = origins[int(rng.integers(len(origins)))]
            pattern = [lookup[(sid0, g0, step)] for step in (0, 1)]
            state = ds.state_raw(sid0, g0, 1).copy()
            state = np.where(np.isfinite(state), state, mu)
            previous = np.full(DIM, np.nan)
            for step, template in enumerate(pattern):
                current_mask = np.isfinite(template["before"])
                next_mask = np.isfinite(template["after"])
                prev_mask = np.isfinite(pattern[0]["before"]) if step else np.zeros(DIM, bool)
                cz = np.where(current_mask, (state - mu) / sd, 0)
                pz = np.where(prev_mask, (previous - mu) / sd, 0)
                x = np.concatenate([cz, current_mask, pz, prev_mask,
                                    ds.action_vec(g), [float(step)]]).astype(np.float32)
                delta = predict_ridges(models, x)
                delta += np.array([rng.choice(e) for e in residuals], np.float32)
                delta = np.clip(delta, -3, 3)
                nxt = state + delta * sd
                valid = (current_mask & next_mask).astype(np.float32)
                output.append(dict(sid=-(n + 1), group=g, step=step, x=x,
                                   delta=np.where(valid, delta, 0).astype(np.float32),
                                   valid=valid, before=np.where(current_mask, state, np.nan),
                                   after=np.where(next_mask, nxt, np.nan)))
                previous, state = state, nxt
    return output


class SmallWorldModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, DIM))

    def forward(self, x):
        return self.net(x)


class MaskedTransformer(nn.Module):
    """Two-step causal state history with a per-dimension observed-value mask."""
    def __init__(self, input_dim):
        super().__init__()
        self.project = nn.Linear(21, 32)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.head = nn.Linear(32, DIM)

    def forward(self, x):
        action = x[:, 32:36]
        step = x[:, 36:37]
        previous = torch.cat([x[:, 16:32], action, torch.zeros_like(step)], dim=1)
        current = torch.cat([x[:, :16], action, step], dim=1)
        tokens = self.project(torch.stack([previous, current], dim=1))
        causal = torch.tensor([[False, True], [False, False]], device=x.device)
        return self.head(self.encoder(tokens, mask=causal)[:, -1])


def train_neural(rows, seed, model_cls=SmallWorldModel, epochs=120):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x = torch.from_numpy(np.stack([r["x"] for r in rows]))
    y = torch.from_numpy(np.stack([r["delta"] for r in rows]))
    mask = torch.from_numpy(np.stack([r["valid"] for r in rows]))
    model = model_cls(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    for _ in range(epochs):
        prediction = model(x)
        loss = (((prediction - y) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def summarize(records):
    output = {}
    for metric, dims in {"valence_proxy_PA_minus_NA": (PA, NA), "PA": (PA,), "NA": (NA,)}.items():
        result = {}
        for name in ("persistence", "ridge", "small_world_real", "small_world_mock",
                     "transformer_real", "transformer_mock"):
            actual, forecast, subject = [], [], []
            for r in records:
                row = r["row"]
                if not all(row["valid"][d] for d in dims):
                    continue
                sign = np.array([1, -1] if len(dims) == 2 else [1])
                actual.append(float(np.sum((row["after"] - row["before"])[list(dims)] * sign)))
                scales = np.array([r["std"][d] for d in dims])
                forecast.append(float(np.sum(r[name][list(dims)] * scales * sign)))
                subject.append(row["sid"])
            a, p = np.array(actual), np.array(forecast)
            eligible = np.abs(a) > 0.5  # Ignore near-zero changes when scoring direction.
            result[name] = {"n": len(a), "mae_raw_units": round(float(np.mean(np.abs(a-p))), 4),
                            "direction_accuracy": round(float(np.mean(np.sign(p[eligible]) == np.sign(a[eligible]))), 4)
                            if eligible.any() else None,
                            "direction_n": int(eligible.sum()),
                            "subjects": len(set(subject))}
        output[metric] = result
    output["arousal"] = {"status": "unavailable", "reason": "C-data has no direct arousal self-report label; KSS and VAS-F measure sleepiness and fatigue."}
    return output


def run():
    source = build_dataset()
    records, folds = [], []
    for held in source.subjects:
        ids = [sid for sid in source.subjects if sid != held]
        train_ds = fold_dataset(source, ids)
        test_ds = replace(train_ds, subjects=[held])
        train, test = real_rows(train_ds), real_rows(test_ds)
        assert all(r["sid"] != held for r in train)
        assert all(r["sid"] == held for r in test)
        ridge, residuals = fit_ridges(train)
        synthetic = mock_rows(train_ds, train, ridge, residuals, seed=SEED + held)
        assert all(r["sid"] < 0 and np.isfinite(r["x"]).all() for r in synthetic)
        real_model = train_neural(train, SEED + held)
        mock_model = train_neural(train + synthetic, SEED + held)
        transformer_real = train_neural(train, SEED + held, MaskedTransformer)
        transformer_mock = train_neural(train + synthetic, SEED + held, MaskedTransformer)
        sd = np.array([train_ds.m_std[k] for k in STATE_KEYS])
        for row in test:
            x = torch.from_numpy(row["x"][None])
            with torch.no_grad():
                pr = real_model(x).numpy()[0]
                pm = mock_model(x).numpy()[0]
                tr = transformer_real(x).numpy()[0]
                tm = transformer_mock(x).numpy()[0]
            records.append({"row": row, "std": sd, "persistence": np.zeros(DIM),
                            "ridge": predict_ridges(ridge, row["x"][None]),
                            "small_world_real": pr, "small_world_mock": pm,
                            "transformer_real": tr, "transformer_mock": tm})
        folds.append({"held_subject": held, "train_subjects": ids, "n_train_real": len(train),
                      "n_train_mock": len(synthetic), "n_test_real": len(test),
                      "PA_train_mean": train_ds.m_mean["PA"]})
        print(f"LOSO held={held}: real={len(train)} mock={len(synthetic)} test={len(test)}", flush=True)
    report = {"protocol": "real-subject LOSO; fold-only moments/generator; input and target masks",
              "seed": SEED, "n_subjects": len(source.subjects), "n_test_transitions": len(records),
              "direction_definition": "sign of next-minus-current, scored only when absolute true change > 0.5 raw units; zero prediction is incorrect",
              "metrics": summarize(records), "folds": folds,
              "caveats": ["Valence is a PA−NA proxy, not a direct SAM valence rating.",
                          "Arousal cannot be scored on this C-data.",
                          "Transitions describe continued exposure within a lighting condition, not observed switches between lighting conditions."]}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    run()
