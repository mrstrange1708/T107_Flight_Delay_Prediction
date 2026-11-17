"""Phase 2 · the network, its training loop and the honest evaluation protocol.
Owner: Siraparapu Shiva Sankar Mani Kanta (Mani Kanta).

Three networks, all in one class so ablations switch parts on and off:
  MLP       numeric features + LEARNED EMBEDDINGS for carrier, origin, destination
  Temporal  + a GRU that reads the last 14 days of delay outcomes (dl_inputs.py)
  Hybrid    + logistic regression's logit as a fixed starting point: the network only learns
            a correction on top of LR (same idea as Phase 1's boosting-from-LR). The last layer
            starts at zero, so before training the hybrid IS logistic regression.
"""
import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .dl_inputs import emb_dim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- network
class DelayNet(nn.Module):
    def __init__(self, n_num, cards, hidden=(256, 128), dropout=0.2, use_emb=True,
                 seq_channels=0, gru_hidden=32, use_offset=False):
        super().__init__()
        self.use_emb, self.use_seq, self.use_offset = use_emb, seq_channels > 0, use_offset
        if use_emb:
            self.embs = nn.ModuleList([nn.Embedding(n, emb_dim(n)) for n in cards])
            d_cat = sum(emb_dim(n) for n in cards)
        else:  # ablation: one-hot instead of learned embeddings
            self.cards = cards
            d_cat = sum(cards)
        if self.use_seq:
            self.gru = nn.GRU(seq_channels, gru_hidden, batch_first=True)
            d_seq = gru_hidden
        else:
            d_seq = 0
        layers, d = [], n_num + d_cat + d_seq
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        if use_offset:   # start exactly at logistic regression: correction = 0
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x_num, x_cat, seq=None, offset=None):
        parts = [x_num]
        if self.use_emb:
            parts += [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        else:
            parts += [nn.functional.one_hot(x_cat[:, i], n).float() for i, n in enumerate(self.cards)]
        if self.use_seq:
            _, h = self.gru(seq)          # h: last hidden state = a summary of the 14 days
            parts.append(h[-1])
        logit = self.head(self.body(torch.cat(parts, dim=1))).squeeze(1)
        if self.use_offset:
            logit = logit + offset
        return logit


# ----------------------------------------------------------------- training
def _batches(n, batch_size, shuffle, gen):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def _tensors(d):
    return {k: torch.as_tensor(v) for k, v in d.items() if v is not None}


def predict_logits(model, data, batch_size=4096):
    model.eval()
    t = _tensors(data)
    out = []
    with torch.no_grad():
        for b in _batches(len(t["x_num"]), batch_size, False, None):
            out.append(model(t["x_num"][b].to(DEVICE), t["x_cat"][b].to(DEVICE),
                             t["seq"][b].to(DEVICE) if "seq" in t else None,
                             t["offset"][b].to(DEVICE) if "offset" in t else None).cpu())
    return torch.cat(out).numpy()


def train(model, data, y, stop=None, y_stop=None, epochs=30, lr=1e-3, weight_decay=1e-4,
          batch_size=1024, patience=3, seed=42, verbose=False):
    """Mini-batch AdamW on log loss (BCE with logits).

    With a stop set: keep the epoch with the best stop-set PR AUC and stop after `patience`
    epochs without improvement (early stopping). Without one: run exactly `epochs` epochs.
    Returns (model, best_epoch, history)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    t, yt = _tensors(data), torch.as_tensor(y, dtype=torch.float32)
    best, best_ep, best_state, hist, wait = -1.0, epochs, None, [], 0
    if stop is not None and getattr(model, "use_offset", False):
        # The hybrid starts as logistic regression (zero correction). Score that "epoch 0" too, so
        # early stopping can decide that NO correction transfers and keep plain LR.
        best = float(average_precision_score(y_stop, predict_logits(model, stop)))
        best_ep, best_state = 0, copy.deepcopy(model.state_dict())
        hist.append({"epoch": 0, "train_loss": float("nan"), "stop_pr_auc": best})
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for b in _batches(len(yt), batch_size, True, gen):
            logit = model(t["x_num"][b].to(DEVICE), t["x_cat"][b].to(DEVICE),
                          t["seq"][b].to(DEVICE) if "seq" in t else None,
                          t["offset"][b].to(DEVICE) if "offset" in t else None)
            loss = loss_fn(logit, yt[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        row = {"epoch": ep, "train_loss": total / len(yt)}
        if stop is not None:
            row["stop_pr_auc"] = float(average_precision_score(y_stop, predict_logits(model, stop)))
            if row["stop_pr_auc"] > best:
                best, best_ep, best_state, wait = row["stop_pr_auc"], ep, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
        hist.append(row)
        if verbose:
            print(row, flush=True)
        if stop is not None and wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, hist


# ----------------------------------------------------------------- honest protocol
def prepare(frame, seq_all, numeric, categorical, fit_idx, idx_list, use_seq=False, use_offset=False,
            lr_C=0.001, offset_numeric=None):
    """Fit every statistic on frame.iloc[fit_idx] ONLY (encoder; logistic regression for the offset)
    and build model inputs for each index array in idx_list. Returns (inputs, n_num, cards).
    offset_numeric: the numeric columns logistic regression uses (default: the same as the network)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from .preprocess import linear_preprocessor
    from .dl_inputs import TabularEncoder
    fit = frame.iloc[fit_idx]
    enc = TabularEncoder(numeric, categorical).fit(fit)
    lr = None
    if use_offset:
        cols = list(offset_numeric or numeric) + list(categorical)
        lr_num = list(offset_numeric or numeric)
        lr = make_pipeline(linear_preprocessor(lr_num, categorical), LogisticRegression(C=lr_C, max_iter=3000))
        lr.fit(fit[cols], fit["delayed"].astype(int))
    out = []
    for idx in idx_list:
        part = frame.iloc[idx]
        x_num, x_cat = enc.transform(part)
        d = {"x_num": x_num, "x_cat": x_cat}
        if use_seq:
            d["seq"] = np.ascontiguousarray(seq_all[idx])
        if use_offset:
            d["offset"] = lr.decision_function(part[cols]).astype(np.float32)
        out.append(d)
    return out, out[0]["x_num"].shape[1], enc.cardinalities()


NET_KEYS = ("hidden", "dropout", "use_emb", "gru_hidden", "use_offset")
TRAIN_KEYS = ("lr", "weight_decay", "batch_size", "patience")


def build_net(cfg, n_num, cards, seq_channels):
    kw = {k: cfg[k] for k in NET_KEYS if k in cfg}
    return DelayNet(n_num, cards, seq_channels=seq_channels if cfg.get("use_seq") else 0, **kw)


def rolling_eval_nn(frame, seq_all, month, cfg, numeric, categorical, months=(7, 8, 9), seed=42, max_epochs=20,
                    offset_numeric=None):
    """Same honest protocol as Phase 1's boosting, with EPOCHS in place of trees. For month m:
      1. fit on months < m-1, early-stop on month m-1  -> best epoch count
      2. refit on all months < m for that many epochs
      3. score month m (never seen in steps 1-2).
    Returns (PR AUC per month, epochs per month)."""
    y = frame["delayed"].astype(int).to_numpy()
    tk = {k: cfg[k] for k in TRAIN_KEYS if k in cfg}
    C = seq_all.shape[2] if seq_all is not None else 0
    scores, epochs = [], []
    for m in months:
        inner, stop = np.where(month < m - 1)[0], np.where(month == m - 1)[0]
        full, test = np.where(month < m)[0], np.where(month == m)[0]
        (di, ds), n_num, cards = prepare(frame, seq_all, numeric, categorical, inner, [inner, stop],
                                          cfg.get("use_seq", False), cfg.get("use_offset", False),
                                          offset_numeric=offset_numeric)
        net = build_net(cfg, n_num, cards, C)
        _, ep, _ = train(net, di, y[inner], ds, y[stop], epochs=max_epochs, seed=seed, **tk)
        (df_, dt), n_num, cards = prepare(frame, seq_all, numeric, categorical, full, [full, test],
                                           cfg.get("use_seq", False), cfg.get("use_offset", False),
                                          offset_numeric=offset_numeric)
        net = build_net(cfg, n_num, cards, C)
        net, _, _ = train(net, df_, y[full], epochs=ep, seed=seed, **tk)
        scores.append(float(average_precision_score(y[test], predict_logits(net, dt))))
        epochs.append(int(ep))
    return scores, epochs
