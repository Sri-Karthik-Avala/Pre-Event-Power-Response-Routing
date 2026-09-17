# made by - Karthik
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

SEED = 42
EPOCHS = 30
FOLDS = 5
SEEDS = [0]
BATCH = 64
LR = 2e-3
HID = 96
W_NET = 0.20
W_ET = 0.50
W_GB = 0.30

DEVICE = torch.device("cpu")
torch.set_num_threads(8)
torch.manual_seed(SEED)
np.random.seed(SEED)

SEQ_COLS = [
    "voltage_V_sequence",
    "current_A_sequence",
    "power_W_sequence",
    "frequency_Hz_sequence",
    "power_factor_sequence",
    "wifi_rssi_dBm_sequence",
]

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")


def stack_sequences(df):
    chans = []
    for col in SEQ_COLS:
        chans.append(np.array([json.loads(s) for s in df[col]], dtype=np.float32))
    return np.stack(chans, 1)


def window_groups(arr):
    span = 8
    index = defaultdict(list)
    n = len(arr)
    ref = arr[:, 2]
    for i in range(n):
        for j in range(ref.shape[1] - span + 1):
            index[tuple(ref[i, j:j + span])].append(i)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in index.values():
        uniq = sorted(set(members))
        for other in uniq[1:]:
            a, b = find(uniq[0]), find(other)
            if a != b:
                parent[a] = b
    return np.array([find(i) for i in range(n)])


def summary_features(arr):
    v, i, p, f, pf, w = [arr[:, k].astype(np.float64) for k in range(6)]
    cols = []
    for a in (v, i, p, f, pf, w):
        m = a.mean(1)
        s = a.std(1)
        cols += [
            m, s, s / (np.abs(m) + 1e-6), np.ptp(a, 1),
            np.percentile(a, 10, 1) - m, np.percentile(a, 90, 1) - m,
            np.percentile(a, 75, 1) - np.percentile(a, 25, 1),
        ]
    cols += [
        np.log1p(i.mean(1)), np.log1p(p.mean(1)),
        (w.std(1) == 0).astype(np.float64), w.min(1), w.max(1) - w.min(1),
        (v.std(1) / v.mean(1)) / (i.std(1) / (i.mean(1) + 1e-6) + 1e-6),
        pf.std(1) / (v.std(1) / v.mean(1) + 1e-9),
        i.mean(1) / (v.mean(1) + 1e-6),
        p.mean(1) / (v.mean(1) * i.mean(1) + 1e-6),
    ]
    return np.nan_to_num(np.stack(cols, 1).astype(np.float32))


def sequence_tensor(arr, mu, sd):
    absolute = (arr - mu) / sd
    med = np.median(arr, axis=2, keepdims=True)
    spread = (np.percentile(arr, 75, 2, keepdims=True) - np.percentile(arr, 25, 2, keepdims=True)) + 1e-6
    shape = (arr - med) / spread
    return np.concatenate([absolute, shape], 1).astype(np.float32)


class RouteNet(nn.Module):
    def __init__(self, cin, naux, hid):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, hid, 5, padding=2), nn.BatchNorm1d(hid), nn.GELU(),
            nn.Conv1d(hid, hid, 3, padding=1), nn.BatchNorm1d(hid), nn.GELU(),
            nn.Conv1d(hid, hid, 3, padding=1), nn.BatchNorm1d(hid), nn.GELU(),
        )
        self.aux = nn.Sequential(
            nn.Linear(naux, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hid * 3 + 64, 128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.route = nn.Linear(64, 3)
        self.urgency = nn.Linear(64, 4)

    def forward(self, x, f):
        z = self.conv(x)
        z = torch.cat([z.mean(2), z.amax(2), z.std(2)], 1)
        z = self.head(torch.cat([z, self.aux(f)], 1))
        return self.route(z), self.urgency(z)


def fit_net(xs, ax, yr, yu, sw, cwr, cwu, tr_idx, seed):
    torch.manual_seed(seed)
    model = RouteNet(xs.shape[1], ax.shape[1], HID).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = EPOCHS * max(1, len(tr_idx) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    xt = torch.tensor(xs[tr_idx]).to(DEVICE)
    at = torch.tensor(ax[tr_idx]).to(DEVICE)
    rt = torch.tensor(yr[tr_idx]).to(DEVICE)
    ut = torch.tensor(yu[tr_idx]).to(DEVICE)
    wt = torch.tensor(sw[tr_idx]).to(DEVICE)
    cr = torch.tensor(cwr).to(DEVICE)
    cu = torch.tensor(cwu).to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    done = 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr_idx), generator=gen)
        for s in range(0, len(tr_idx) - 1, BATCH):
            b = perm[s:s + BATCH]
            if len(b) < 8:
                continue
            x = xt[b] + torch.randn(xt[b].shape, generator=gen) * 0.05
            pr, pu = model(x, at[b])
            lr_loss = (nn.functional.cross_entropy(pr, rt[b], weight=cr, reduction="none") * wt[b]).mean()
            lu_loss = (nn.functional.cross_entropy(pu, ut[b], weight=cu, reduction="none") * wt[b]).mean()
            loss = 0.65 * lr_loss + 0.35 * lu_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if done < steps:
                sched.step()
                done += 1
    model.eval()
    return model


def net_predict(model, xs, ax, idx):
    with torch.no_grad():
        pr, pu = model(torch.tensor(xs[idx]).to(DEVICE), torch.tensor(ax[idx]).to(DEVICE))
    return torch.softmax(pr, 1).numpy(), torch.softmax(pu, 1).numpy()


routes = sorted(set(t.split("|")[0] for t in train["target"]))
urgencies = sorted(set(t.split("|")[1] for t in train["target"]))
r_index = {k: i for i, k in enumerate(routes)}
u_index = {k: i for i, k in enumerate(urgencies)}

train_signals = stack_sequences(train)
test_signals = stack_sequences(test)

y_route = train["target"].str.split("|").str[0].map(r_index).values.astype(np.int64)
y_urg = train["target"].str.split("|").str[1].map(u_index).values.astype(np.int64)

groups = window_groups(train_signals)
counts = pd.Series(groups).map(pd.Series(groups).value_counts()).values.astype(np.float32)
event_w = (1.0 / counts).astype(np.float32)
sample_w = (event_w / event_w.mean()).astype(np.float32)

mu = train_signals.mean((0, 2), keepdims=True)
sd = train_signals.std((0, 2), keepdims=True) + 1e-6
train_seq = sequence_tensor(train_signals, mu, sd)
test_seq = sequence_tensor(test_signals, mu, sd)

train_aux_raw = summary_features(train_signals)
test_aux_raw = summary_features(test_signals)
a_mu = train_aux_raw.mean(0)
a_sd = train_aux_raw.std(0) + 1e-6
train_aux = ((train_aux_raw - a_mu) / a_sd).clip(-8, 8).astype(np.float32)
test_aux = ((test_aux_raw - a_mu) / a_sd).clip(-8, 8).astype(np.float32)

route_prior = np.bincount(y_route, weights=event_w, minlength=len(routes))
route_prior = route_prior / route_prior.sum()
urg_prior = np.bincount(y_urg, weights=event_w, minlength=len(urgencies))
urg_prior = urg_prior / urg_prior.sum()
cw_route = (1.0 / (route_prior * len(routes))).astype(np.float32)
cw_urg = (1.0 / (urg_prior * len(urgencies))).astype(np.float32)

n_tr = len(train)
n_te = len(test)

wr_all = sample_w * cw_route[y_route]
wu_all = sample_w * cw_urg[y_urg]

net = fit_net(train_seq, train_aux, y_route, y_urg, sample_w, cw_route, cw_urg, np.arange(n_tr), SEEDS[0])
net_r, net_u = net_predict(net, test_seq, test_aux, np.arange(n_te))

er = ExtraTreesClassifier(600, min_samples_leaf=2, random_state=SEEDS[0], n_jobs=4)
er.fit(train_aux, y_route, sample_weight=wr_all)
eu = ExtraTreesClassifier(600, min_samples_leaf=2, random_state=SEEDS[0], n_jobs=4)
eu.fit(train_aux, y_urg, sample_weight=wu_all)

gr = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15,
                                    l2_regularization=0.5, random_state=SEEDS[0])
gr.fit(train_aux, y_route, sample_weight=wr_all)
gu = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=15,
                                    l2_regularization=0.5, random_state=SEEDS[0])
gu.fit(train_aux, y_urg, sample_weight=wu_all)

test_r = W_NET * net_r + W_ET * er.predict_proba(test_aux) + W_GB * gr.predict_proba(test_aux)
test_u = W_NET * net_u + W_ET * eu.predict_proba(test_aux) + W_GB * gu.predict_proba(test_aux)

pick_r = test_r.argmax(1)
pick_u = test_u.argmax(1)
print("route rate", np.bincount(pick_r, minlength=len(routes)) / len(pick_r))
print("urgency rate", np.bincount(pick_u, minlength=len(urgencies)) / len(pick_u))

submission = pd.DataFrame({
    "id": test["id"],
    "target": [routes[a] + "|" + urgencies[b] for a, b in zip(pick_r, pick_u)],
})
submission_out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_out, index=False)
