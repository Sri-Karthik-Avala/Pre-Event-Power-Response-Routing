import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

ROUTES = ["equipment_escalation", "supply_quality", "sensor_telemetry"]
URG = ["incipient", "low", "medium", "urgent"]
SEQ = ['voltage_V_sequence', 'current_A_sequence', 'power_W_sequence',
       'frequency_Hz_sequence', 'power_factor_sequence', 'wifi_rssi_dBm_sequence']

TR = pd.read_csv("train.csv")
TE = pd.read_csv("test.csv")
X = np.load("Xtr.npy").astype(np.float64)      # (1684, 6, 60)  train signals
XT = np.load("Xte.npy").astype(np.float64)     # (676, 6, 60)   test signals
G = np.load("gtr.npy")                         # event group id per train row (183 events)
YR = TR.target.str.split('|').str[0].map({k: i for i, k in enumerate(ROUTES)}).values
YU = TR.target.str.split('|').str[1].map({k: i for i, k in enumerate(URG)}).values
CNT = pd.Series(G).map(pd.Series(G).value_counts()).values
EW = 1.0 / CNT                                 # event weight, exactly as the metric defines it
COMB = YR * 4 + YU
FOLDS = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X.reshape(len(X), -1), COMB, G))

CEILING = 0.65 / 3 + 0.35 / 4                  # 0.3042 zero-information optimum


def score(pr, pu):
    a = f1_score(YR, pr, labels=[0, 1, 2], average='macro', sample_weight=EW)
    b = f1_score(YU, pu, labels=[0, 1, 2, 3], average='macro', sample_weight=EW)
    return 0.65 * a + 0.35 * b, a, b


def _fit(mk, F, y, K):
    P = np.zeros((len(F), K))
    for a, b in FOLDS:
        sc = StandardScaler().fit(F[a])
        A, B = sc.transform(F[a]), sc.transform(F[b])
        cw = EW[a] / np.bincount(y[a], weights=EW[a], minlength=K)[y[a]]
        m = mk()
        m.fit(A, y[a], sample_weight=cw)
        P[b] = m.predict_proba(B)
    return P


MODELS = {
    "LR": lambda: LogisticRegression(max_iter=2000, C=0.3),
    "HGB": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=7,
                                                  l2_regularization=5.0, random_state=0),
    "ET": lambda: ExtraTreesClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=2),
}


def run(F, name):
    F = np.nan_to_num(np.asarray(F, dtype=np.float64))
    best = None
    for mn, mk in MODELS.items():
        PR = _fit(mk, F, YR, 3)
        PU = _fit(mk, F, YU, 4)
        s = score(PR.argmax(1), PU.argmax(1))
        print("%-28s %-4s score %.4f route %.4f urg %.4f  (ceiling %.4f)" % (name, mn, s[0], s[1], s[2], CEILING))
        if best is None or s[0] > best[0]:
            best = s
    return best
