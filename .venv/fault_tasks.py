"""
fault_tasks.py
Training classes for Tasks 1-5. Runs after feature_engineering.py.

    from feature_engineering import FeatureEngineering
    from fault_tasks import TaskTrainer

    pipe = FeatureEngineering(training_data, training_labels)
    pipe.run()                                   # or pipe.load_features()
    trainer = TaskTrainer(pipe).run_all()        # train + evaluate + save models
    trainer.show()                               # results tables + confusion matrices

Tasks
    1  Normal vs abnormal                    (all recordings)
    2  Bubble / SV fault / Unknown           (abnormal recordings)
    3  Bubble location  BP1-BP7, BV1         (bubble recordings)
    4  Failed valve     SV1-SV4              (SV-fault recordings)
    5  Opening ratio of the failed valve     (SV-fault recordings, regression)

Every prediction is made per window and combined per recording (majority vote of the
recording's windows; mean for the Task 5 regression). All metrics are at recording level.
Saved: Tasks 1-4 -> ANN, GradBoost, LogReg;  Task 5 -> Ridge.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, mean_absolute_error, r2_score)

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

META_COLS = ["recording_id", "window", "label", "label_name"]
SV_COLS = ["SV1", "SV2", "SV3", "SV4"]
LOC_COLS = ["BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7", "BV1"]
LABEL_COLS = ["case", "spacecraft", "condition"] + SV_COLS + LOC_COLS
SAVE_CLASSIFIERS = ("ANN", "GradBoost", "LogReg")
SAVE_REGRESSOR = "Ridge"


# ----------------------------------------------------------------------
# Labels table
# ----------------------------------------------------------------------
def read_label_table(labels_path):
    """Full labels table: case, spacecraft, condition, SV1-SV4 (opening %), BP1-BP7, BV1 (Yes/No)."""
    raw = pd.read_excel(labels_path.rstrip("/\\"), header=None)
    norm = raw.astype(str).apply(lambda col: col.str.strip().str.lower())
    hdr_row = norm.index[(norm == "condition").any(axis=1)][0]
    lab = raw.loc[hdr_row + 1:, :].iloc[:, :len(LABEL_COLS)].copy()
    lab.columns = LABEL_COLS
    lab["case"] = pd.to_numeric(lab["case"], errors="coerce")
    lab = lab.dropna(subset=["case"]).copy()
    lab["case"] = lab["case"].astype(int)
    lab["condition"] = lab["condition"].astype(str).str.strip()
    for c in SV_COLS:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    for c in LOC_COLS:
        lab[c] = lab[c].astype(str).str.strip()
    return lab.reset_index(drop=True)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def make_supervised(n_classes=2):
    return {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=0),
        "LogReg":       LogisticRegression(max_iter=1000, class_weight="balanced", C=10.0),
        "NaiveBayes":   GaussianNB(priors=[1.0 / n_classes] * n_classes),
        "KNN(k=3)":     KNeighborsClassifier(n_neighbors=3),
        "ANN":          MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                      alpha=1e-3, max_iter=1000, random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=0, n_jobs=-1),
        "GradBoost":    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                   max_depth=3, random_state=0),
    }


def make_regressors():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=10.0),
        "DecisionTree":     DecisionTreeRegressor(max_depth=5, random_state=0),
    }


def thr_3sigma(s):
    return s.mean() + 3 * s.std()


class PCAScorer:
    """Reconstruction error with 95% of the variance (larger = further from the training windows)."""
    name = "PCA"

    def fit(self, X):
        self.pca = PCA(n_components=0.95, svd_solver="full").fit(X)
        self.train_scores = self.score(X)
        return self

    def score(self, X):
        R = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - R) ** 2).sum(axis=1)


class KMeansScorer:
    """Distance to the nearest K-Means centre."""
    def __init__(self, k=3):
        self.k = k
        self.name = f"KMeans(k={k})"

    def fit(self, X):
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=0).fit(X)
        self.train_scores = self.score(X)
        return self

    def score(self, X):
        return self.km.transform(X).min(axis=1)


class DBSCANScorer:
    """Distance to the nearest DBSCAN core point (eps = 90th percentile of k-distances)."""
    name = "DBSCAN"

    def __init__(self, minpts=5):
        self.minpts = minpts

    def fit(self, X):
        nn_k = NearestNeighbors(n_neighbors=self.minpts).fit(X)
        eps = np.percentile(nn_k.kneighbors(X)[0][:, -1], 90)
        core_idx = DBSCAN(eps=eps, min_samples=self.minpts).fit(X).core_sample_indices_
        assert len(core_idx) >= 2, "DBSCAN found <2 core points; increase eps percentile"
        self.nn_core = NearestNeighbors(n_neighbors=2).fit(X[core_idx])
        d2 = self.nn_core.kneighbors(X, n_neighbors=2)[0]
        is_core = np.zeros(len(X), dtype=bool)
        is_core[core_idx] = True
        self.train_scores = np.where(is_core, d2[:, 1], d2[:, 0])     # leave-self-out for core points
        return self

    def score(self, X):
        return self.nn_core.kneighbors(X, n_neighbors=1)[0][:, 0]


def make_scorer(name):
    return {"PCA": PCAScorer, "KMeans(k=3)": KMeansScorer, "DBSCAN": DBSCANScorer}[name]()


def make_unsupervised(Xn):
    """Fit each scorer on reference windows; threshold = mean + 3 sigma of the training scores."""
    out = {}
    for name in ["PCA", "KMeans(k=3)", "DBSCAN"]:
        sc = make_scorer(name).fit(Xn)
        sc.threshold = thr_3sigma(sc.train_scores)
        out[name] = sc
    return out


def majority_vote(rec_ids, pred):
    """One label per recording: majority vote of its windows."""
    t = pd.DataFrame({"recording_id": np.asarray(rec_ids), "pred": np.asarray(pred).astype(int)})
    return t.groupby("recording_id")["pred"].agg(lambda p: int(np.bincount(p).argmax()))


# ----------------------------------------------------------------------
# Base class for one task
# ----------------------------------------------------------------------
class BaseTask:
    key = ""                  # "task1" ... "task5"
    title = ""
    class_names = []
    positive_name = None      # binary tasks: name of class 1
    kind = "classification"
    has_val = True
    unsupervised = False      # also fit PCA / KMeans / DBSCAN on class-0 windows
    save_models = SAVE_CLASSIFIERS

    def __init__(self, feat_df, labels, seed=42):
        self.feat_df = feat_df
        self.labels = labels
        self.seed = seed
        self.models, self.preds, self.results = {}, {}, {}

    # subclasses provide these two
    def target(self):
        """{recording_id: target value} for the recordings this task uses."""
        raise NotImplementedError

    def make_split(self):
        """{"train": [...], "val": [...], "test": [...]} recording ids."""
        raise NotImplementedError

    # ---------- data ----------
    @property
    def eval_splits(self):
        return ["val", "test"] if self.has_val else ["test"]

    @property
    def n_classes(self):
        return len(self.class_names)

    def _stratified(self, ids, y, sizes):
        """sizes = (val, test) fractions; val may be 0."""
        val_f, test_f = sizes
        tr, tmp, y_tr, y_tmp = train_test_split(ids, y, test_size=val_f + test_f, stratify=y,
                                                random_state=self.seed)
        if val_f == 0:
            return {"train": tr, "test": tmp}
        va, te = train_test_split(tmp, test_size=test_f / (val_f + test_f), stratify=y_tmp,
                                  random_state=self.seed)
        return {"train": tr, "val": va, "test": te}

    def _windows(self, ids):
        d = self.feat_df[self.feat_df["recording_id"].isin(ids)].copy()
        d["label"] = d["recording_id"].map(self.y_of)
        return d.reset_index(drop=True)

    def prepare(self):
        self.y_of = self.target()
        split = self.make_split()
        self.split = {s: sorted(int(i) for i in ids) for s, ids in split.items()}
        names = list(self.split)
        for i in range(len(names)):                     # leakage check
            for j in range(i + 1, len(names)):
                assert not set(self.split[names[i]]) & set(self.split[names[j]]), "recording in two sets"
        self.F = {s: self._windows(ids) for s, ids in self.split.items()}
        fcols = [c for c in self.F["train"].columns if c not in META_COLS]
        keep = self.F["train"][fcols].to_numpy(float).std(axis=0) > 1e-12
        self.fcols = [c for c, k in zip(fcols, keep) if k]
        self.scaler = StandardScaler().fit(self.F["train"][self.fcols])
        self.X = {s: self.scaler.transform(self.F[s][self.fcols]) for s in self.F}
        self.ytr = self.F["train"]["label"].to_numpy()
        return self

    # ---------- evaluation ----------
    def evaluate(self, split_name, pred):
        """Recording-level metrics; binary: class 1 is positive, multi-class: macro average."""
        d = self.F[split_name]
        rec = majority_vote(d["recording_id"], pred)
        yt = rec.index.map(self.y_of).to_numpy().astype(int)
        yp = rec.to_numpy()
        L = list(range(self.n_classes))
        kw = dict(pos_label=1, average="binary") if self.n_classes == 2 else dict(labels=L, average="macro")
        return {
            "acc":       accuracy_score(yt, yp),
            "f1":        f1_score(yt, yp, zero_division=0, **kw),
            "precision": precision_score(yt, yp, zero_division=0, **kw),
            "recall":    recall_score(yt, yp, zero_division=0, **kw),
            "cm":        confusion_matrix(yt, yp, labels=L),      # rows = true, cols = predicted
            "n":         len(yt),
        }

    # ---------- training ----------
    def fit(self):
        if self.unsupervised:
            for name, sc in make_unsupervised(self.X["train"][self.ytr == 0]).items():
                self.preds[name] = {s: self.post(s, (sc.score(self.X[s]) > sc.threshold).astype(int))
                                    for s in self.eval_splits}
                self.results[name] = {"type": "unsupervised",
                                      **{s: self.evaluate(s, self.preds[name][s]) for s in self.eval_splits}}
        for name, clf in make_supervised(self.n_classes).items():
            clf.fit(self.X["train"], self.ytr)
            self.models[name] = clf
            self.preds[name] = {s: self.post(s, clf.predict(self.X[s])) for s in self.eval_splits}
            self.results[name] = {"type": "supervised",
                                  **{s: self.evaluate(s, self.preds[name][s]) for s in self.eval_splits}}
        return self

    def post(self, split_name, pred):
        """Hook applied to every model's window predictions (Task 2 adds the Unknown gate)."""
        return np.asarray(pred).astype(int)

    def run(self):
        return self.prepare().fit()

    # ---------- output ----------
    def table(self):
        rows = []
        for name, r in self.results.items():
            row = {"model": name, "type": r["type"]}
            if self.has_val:
                row["val_acc%"] = round(100 * r["val"]["acc"], 1)
                row["val_F1"] = round(r["val"]["f1"], 3)
            t = r["test"]
            row.update({"test_acc%": round(100 * t["acc"], 1), "test_F1": round(t["f1"], 3),
                        "test_precision": round(t["precision"], 3), "test_recall": round(t["recall"], 3)})
            if "n_unknown" in t:
                row["test_unknown"] = t["n_unknown"]
            if "MAE" in t:
                row["test_MAE"] = round(t["MAE"], 1)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def split_counts(self):
        """Recordings per set and class."""
        rows = []
        for s, ids in self.split.items():
            cnt = pd.Series([self.y_of[i] for i in ids]).value_counts()
            rows.append({"set": s, "recordings": len(ids),
                         **{self._class_label(k): int(cnt.get(k, 0)) for k in self._class_keys()}})
        return pd.DataFrame(rows).set_index("set")

    def _class_keys(self):
        return list(range(self.n_classes))

    def _class_label(self, k):
        return self.class_names[k]

    def tick_labels(self):
        return self.class_names

    def plot_confusions(self, split_name="test", models=None, ncol=4):
        names = models or list(self.results)
        nrow = int(np.ceil(len(names) / ncol))
        k = len(self.tick_labels())
        size = 3.2 + 0.18 * k
        fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow), squeeze=False)
        axes = axes.ravel()
        for ax, nm in zip(axes, names):
            r = self.results[nm][split_name]
            plot_cm(ax, r["cm"], self.tick_labels(),
                    f"{nm}\nAcc={100 * r['acc']:.1f}%  F1={r['f1']:.3f}")
        for ax in axes[len(names):]:
            ax.axis("off")
        fig.suptitle(f"{self.title} - confusion matrices ({split_name}, recording level)", fontsize=12)
        fig.tight_layout()
        return fig

    def bundle(self):
        """What gets saved for prediction on new data."""
        return {"task": self.key, "title": self.title, "kind": self.kind,
                "models": {m: self.models[m] for m in self.save_models},
                "scaler": self.scaler, "fcols": self.fcols, "classes": self.class_names}

    def summary(self):
        """Picklable results for the dashboard."""
        return {"title": self.title, "kind": self.kind, "class_names": self.tick_labels(),
                "positive": self.positive_name, "has_val": self.has_val,
                "results": self.results, "table": self.table(), "split_counts": self.split_counts(),
                "split": self.split, "n_features": len(self.fcols), "saved_models": list(self.save_models)}


def plot_cm(ax, cm, labels, title):
    k = len(labels)
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=11 if k <= 4 else 8,
                    color="white" if cm.max() and cm[i, j] > cm.max() / 2 else "#0b0b0b")
    rot = 90 if k > 4 else 0
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=rot, fontsize=8)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)


# ----------------------------------------------------------------------
# The five tasks
# ----------------------------------------------------------------------
class Task1NormalAbnormal(BaseTask):
    key, title = "task1", "Task 1: Normal vs abnormal"
    class_names, positive_name = ["Normal", "Abnormal"], "Abnormal"
    unsupervised = True

    def target(self):
        return self.feat_df.groupby("recording_id")["label"].first().astype(int).to_dict()

    def make_split(self):
        ids = np.array(sorted(self.y_of))
        return self._stratified(ids, np.array([self.y_of[i] for i in ids]), (0.20, 0.20))


class Task2FaultType(BaseTask):
    """Abnormal recordings -> Bubble (0) / SV fault (1) / Unknown (2).
    Stage 2a: an unsupervised detector fitted on the KNOWN abnormal training windows flags a window
              as Unknown when its score is above mean + 3 sigma of OUT-OF-FOLD training scores
              (5 folds by recording: each window is scored by a detector that never saw its recording;
              in-sample scores are too low and flag many known recordings as Unknown).
    Stage 2b: the classifier (Bubble vs SV fault) labels the windows judged known.
    Recording = majority vote of its windows; no unique majority -> Unknown.
    The training data has no unknown faults, so on validation / test every Unknown is an error;
    test_unknown counts how many known recordings were wrongly flagged."""
    key, title = "task2", "Task 2: Bubble / SV fault / Unknown"
    class_names, positive_name = ["Bubble", "SV fault"], "SV fault"
    unknown_name = "Unknown"
    unsupervised = True

    def __init__(self, feat_df, labels, task1_split, seed=42, unknown_detector="KMeans(k=3)"):
        super().__init__(feat_df, labels, seed)
        self.task1_split = task1_split
        self.unknown_detector = unknown_detector

    def target(self):
        cond = self.labels.set_index("case")["condition"].str.lower()
        return {int(c): (1 if v == "fault" else 0) for c, v in cond.items() if v != "normal"}

    def make_split(self):                    # same recordings per set as Task 1
        return {s: [i for i in ids if i in self.y_of] for s, ids in self.task1_split.items()}

    def tick_labels(self):
        return self.class_names + [self.unknown_name]

    # ---- stage 2a ----
    def fit(self):
        X, groups = self.X["train"], self.F["train"]["recording_id"].to_numpy()
        oof = np.zeros(len(X))
        for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, groups=groups):
            oof[te] = make_scorer(self.unknown_detector).fit(X[tr]).score(X[te])
        self.gate = make_scorer(self.unknown_detector).fit(X)       # final detector: all known abnormal windows
        self.gate.threshold = thr_3sigma(oof)
        self.gate.oof_scores = oof
        return super().fit()

    def is_unknown(self, X):
        return self.gate.score(X) > self.gate.threshold

    def post(self, split_name, pred):
        pred = np.asarray(pred).astype(int).copy()
        pred[self.is_unknown(self.X[split_name])] = 2
        return pred

    @staticmethod
    def vote3(rec_ids, pred):
        """Majority vote over Bubble / SV fault / Unknown; no unique majority -> Unknown."""
        t = pd.DataFrame({"recording_id": np.asarray(rec_ids), "pred": np.asarray(pred).astype(int)})

        def v(p):
            cnt = np.bincount(p, minlength=3)
            top = np.flatnonzero(cnt == cnt.max())
            return int(top[0]) if len(top) == 1 else 2
        return t.groupby("recording_id")["pred"].agg(v)

    def evaluate(self, split_name, pred):
        """SV fault = positive class for F1 / precision / recall; Unknown counts as wrong."""
        d = self.F[split_name]
        rec = self.vote3(d["recording_id"], pred)
        yt = rec.index.map(self.y_of).to_numpy().astype(int)
        yp = rec.to_numpy()
        kw = dict(labels=[1], average="macro", zero_division=0)
        return {"acc": accuracy_score(yt, yp), "f1": f1_score(yt, yp, **kw),
                "precision": precision_score(yt, yp, **kw), "recall": recall_score(yt, yp, **kw),
                "cm": confusion_matrix(yt, yp, labels=[0, 1, 2]), "n": len(yt),
                "n_unknown": int((yp == 2).sum())}

    def bundle(self):
        b = super().bundle()
        b.update({"unknown_detector": self.gate, "unknown_name": self.unknown_name,
                  "unknown_detector_name": self.unknown_detector})
        return b

    def summary(self):
        s = super().summary()
        s.update({"unknown_detector": self.unknown_detector,
                  "unknown_threshold": float(self.gate.threshold)})
        return s


class Task3BubbleLocation(BaseTask):
    key, title = "task3", "Task 3: Bubble location"
    class_names = LOC_COLS

    def target(self):
        bub = self.labels[self.labels["condition"].str.lower() == "anomaly"]
        is_yes = bub[LOC_COLS].apply(lambda c: c.str.lower() == "yes")
        loc = is_yes.apply(lambda r: "+".join(r.index[r]), axis=1)
        assert set(loc) <= set(LOC_COLS), f"unexpected bubble locations: {set(loc) - set(LOC_COLS)}"
        return {int(c): LOC_COLS.index(l) for c, l in zip(bub["case"], loc)}

    def make_split(self):
        """Per location: one recording to val, one to test, the rest to train."""
        rng = np.random.RandomState(self.seed)
        split = {"train": [], "val": [], "test": []}
        for k in range(self.n_classes):
            ids_k = sorted(i for i, v in self.y_of.items() if v == k)
            assert len(ids_k) >= 3, f"{LOC_COLS[k]} has only {len(ids_k)} recordings (need 3)"
            rng.shuffle(ids_k)
            split["train"] += ids_k[:-2]
            split["val"].append(ids_k[-2])
            split["test"].append(ids_k[-1])
        return split


class Task4FailedValve(BaseTask):
    key, title = "task4", "Task 4: Failed valve"
    class_names = SV_COLS

    def target(self):
        sv = self.labels[self.labels["condition"].str.lower() == "fault"]
        failed = sv[SV_COLS].lt(100)
        assert (failed.sum(axis=1) == 1).all(), "some fault recordings have 0 or several failed valves"
        return {int(c): SV_COLS.index(v) for c, v in zip(sv["case"], failed.idxmax(axis=1))}

    def make_split(self):
        ids = np.array(sorted(self.y_of))
        return self._stratified(ids, np.array([self.y_of[i] for i in ids]), (0.20, 0.20))


class Task5OpeningRatio(BaseTask):
    key, title = "task5", "Task 5: Opening ratio of the failed valve"
    kind, has_val = "regression", False
    save_models = (SAVE_REGRESSOR,)

    def target(self):
        sv = self.labels[self.labels["condition"].str.lower() == "fault"]
        failed = sv[SV_COLS].lt(100)
        assert (failed.sum(axis=1) == 1).all(), "some fault recordings have 0 or several failed valves"
        self.valve_of = {int(c): v for c, v in zip(sv["case"], failed.idxmax(axis=1))}
        ratio = sv[SV_COLS].where(failed).min(axis=1).astype(int)
        y = {int(c): int(r) for c, r in zip(sv["case"], ratio)}
        self.levels = sorted(set(y.values()))
        self.class_names = [f"{L}%" for L in self.levels]
        return y

    def make_split(self):                    # 70 / 30, no validation set
        ids = np.array(sorted(self.y_of))
        return self._stratified(ids, np.array([self.y_of[i] for i in ids]), (0, 0.30))

    def _windows(self, ids):
        d = super()._windows(ids)
        for v in SV_COLS:                    # failed valve as extra inputs (one-hot)
            d[f"valve_{v}"] = (d["recording_id"].map(self.valve_of) == v).astype(float)
        return d

    def _class_keys(self):
        return self.levels

    def _class_label(self, k):
        return f"{k}%"

    def snap(self, v):
        lv = np.array(self.levels)
        return lv[np.abs(np.asarray(v)[:, None] - lv[None, :]).argmin(axis=1)]

    def evaluate(self, split_name, pred_pct):
        """Mean of the recording's windows; snapped to the nearest level for acc / F1 / precision / recall."""
        d = self.F[split_name]
        m = pd.DataFrame({"recording_id": d["recording_id"].to_numpy(),
                          "pred": np.clip(pred_pct, self.levels[0], self.levels[-1])})
        rec = m.groupby("recording_id")["pred"].mean()
        yt = rec.index.map(self.y_of).to_numpy().astype(int)
        yp = rec.to_numpy(float)
        ys = self.snap(yp)
        kw = dict(labels=self.levels, average="macro", zero_division=0)
        return {"acc": accuracy_score(yt, ys), "f1": f1_score(yt, ys, **kw),
                "precision": precision_score(yt, ys, **kw), "recall": recall_score(yt, ys, **kw),
                "MAE": mean_absolute_error(yt, yp), "R2": r2_score(yt, yp),
                "cm": confusion_matrix(yt, ys, labels=self.levels), "n": len(yt)}

    def fit(self):
        for name, reg in make_regressors().items():
            reg.fit(self.X["train"], self.ytr / 100.0)          # target scaled to 0-1
            self.models[name] = reg
            self.preds[name] = {s: 100.0 * reg.predict(self.X[s]) for s in self.eval_splits}
            self.results[name] = {"type": "regression",
                                  **{s: self.evaluate(s, self.preds[name][s]) for s in self.eval_splits}}
        base = float(self.ytr.mean())                         # reference: training mean
        self.results["Baseline (train mean)"] = {
            "type": "baseline",
            **{s: self.evaluate(s, np.full(len(self.F[s]), base)) for s in self.eval_splits}}
        return self

    def bundle(self):
        b = super().bundle()
        b.update({"levels": self.levels, "valve_names": SV_COLS, "target_scale": 100.0})
        return b


# ----------------------------------------------------------------------
# Runs all five tasks
# ----------------------------------------------------------------------
class TaskTrainer:
    def __init__(self, pipe, model_dir="saved_models", seed=42, unknown_detector="KMeans(k=3)"):
        self.pipe = pipe
        self.feat_df = pipe.features_df if pipe.features_df is not None else pipe.load_features(verbose=False)
        self.labels = read_label_table(pipe.labels_path)
        self.model_dir = model_dir
        self.seed = seed
        self.unknown_detector = unknown_detector      # "PCA", "KMeans(k=3)" or "DBSCAN"
        self.tasks = {}

    def run_all(self, save=True, verbose=True):
        self._check_labels()
        t1 = Task1NormalAbnormal(self.feat_df, self.labels, self.seed).run()
        self.tasks = {
            "task1": t1,
            "task2": Task2FaultType(self.feat_df, self.labels, t1.split, self.seed,
                                    self.unknown_detector).run(),
            "task3": Task3BubbleLocation(self.feat_df, self.labels, self.seed).run(),
            "task4": Task4FailedValve(self.feat_df, self.labels, self.seed).run(),
            "task5": Task5OpeningRatio(self.feat_df, self.labels, self.seed).run(),
        }
        if verbose:
            for t in self.tasks.values():
                print(f"{t.title}: {sum(len(v) for v in t.split.values())} recordings | "
                      + ", ".join(f"{s}={len(v)}" for s, v in t.split.items()))
        if save:
            self.save()
        return self

    def _check_labels(self):
        rec = self.feat_df.groupby("recording_id")["label"].first()
        cond = self.labels.set_index("case")["condition"].str.lower()
        missing = sorted(set(rec.index) - set(cond.index))
        assert not missing, f"recordings without a row in the labels file: {missing[:5]}"
        bad = [i for i in rec.index if (cond[i] != "normal") != bool(rec[i])]
        assert not bad, f"feature labels disagree with the labels file for: {bad[:5]}"

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        for key, t in self.tasks.items():
            joblib.dump(t.bundle(), os.path.join(self.model_dir, f"{key}.joblib"))
        p = self.pipe
        settings = {"windows": [list(w) for w in p.windows_def], "sensors": list(p.sensors),
                    "fs": p.fs, "n_rows": p.n_rows, "drop_features": sorted(p.drop_features)}
        with open(os.path.join(self.model_dir, "feature_settings.json"), "w") as f:
            json.dump(settings, f, indent=2)
        joblib.dump(self.summary(), os.path.join(self.model_dir, "training_results.joblib"))
        print("Models saved to:", os.path.abspath(self.model_dir))

    def summary(self):
        p = self.pipe
        return {"tasks": {k: t.summary() for k, t in self.tasks.items()},
                "labels": self.labels,
                "feature_csv": os.path.abspath(p.feature_csv),
                "feature_settings": {"windows": [list(w) for w in p.windows_def], "sensors": list(p.sensors),
                                     "fs": p.fs, "n_rows": p.n_rows,
                                     "drop_features": sorted(p.drop_features)}}

    def show(self, confusions=True):
        """Results table (and test confusion matrices) for every task."""
        for t in self.tasks.values():
            print(f"\n===== {t.title} =====")
            display(t.split_counts())
            display(t.table())
            if t.key == "task2":
                print(f"Unknown detector: {t.unknown_detector}, threshold {t.gate.threshold:.3f}. "
                      "test_unknown = known recordings wrongly flagged Unknown.")
            if confusions:
                t.plot_confusions("test")
                plt.show()

    def saved_models_table(self):
        rows = []
        for key, t in self.tasks.items():
            for m in t.save_models:
                r = t.results[m]["test"]
                row = {"task": t.title, "model": m, "test_acc%": round(100 * r["acc"], 1),
                       "test_F1": round(r["f1"], 3), "test_precision": round(r["precision"], 3),
                       "test_recall": round(r["recall"], 3)}
                if "n_unknown" in r:
                    row["test_unknown"] = r["n_unknown"]
                if "MAE" in r:
                    row["test_MAE"] = round(r["MAE"], 1)
                rows.append(row)
        return pd.DataFrame(rows).set_index(["task", "model"])
