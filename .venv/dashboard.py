"""
dashboard.py
One HTML dashboard: dataset, extracted features, model results, performance measures
and (optionally) the predictions on the test data.

    from dashboard import Dashboard
    Dashboard(pipe=pipe, trainer=trainer, predictions=pred).build("dashboard.html")

Works from saved files too (new session, no retraining):
    Dashboard(model_dir="saved_models", predictions_csv="predictions_test.csv").build()

The file is self-contained (charts embedded as images, tabs in plain CSS): open it in any
browser, or it is shown inside the notebook when show=True.
"""
import os
import io
import base64
import html
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

META_COLS = ["recording_id", "window", "label", "label_name"]

# reference palette (categorical slots in fixed order) and text / surface tokens
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLASS_COLOR = {"Normal": C_BLUE, "Bubble": C_ORANGE, "SV fault": C_AQUA}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 1.4,
}

MEASURES = [
    ("Accuracy", "Share of recordings whose predicted class is correct."),
    ("Precision", "Of the recordings predicted as a class, the share that truly belong to it "
                  "(binary tasks: the positive class; multi-class: macro average over classes)."),
    ("Recall", "Of the recordings that truly belong to a class, the share that were found "
               "(binary: positive class; multi-class: macro average)."),
    ("F1 score", "Harmonic mean of precision and recall; low when either is low."),
    ("Unknown (Task 2)", "A window is Unknown when the stage-2a detector's score is above mean + 3 sigma of "
                         "the known abnormal training windows. Training data has no unknown faults, so "
                         "Unknown predictions on validation / test count as errors."),
    ("MAE (Task 5)", "Mean absolute error of the predicted opening ratio, in percentage points, "
                     "before rounding to the nearest level."),
    ("Recording level", "Each recording has 3 valve-opening windows. Classifiers vote over the windows; "
                        "the Task 5 regression averages them."),
    ("Validation / test", "Validation is used while choosing models; test is the held-out check. "
                          "Task 5 uses a 70/30 train/test split without validation."),
]


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig, alt):
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{_b64(fig)}"/>'


def _table(df, index=True):
    return df.to_html(index=index, border=0, classes="fd-table", na_rep="-")


def _cohens_d(a, b):
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2) / max(len(a) + len(b) - 2, 1))
    return 0.0 if sp < 1e-12 else (b.mean() - a.mean()) / sp


class Dashboard:
    def __init__(self, pipe=None, trainer=None, predictions=None, model_dir="saved_models",
                 predictions_csv=None):
        self.summary = trainer.summary() if trainer is not None else \
            joblib.load(os.path.join(model_dir, "training_results.joblib"))
        if pipe is not None and pipe.features_df is not None:
            self.feat_df = pipe.features_df
        else:
            self.feat_df = pd.read_csv(self.summary["feature_csv"])
        self.recordings = getattr(pipe, "recordings", {}) or {}
        self.labels = self.summary["labels"]
        self.settings = self.summary["feature_settings"]
        if predictions is None and predictions_csv and os.path.exists(predictions_csv):
            predictions = pd.read_csv(predictions_csv, index_col=0)
        self.pred = predictions
        self.fcols = [c for c in self.feat_df.columns if c not in META_COLS]
        cond = self.labels.set_index("case")["condition"].str.lower()
        name = {"normal": "Normal", "anomaly": "Bubble", "fault": "SV fault"}
        self.class_of = {int(c): name.get(v, v) for c, v in cond.items()}

    # ------------------------------------------------------------------
    def build(self, out_html="dashboard.html", show=True):
        with matplotlib.rc_context(STYLE):
            tabs = [("Overview", self._overview()), ("Dataset", self._dataset()),
                    ("Features", self._features()), ("Model results", self._models()),
                    ("Performance", self._performance())]
            if self.pred is not None:
                tabs.append(("Test predictions", self._predictions()))
        page = self._page(tabs)
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(page)
        print("Dashboard saved to:", os.path.abspath(out_html))
        if show:
            try:
                from IPython.display import HTML, display
                display(HTML(page))
            except ImportError:
                pass
        return os.path.abspath(out_html)

    # ------------------------------------------------------------------
    def _overview(self):
        n_rec = self.feat_df["recording_id"].nunique()
        n_win = len(self.feat_df)
        counts = pd.Series(self.class_of).value_counts()
        s = self.settings
        tiles = [("Recordings", n_rec), ("Windows", f"{n_win} ({n_win // max(n_rec, 1)} per rec.)"),
                 ("Sensors", len(s["sensors"])), ("Features", len(self.fcols)),
                 ("Normal", int(counts.get("Normal", 0))), ("Bubble (anomaly)", int(counts.get("Bubble", 0))),
                 ("SV fault", int(counts.get("SV fault", 0))),
                 ("Sampling rate", f"{s['fs']} Hz")]
        if self.pred is not None:
            tiles.append(("Test recordings", len(self.pred)))
        tile_html = "".join(f'<div class="fd-tile"><div class="fd-tile-v">{html.escape(str(v))}</div>'
                            f'<div class="fd-tile-k">{html.escape(k)}</div></div>' for k, v in tiles)

        best = self._saved_table()
        windows = ", ".join(f"{a}-{b} ms" for a, b in s["windows"])
        text = (f"<p>Each recording has {len(s['sensors'])} pressure sensors ({', '.join(s['sensors'])}) "
                f"sampled at {s['fs']} Hz. Features are computed on the valve-opening windows ({windows}), "
                f"one row per window. Five tasks are trained on these features; the saved models are listed "
                f"below with their test results (recording level).</p>")
        return f'<div class="fd-tiles">{tile_html}</div>{text}<h3>Saved models</h3>{_table(best)}'

    # ------------------------------------------------------------------
    def _dataset(self):
        out = []
        lab = self.labels.copy()
        lab["class"] = lab["case"].map(self.class_of)

        # recordings per class
        counts = lab["class"].value_counts().reindex(["Normal", "Bubble", "SV fault"]).fillna(0)
        fig, ax = plt.subplots(figsize=(5.2, 2.0))
        ax.barh(counts.index[::-1], counts.values[::-1], color=C_BLUE, height=0.55)
        for i, v in enumerate(counts.values[::-1]):
            ax.text(v + 0.5, i, int(v), va="center", color=INK, fontsize=8)
        ax.set_xlabel("Recordings"); ax.set_title("Recordings per class")
        ax.grid(axis="y", visible=False)
        out.append('<div class="fd-row">' + _img(fig, "Recordings per class"))

        # bubble location / SV fault structure
        bub = lab[lab["class"] == "Bubble"]
        if len(bub):
            locs = bub[["BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7", "BV1"]].apply(
                lambda c: c.str.lower() == "yes").sum()
            fig, ax = plt.subplots(figsize=(5.2, 2.0))
            ax.bar(locs.index, locs.values, color=C_ORANGE, width=0.6)
            ax.set_ylabel("Recordings"); ax.set_title("Bubble recordings per location")
            ax.grid(axis="x", visible=False)
            out.append(_img(fig, "Bubble recordings per location"))
        out.append("</div>")

        sv = lab[lab["class"] == "SV fault"].copy()
        if len(sv):
            failed = sv[["SV1", "SV2", "SV3", "SV4"]].lt(100)
            sv["valve"] = failed.idxmax(axis=1)
            sv["ratio"] = sv[["SV1", "SV2", "SV3", "SV4"]].where(failed).min(axis=1).astype(int)
            ct = pd.crosstab(sv["valve"], sv["ratio"].map(lambda r: f"{r}%"))
            out.append("<h3>SV-fault recordings: failed valve x opening ratio</h3>" + _table(ct))
        if "spacecraft" in lab.columns and lab["spacecraft"].notna().any():
            ct = pd.crosstab(lab["spacecraft"].astype(str), lab["class"])
            out.append("<h3>Recordings per spacecraft and class</h3>" + _table(ct))

        # raw signals + spectra (only when the recordings were loaded with pipe.run())
        if self.recordings:
            out.append("<h3>Example signals</h3>" + self._signals_fig())
            out.append("<h3>Mean amplitude spectrum of the valve-opening windows</h3>" + self._spectra_fig())
        else:
            out.append('<p class="fd-note">Raw signals are not shown: pass a pipe on which run() was called.</p>')
        return "".join(out)

    def _example_ids(self):
        ex = {}
        for cls in ["Normal", "Bubble", "SV fault"]:
            ids = sorted(i for i, c in self.class_of.items() if c == cls and i in self.recordings)
            if ids:
                ex[cls] = ids[0]
        return ex

    def _signals_fig(self):
        sensors = self.settings["sensors"]
        ex = self._example_ids()
        fs = self.settings["fs"]
        fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 1.25 * len(sensors)), sharex=True)
        for j, (ax, s) in enumerate(zip(np.atleast_1d(axes), sensors)):
            for cls, rid in ex.items():
                x = self.recordings[rid][:, j]
                t = np.arange(len(x)) * 1000.0 / fs
                ax.plot(t, x, color=CLASS_COLOR[cls], lw=0.9, label=f"{cls} (case {rid})")
            for a, b in self.settings["windows"]:
                ax.axvspan(a * 1000.0 / fs, b * 1000.0 / fs, color=GRID, alpha=0.8, lw=0)
            ax.set_ylabel(s, rotation=0, ha="right", va="center")
            ax.grid(axis="x", visible=False)
        np.atleast_1d(axes)[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3)
        np.atleast_1d(axes)[-1].set_xlabel("Time (ms) - shaded: valve-opening windows used for features")
        return _img(fig, "Example signals per sensor")

    def _spectra_fig(self):
        sensors = self.settings["sensors"]
        fs = self.settings["fs"]
        spec = {}
        for rid, arr in self.recordings.items():
            cls = self.class_of.get(rid)
            if cls is None:
                continue
            for a, b in self.settings["windows"]:
                seg = arr[a:b]
                w = np.hanning(len(seg))
                amp = (2.0 / w.sum()) * np.abs(np.fft.rfft((seg - seg.mean(axis=0)) * w[:, None], axis=0))
                spec.setdefault(cls, []).append(amp)
        freqs = np.fft.rfftfreq(self.settings["windows"][0][1] - self.settings["windows"][0][0], 1.0 / fs)
        ncol = 4
        nrow = int(np.ceil(len(sensors) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.3 * nrow), sharex=True, squeeze=False)
        axes = axes.ravel()
        for j, (ax, s) in enumerate(zip(axes, sensors)):
            for cls in ["Normal", "Bubble", "SV fault"]:
                if cls in spec:
                    ax.plot(freqs[1:], np.mean(spec[cls], axis=0)[1:, j], color=CLASS_COLOR[cls], label=cls)
            ax.set_title(s)
            ax.set_xlim(0, min(200, freqs[-1]))
        for ax in axes[len(sensors):]:
            ax.axis("off")
        axes[0].legend()
        for ax in axes[(nrow - 1) * ncol:]:
            ax.set_xlabel("Frequency (Hz)")
        fig.tight_layout()
        return _img(fig, "Mean amplitude spectra")

    # ------------------------------------------------------------------
    def _features(self):
        out = []
        sensors = self.settings["sensors"]
        types = list(dict.fromkeys(c.split("_", 1)[1] for c in self.fcols))
        dropped = self.settings.get("drop_features", [])
        out.append(f"<p>{len(self.fcols)} features = {len(types)} per sensor x {len(sensors)} sensors, "
                   f"per window. Types: <b>{', '.join(types)}</b>."
                   + (f" Left out: {', '.join(dropped)}." if dropped else "") + "</p>")

        df = self.feat_df.copy()
        df["class"] = df["recording_id"].map(self.class_of)
        norm, abn = df[df["label"] == 0], df[df["label"] == 1]

        # effect size per sensor x feature type (normal vs abnormal)
        d = pd.Series({c: _cohens_d(norm[c].to_numpy(float), abn[c].to_numpy(float)) for c in self.fcols})
        grid = pd.DataFrame(index=sensors, columns=types, dtype=float)
        for c, v in d.items():
            s, t = c.split("_", 1)
            grid.loc[s, t] = abs(v)
        fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(types), 0.5 + 0.42 * len(sensors)))
        im = ax.imshow(grid.to_numpy(float), cmap="Blues", aspect="auto", vmin=0)
        for i in range(len(sensors)):
            for j in range(len(types)):
                v = grid.iat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(grid.to_numpy(float)) / 2 else INK)
        ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=45, ha="right")
        ax.set_yticks(range(len(sensors))); ax.set_yticklabels(sensors)
        ax.grid(False)
        ax.set_title("How strongly each feature separates normal from abnormal windows (|Cohen's d|)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        out.append(_img(fig, "Effect size heatmap"))

        # top features
        top = d.abs().sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.barh(top.index[::-1], top.values[::-1], color=C_BLUE, height=0.6)
        ax.set_xlabel("|Cohen's d|  (normal vs abnormal)"); ax.set_title("Top 12 features")
        ax.grid(axis="y", visible=False)
        row = '<div class="fd-row">' + _img(fig, "Top 12 features")

        # PCA of all windows
        X = StandardScaler().fit_transform(df[self.fcols].to_numpy(float))
        pca = PCA(n_components=2).fit(X)
        Z = pca.transform(X)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for cls in ["Normal", "Bubble", "SV fault"]:
            m = (df["class"] == cls).to_numpy()
            if m.any():
                ax.scatter(Z[m, 0], Z[m, 1], s=14, color=CLASS_COLOR[cls], alpha=0.75,
                           edgecolors=SURFACE, linewidths=0.6, label=cls)
        ev = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
        ax.set_title("All windows, first two principal components")
        ax.legend()
        row += _img(fig, "PCA of windows") + "</div>"
        out.append(row)

        # class means of the top features
        means = df.groupby("class")[list(top.index)].mean().T
        means = means[[c for c in ["Normal", "Bubble", "SV fault"] if c in means.columns]]
        means.insert(0, "|d|", top.round(2))
        out.append("<h3>Class means of the top features (window level)</h3>" + _table(means.round(4)))
        return "".join(out)

    # ------------------------------------------------------------------
    def _models(self):
        out = []
        for key, t in self.summary["tasks"].items():
            out.append(f"<h3>{html.escape(t['title'])}</h3>")
            out.append('<p class="fd-note">Recordings per set: '
                       + _table(t["split_counts"]).replace("\n", "") + "</p>")
            out.append(_table(t["table"]))
            if t.get("unknown_detector"):
                note = (f"Stage 2a unknown detector: <b>{html.escape(t['unknown_detector'])}</b>, fitted on the "
                        f"known abnormal training windows; a window is Unknown above mean + 3 sigma of out-of-fold scores "
                        f"(threshold {t['unknown_threshold']:.3f}). The training data has no unknown faults, "
                        f"so <b>test_unknown</b> counts known recordings wrongly flagged as Unknown.")
                out.append(f'<p class="fd-note">{note}</p>')

            # accuracy and F1 per model (test)
            names = list(t["results"])
            acc = [t["results"][n]["test"]["acc"] for n in names]
            f1 = [t["results"][n]["test"]["f1"] for n in names]
            fig, ax = plt.subplots(figsize=(min(1.0 + 0.85 * len(names), 9.5), 2.6))
            x = np.arange(len(names))
            ax.bar(x - 0.19, acc, width=0.36, color=C_BLUE, label="Accuracy")
            ax.bar(x + 0.19, f1, width=0.36, color=C_ORANGE, label="F1")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylim(0, 1.05); ax.set_ylabel("Test score")
            ax.grid(axis="x", visible=False)
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax.set_title("Test set, recording level")
            out.append(_img(fig, f"{t['title']} scores"))

            # confusion matrices of the saved models
            saved = t["saved_models"]
            k = len(t["class_names"])
            size = 2.8 + 0.32 * k
            fig, axes = plt.subplots(1, len(saved), figsize=(size * len(saved), size), squeeze=False)
            for ax, m in zip(axes[0], saved):
                r = t["results"][m]["test"]
                self._cm(ax, r["cm"], t["class_names"], f"{m}  (acc {100 * r['acc']:.0f}%)")
            fig.suptitle("Saved models - test confusion matrices", fontsize=10)
            fig.tight_layout()
            out.append(_img(fig, f"{t['title']} confusion matrices"))
        return "".join(out)

    @staticmethod
    def _cm(ax, cm, labels, title):
        k = len(labels)
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10 if k <= 4 else 7,
                        color="white" if cm.max() and cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=90 if k > 4 else 0, fontsize=7)
        ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(title, fontsize=9)

    # ------------------------------------------------------------------
    def _saved_table(self):
        rows = []
        for key, t in self.summary["tasks"].items():
            for m in t["saved_models"]:
                r = t["results"][m]["test"]
                row = {"task": t["title"], "model": m, "test recordings": r["n"],
                       "accuracy %": round(100 * r["acc"], 1), "F1": round(r["f1"], 3),
                       "precision": round(r["precision"], 3), "recall": round(r["recall"], 3),
                       "wrongly Unknown": r.get("n_unknown"),
                       "MAE (pp)": round(r["MAE"], 1) if "MAE" in r else None}
                rows.append(row)
        return pd.DataFrame(rows).set_index(["task", "model"])

    def _performance(self):
        out = ["<h3>Performance measures used</h3>",
               _table(pd.DataFrame(MEASURES, columns=["measure", "meaning"]), index=False)
               .replace('class="dataframe fd-table"', 'class="dataframe fd-table fd-text"')]
        tab = self._saved_table()
        out.append("<h3>Saved models on the test set</h3>" + _table(tab))

        # F1 of the saved models per task
        tasks = list(self.summary["tasks"].values())
        labels = [t["title"].split(":")[0] for t in tasks]
        series = {}
        for t in tasks:
            for i, m in enumerate(t["saved_models"]):
                slot = m if t["kind"] == "classification" else "Ridge"
                series.setdefault(slot, {})[t["title"].split(":")[0]] = t["results"][m]["test"]["f1"]
        names = list(series)
        colors = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        x = np.arange(len(labels))
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            vals = [series[n].get(l, np.nan) for l in labels]
            ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w * 0.92, color=colors[i % 4], label=n)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Test F1")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title("Test F1 of the saved models per task")
        out.append(_img(fig, "Saved models F1"))
        out.append('<p class="fd-note">Test sets are small (see "test recordings"), so one recording '
                   'can move a score by several points. Task 3 has one test recording per location.</p>')
        return "".join(out)

    # ------------------------------------------------------------------
    def _predictions(self):
        p = self.pred
        out = []
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.4))
        parts = [("condition", "Task 1: condition"), ("fault_type", "Task 2: fault type"),
                 ("bubble_location", "Task 3: bubble location"), ("failed_valve", "Task 4: failed valve")]
        for ax, (col, title) in zip(axes, parts):
            v = p[col][p[col] != "-"].value_counts().sort_index()
            ax.bar(v.index.astype(str), v.values, color=C_BLUE, width=0.6)
            ax.set_title(title)
            ax.grid(axis="x", visible=False)
            if len(v) > 4:
                ax.tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("Recordings")
        fig.tight_layout()
        out.append(_img(fig, "Prediction counts"))
        r = p["opening_ratio_%"][p["opening_ratio_%"].astype(str) != "-"]
        if len(r):
            ct = pd.crosstab(p.loc[r.index, "failed_valve"], r.astype(str) + "%")
            out.append("<h3>Task 5: predicted opening ratio per failed valve</h3>" + _table(ct))
        out.append(f"<h3>All {len(p)} test recordings</h3>" + _table(p))
        return "".join(out)

    # ------------------------------------------------------------------
    @staticmethod
    def _page(tabs):
        radios, labels, panels, css_rules = [], [], [], []
        for i, (name, body) in enumerate(tabs):
            rid = f"fd-tab-{i}"
            radios.append(f'<input type="radio" name="fd-tabs" id="{rid}" {"checked" if i == 0 else ""}/>')
            labels.append(f'<label for="{rid}">{html.escape(name)}</label>')
            panels.append(f'<section class="fd-panel" id="fd-panel-{i}">{body}</section>')
            css_rules.append(f"#{rid}:checked ~ .fd-panels #fd-panel-{i} {{display:block}}"
                             f" #{rid}:checked ~ .fd-bar label[for={rid}] {{color:{INK};border-color:{C_BLUE}}}")
        css = f"""
.fd-dash {{font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color:{INK};
  background:{SURFACE}; padding:16px 20px; max-width:1200px}}
.fd-dash h1 {{font-size:20px; margin:0 0 12px}}
.fd-dash h3 {{font-size:14px; margin:22px 0 8px; color:{INK}}}
.fd-dash p {{color:{INK2}; font-size:13px; line-height:1.5; max-width:900px}}
.fd-dash .fd-note {{color:{MUTED}; font-size:12px}}
.fd-dash input[name=fd-tabs] {{display:none}}
.fd-dash .fd-bar {{display:flex; gap:4px; border-bottom:1px solid {GRID}; margin-bottom:14px; flex-wrap:wrap}}
.fd-dash .fd-bar label {{padding:8px 14px; cursor:pointer; color:{INK2}; font-size:13px;
  border-bottom:2px solid transparent; margin-bottom:-1px}}
.fd-dash .fd-panel {{display:none}}
.fd-dash img {{max-width:100%; height:auto; margin:6px 0}}
.fd-dash .fd-row {{display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start}}
.fd-dash .fd-row img {{max-width:calc(50% - 10px); min-width:320px}}
.fd-dash .fd-tiles {{display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px}}
.fd-dash .fd-tile {{border:1px solid {GRID}; border-radius:8px; padding:10px 14px; min-width:120px; background:#fff}}
.fd-dash .fd-tile-v {{font-size:22px; font-weight:600; color:{INK}}}
.fd-dash .fd-tile-k {{font-size:12px; color:{INK2}; margin-top:2px}}
.fd-dash table.fd-table {{border-collapse:collapse; font-size:12px; margin:4px 0 10px; background:#fff}}
.fd-dash table.fd-table th, .fd-dash table.fd-table td {{padding:4px 10px; border-bottom:1px solid {GRID};
  text-align:right; color:{INK}}}
.fd-dash table.fd-table th {{color:{INK2}; font-weight:600; background:#f4f3f0}}
.fd-dash .fd-note table.fd-table {{display:inline-table; vertical-align:middle; margin-left:6px}}
.fd-dash table.fd-text td, .fd-dash table.fd-text th {{text-align:left}}
{chr(10).join(css_rules)}
"""
        return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Fault diagnosis dashboard</title>"
                f"<style>{css}</style></head><body style='margin:0;background:{SURFACE}'>"
                f"<div class='fd-dash'><h1>Valve and bubble fault diagnosis - dashboard</h1>"
                f"{''.join(radios)}<div class='fd-bar'>{''.join(labels)}</div>"
                f"<div class='fd-panels'>{''.join(panels)}</div></div></body></html>")
