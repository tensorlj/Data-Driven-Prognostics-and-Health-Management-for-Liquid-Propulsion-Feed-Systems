"""
feature_engineering.py
Feature engineering: valve-opening windows (first 100 ms of each 400 ms cycle).

    pipe = FeatureEngineering(training_data, training_labels)
    feat_df = pipe.run()                 # load + check + slice + features + save CSV, one call
    feat_df = pipe.load_features()       # later: just load the saved CSV into a dataframe
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

META_COLS = ["recording_id", "window", "label", "label_name"]


# ----------------------------------------------------------------------
class FeatureEngineering:
    def __init__(self, data_dir, labels_path, feature_csv="features_sliced.csv",
                 windows=((0, 100), (400, 500), (800, 900)),
                 sensors=("P1", "P2", "P3", "P4", "P5", "P6", "P7"),
                 fs=1000, n_rows=1200,
                 drop_features=("rms", "fpeak1", "fpeak2", "fpeak3", "crest")):
        self.data_dir = data_dir
        self.labels_path = labels_path.rstrip("/\\")      # tolerate a trailing slash
        self.feature_csv = feature_csv
        self.windows_def = list(windows)                  # half-open row ranges [start, end)
        self.sensors = list(sensors)
        self.fs, self.n_rows = fs, n_rows
        self.drop_features = set(drop_features)           # feature types left out of the output
        self.label_map = {"normal": 0, "fault": 1, "anomaly": 1}

        self.body = None            # labels table
        self.recordings = {}        # {case_id: array (n_rows, 7)}
        self.label_of = {}          # {case_id: 0/1}
        self.windows = {}           # {(case_id, window_index): array (100, 7)}
        self.features_df = None
        self.feature_cols = []

    # ---------- step 1: labels, file checks, load recordings ----------
    def load(self, verbose=True):
        raw = pd.read_excel(self.labels_path, header=None)
        norm = raw.astype(str).apply(lambda col: col.str.strip().str.lower())
        hdr_row = norm.index[(norm == "condition").any(axis=1)][0]
        cond_col = norm.columns[norm.loc[hdr_row] == "condition"][0]
        case_col = norm.columns[norm.loc[hdr_row].str.startswith("case")][0]

        body = raw.loc[hdr_row + 1:, [case_col, cond_col]].copy()
        body.columns = ["case", "condition"]
        body["case"] = pd.to_numeric(body["case"], errors="coerce")
        body = body.dropna(subset=["case"]).reset_index(drop=True)
        body["case"] = body["case"].astype(int)
        body["condition"] = body["condition"].astype(str).str.strip()

        cond_lower = body["condition"].str.lower()
        unknown = set(cond_lower) - set(self.label_map)
        assert not unknown, f"Unexpected Condition values: {unknown}"
        assert body["case"].is_unique, "Duplicate case numbers in labels file"
        body["label"] = cond_lower.map(self.label_map)
        body["file"] = body["case"].map(lambda c: os.path.join(self.data_dir, f"Case{c:03d}.csv"))

        missing = body[~body["file"].map(os.path.exists)]
        assert len(missing) == 0, f"{len(missing)} CSV files missing, e.g.\n{missing.head()}"

        recordings, bad_time = {}, []
        for cid, f in zip(body["case"], body["file"]):
            d = pd.read_csv(f)
            d.columns = [c.strip().upper() for c in d.columns]
            assert all(s in d.columns for s in self.sensors), f"Case {cid}: sensor columns missing"
            arr = d[self.sensors].to_numpy(float)[:self.n_rows]
            assert arr.shape == (self.n_rows, len(self.sensors)), f"Case {cid}: shape {arr.shape}"
            assert not np.isnan(arr).any(), f"Case {cid}: NaNs found"
            if "TIME" in d.columns and not np.allclose(
                    np.diff(d["TIME"].to_numpy(float)[:self.n_rows]), 0.001, atol=1e-6):
                bad_time.append(cid)
            recordings[cid] = arr

        self.body, self.recordings = body, recordings
        self.label_of = dict(zip(body["case"], body["label"]))

        if verbose:
            print("Recordings in labels file:", len(body))
            print(body["condition"].value_counts().to_string())
            print("\nBinary labels (0 = normal, 1 = abnormal):")
            print(body["label"].value_counts().sort_index().to_string())
            print(f"\nLoaded {len(recordings)} recordings, each {next(iter(recordings.values())).shape}")
            print("Recordings with uneven TIME spacing:", bad_time if bad_time else "none")
        return self

    # ---------- step 2: slice into windows ----------
    def slice_windows(self, verbose=True):
        assert self.recordings, "Call load() first"
        self.windows = {}
        for rid, arr in self.recordings.items():
            for w, (a, b) in enumerate(self.windows_def):
                self.windows[(rid, w)] = arr[a:b]
        if verbose:
            shape = next(iter(self.windows.values())).shape
            print(f"Windows: {len(self.windows)} = {len(self.recordings)} recordings "
                  f"x {len(self.windows_def)} windows, each {shape}")
        return self

    # ---------- step 3: features (17 computed per sensor, minus the dropped types) ----------
    @staticmethod
    def _stat_features(x):
        rms = np.sqrt(np.mean(x ** 2))
        mean_abs = np.mean(np.abs(x))
        peak = np.max(np.abs(x))
        sqrt_mean = np.mean(np.sqrt(np.abs(x)))
        div = lambda a, b: a / b if b > 1e-12 else 0.0
        return {
            "rms": rms, "mean": np.mean(x), "var": np.var(x),
            "kurt": np.nan_to_num(kurtosis(x)), "skew": np.nan_to_num(skew(x)),
            "p2p": np.ptp(x), "energy": np.sum(x ** 2),
            "crest": div(peak, rms), "shape": div(rms, mean_abs),
            "clearance": div(peak, sqrt_mean ** 2), "impulse": div(peak, mean_abs),
        }

    def _fft_features(self, x):
        n = len(x)
        w = np.hanning(n)
        amp = (2.0 / np.sum(w)) * np.abs(np.fft.rfft((x - x.mean()) * w))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        a, f = amp[1:], freqs[1:]                      # skip 0 Hz
        pk, _ = find_peaks(a)                          # distinct local peaks
        top = pk[np.argsort(a[pk])[::-1][:3]]          # 3 largest
        out = {}
        for k in range(3):
            out[f"fpeak{k+1}"] = f[top[k]] if k < len(top) else 0.0
            out[f"apeak{k+1}"] = a[top[k]] if k < len(top) else 0.0
        return out

    def window_features(self, seg):
        """seg: (n_samples, 7) -> dict of features (17 per sensor minus the dropped ones)."""
        row = {}
        for j, s in enumerate(self.sensors):
            x = seg[:, j]
            for name, val in {**self._stat_features(x), **self._fft_features(x)}.items():
                if name in self.drop_features:
                    continue
                row[f"{s}_{name}"] = val
        return row

    def extract_features(self, save=True, verbose=True):
        assert self.windows, "Call slice_windows() first"
        rows = []
        for (rid, w), seg in self.windows.items():
            row = {"recording_id": rid, "window": w, "label": int(self.label_of[rid]),
                   "label_name": "abnormal" if self.label_of[rid] == 1 else "normal"}
            row.update(self.window_features(seg))
            rows.append(row)
        df = pd.DataFrame(rows).sort_values(["recording_id", "window"]).reset_index(drop=True)
        fc = [c for c in df.columns if c not in META_COLS]
        df[fc] = np.nan_to_num(df[fc].to_numpy(float))
        self.features_df, self.feature_cols = df, fc
        if verbose:
            per_sensor = len(fc) // len(self.sensors)
            print(f"features_df shape: {df.shape} | {len(fc)} features "
                  f"({per_sensor} per sensor x {len(self.sensors)} sensors)")
            if self.drop_features:
                print("Dropped feature types:", sorted(self.drop_features))
        if save:
            df.to_csv(self.feature_csv, index=False)
            print("Saved to:", os.path.abspath(self.feature_csv))
        return df

    # ---------- everything in one call ----------
    def run(self, save=True, verbose=True):
        """load -> check -> slice -> features -> save CSV. Returns the features dataframe."""
        self.load(verbose=verbose)
        self.slice_windows(verbose=verbose)
        return self.extract_features(save=save, verbose=verbose)

    # ---------- load the saved features into a dataframe ----------
    def load_features(self, verbose=True):
        df = pd.read_csv(self.feature_csv)
        self.features_df = df
        self.feature_cols = [c for c in df.columns if c not in META_COLS]
        if verbose:
            print("Shape:", df.shape, "| features:", len(self.feature_cols))
            print("\nWindows per label:")
            print(df["label_name"].value_counts().to_string())
            print("\nWindows per recording:", sorted(df.groupby("recording_id").size().unique()))
            print("NaN values:", int(df[self.feature_cols].isnull().sum().sum()))
            display(df.head())
        return df

    # ---------- viewing features together with their labels ----------
    def _ensure_features(self):
        if self.features_df is None:
            self.load_features(verbose=False)
        return self.features_df

    def _select_columns(self, sensors=None, feature_types=None):
        """Feature columns filtered by sensor (e.g. ['P1','P3']) and/or type (e.g. ['rms','var'])."""
        cols = self.feature_cols
        if sensors is not None:
            sensors = [sensors] if isinstance(sensors, str) else list(sensors)
            cols = [c for c in cols if c.split("_", 1)[0] in sensors]
        if feature_types is not None:
            feature_types = [feature_types] if isinstance(feature_types, str) else list(feature_types)
            cols = [c for c in cols if c.split("_", 1)[1] in feature_types]
        return cols

    def show(self, n=10, recording_id=None, label=None, sensors=None, feature_types=None):
        """
        Features together with their labels.
          recording_id : one id or a list of ids
          label        : 0 / 1 or 'normal' / 'abnormal'
          sensors      : e.g. 'P6' or ['P1', 'P6']
          feature_types: e.g. 'rms' or ['rms', 'p2p', 'fpeak1']
          n            : number of rows to show (None = all)
        """
        d = self._ensure_features()
        if recording_id is not None:
            ids = [recording_id] if np.isscalar(recording_id) else list(recording_id)
            d = d[d["recording_id"].isin(ids)]
        if label is not None:
            if isinstance(label, str):
                d = d[d["label_name"] == label.lower()]
            else:
                d = d[d["label"] == label]
        cols = self._select_columns(sensors, feature_types)
        out = d[META_COLS + cols]
        out = out if n is None else out.head(n)
        display(out)
        return out

    def label_counts(self):
        """Number of windows and recordings per label."""
        d = self._ensure_features()
        out = d.groupby("label_name").agg(windows=("recording_id", "size"),
                                          recordings=("recording_id", "nunique"))
        display(out)
        return out

    def recording_table(self):
        """One row per recording: id, label and the original Condition (Normal / Fault / Anomaly)."""
        d = self._ensure_features()
        t = d[["recording_id", "label", "label_name"]].drop_duplicates().reset_index(drop=True)
        if self.body is not None:
            t = t.merge(self.body[["case", "condition"]], left_on="recording_id", right_on="case",
                        how="left").drop(columns="case")
        display(t)
        return t

    def summary_by_label(self, stat="mean", sensors=None, feature_types=None):
        """Feature value per class (rows = features, columns = normal / abnormal), plus the ratio."""
        d = self._ensure_features()
        cols = self._select_columns(sensors, feature_types)
        g = d.groupby("label_name")[cols].agg(stat).T
        if {"normal", "abnormal"} <= set(g.columns):
            g["abnormal/normal"] = g["abnormal"] / g["normal"].replace(0, np.nan)
        display(g)
        return g
