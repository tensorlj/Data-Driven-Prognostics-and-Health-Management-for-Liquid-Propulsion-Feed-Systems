"""
fault_predictor.py
Runs the saved Task 1-5 models on a folder of NEW recordings and writes the results
into the test labels Excel file. No training happens here.

    from fault_predictor import FaultPredictor

    predictor = FaultPredictor("saved_models")
    pred = predictor.run(test_data, test_labels)     # features -> Tasks 1-5 -> labels file + CSV

Order of the tasks for each recording
    Task 1  Normal / Abnormal
    Task 2  abnormal -> Bubble (Anomaly) / SV fault (Fault) / Unknown
            (stage 2a: unknown detector fitted on known abnormal windows; stage 2b: classifier)
    Task 3  Bubble   -> location BP1-BP7, BV1
    Task 4  SV fault -> failed valve SV1-SV4
    Task 5  SV fault -> opening ratio of that valve (uses the Task 4 valve)

Written to the labels file (same vocabulary as the training labels):
    Condition  Normal / Anomaly / Fault / Unknown
    SV1-SV4    100, except the failed valve = predicted opening ratio
    BP1-BV1    Yes at the predicted bubble location, otherwise No
"""
import os
import re
import glob
import json
import shutil
import joblib
import numpy as np
import pandas as pd

from feature_engineering import FeatureEngineering

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

TASKS = ["task1", "task2", "task3", "task4", "task5"]
SV_COLS = ["SV1", "SV2", "SV3", "SV4"]
LOC_COLS = ["BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7", "BV1"]
# words written to the Condition column (change here if your file expects other words)
CONDITION_WORD = {"Normal": "Normal", "Bubble": "Anomaly", "SV fault": "Fault", "Unknown": "Unknown"}


def vote3(rec_ids, pred):
    """Task 2: majority vote over Bubble / SV fault / Unknown; no unique majority -> Unknown."""
    t = pd.DataFrame({"recording_id": np.asarray(rec_ids), "pred": np.asarray(pred).astype(int)})

    def v(p):
        cnt = np.bincount(p, minlength=3)
        top = np.flatnonzero(cnt == cnt.max())
        return int(top[0]) if len(top) == 1 else 2
    return t.groupby("recording_id")["pred"].agg(v)


def majority_vote(rec_ids, pred):
    t = pd.DataFrame({"recording_id": np.asarray(rec_ids), "pred": np.asarray(pred).astype(int)})
    return t.groupby("recording_id")["pred"].agg(lambda p: int(np.bincount(p).argmax()))


class FaultPredictor:
    def __init__(self, model_dir="saved_models", use=None):
        """use: model per classification task, e.g. {"task1": "ANN", "task4": "LogReg"};
        choices are "ANN", "GradBoost", "LogReg". Task 5 always uses Ridge."""
        self.model_dir = model_dir
        self.B = {t: joblib.load(os.path.join(model_dir, f"{t}.joblib")) for t in TASKS}
        with open(os.path.join(model_dir, "feature_settings.json")) as f:
            self.settings = json.load(f)
        self.use = {"task1": "ANN", "task2": "ANN", "task3": "ANN", "task4": "ANN"}
        self.use.update(use or {})
        for t, m in self.use.items():
            assert m in self.B[t]["models"], f"{t}: no saved model '{m}' (have {list(self.B[t]['models'])})"
        self.use["task5"] = next(iter(self.B["task5"]["models"]))
        s = self.settings
        self.fe = FeatureEngineering(".", "", windows=[tuple(w) for w in s["windows"]],
                                     sensors=s["sensors"], fs=s["fs"], n_rows=s["n_rows"],
                                     drop_features=s["drop_features"])
        self.features_df = None
        self.predictions = None

    # ---------- features of the new recordings ----------
    def features(self, data_dir):
        """Window features for every CSV in the folder (same settings as training)."""
        files = sorted(glob.glob(os.path.join(data_dir.rstrip("/\\"), "*.csv")))
        assert files, f"no CSV files in {data_dir}"
        fe = self.fe
        need = max(b for _, b in fe.windows_def)
        rows = []
        for f in files:
            m = re.search(r"(\d+)", os.path.basename(f))
            assert m, f"{f}: no case number in the file name"
            rid = int(m.group(1))
            d = pd.read_csv(f)
            d.columns = [c.strip().upper() for c in d.columns]
            assert all(s in d.columns for s in fe.sensors), f"{f}: sensor columns missing"
            arr = d[fe.sensors].to_numpy(float)[:fe.n_rows]
            assert len(arr) >= need, f"{f}: only {len(arr)} rows, need {need}"
            for w, (a, b) in enumerate(fe.windows_def):
                row = {"recording_id": rid, "window": w, "file": os.path.basename(f)}
                row.update(fe.window_features(arr[a:b]))
                rows.append(row)
        df = pd.DataFrame(rows)
        assert df.groupby("recording_id")["file"].nunique().max() == 1, "two files with the same case number"
        fc = [c for c in df.columns if c not in ("recording_id", "window", "file")]
        df[fc] = np.nan_to_num(df[fc].to_numpy(float))
        self.features_df = df
        return df

    # ---------- models ----------
    def _X(self, task, df):
        b = self.B[task]
        missing = [c for c in b["fcols"] if c not in df.columns]
        assert not missing, f"{task}: features missing in the new data, e.g. {missing[:3]}"
        return b["scaler"].transform(df[b["fcols"]])

    def _classify(self, task, df):
        model = self.B[task]["models"][self.use[task]]
        return majority_vote(df["recording_id"], model.predict(self._X(task, df)))

    def _task2(self, df):
        """Stage 2a: unknown gate; stage 2b: Bubble / SV fault classifier."""
        b = self.B["task2"]
        X = self._X("task2", df)
        pred = b["models"][self.use["task2"]].predict(X).astype(int)
        gate = b["unknown_detector"]
        pred[gate.score(X) > gate.threshold] = 2
        return vote3(df["recording_id"], pred)

    def predict(self, data_dir):
        """One row per recording with the result of every task ("-" where a task doesn't apply)."""
        df = self.features(data_dir)
        out = df.groupby("recording_id")[["file"]].first()
        for c in ["condition", "fault_type", "bubble_location", "failed_valve", "opening_ratio_%"]:
            out[c] = pd.Series("-", index=out.index, dtype=object)

        t1 = self._classify("task1", df)
        out["condition"] = t1.map(dict(enumerate(self.B["task1"]["classes"])))
        abn = df[df["recording_id"].map(t1) == 1]
        if len(abn):
            t2 = self._task2(abn)
            names2 = dict(enumerate(self.B["task2"]["classes"] + [self.B["task2"]["unknown_name"]]))
            out.loc[t2.index, "fault_type"] = t2.map(names2)      # Unknown recordings stop here

            bub = abn[abn["recording_id"].map(t2) == 0]
            if len(bub):
                t3 = self._classify("task3", bub)
                out.loc[t3.index, "bubble_location"] = t3.map(dict(enumerate(self.B["task3"]["classes"])))

            svf = abn[abn["recording_id"].map(t2) == 1]
            if len(svf):
                t4 = self._classify("task4", svf)
                out.loc[t4.index, "failed_valve"] = t4.map(dict(enumerate(self.B["task4"]["classes"])))

                b5 = self.B["task5"]
                d5 = svf.copy()
                for i, v in enumerate(b5["valve_names"]):           # valve from the Task 4 prediction
                    d5[f"valve_{v}"] = (d5["recording_id"].map(t4) == i).astype(float)
                pct = b5["target_scale"] * b5["models"][self.use["task5"]].predict(self._X("task5", d5))
                lv = np.array(b5["levels"])
                rec = (pd.Series(np.clip(pct, lv[0], lv[-1]), index=d5["recording_id"].to_numpy())
                       .groupby(level=0).mean())
                snapped = lv[np.abs(rec.to_numpy()[:, None] - lv[None, :]).argmin(axis=1)]
                out.loc[rec.index, "opening_ratio_%"] = snapped.astype(int)

        # label-file vocabulary
        out["label_condition"] = [CONDITION_WORD["Normal"] if c == "Normal" else CONDITION_WORD[f]
                                  for c, f in zip(out["condition"], out["fault_type"])]
        self.predictions = out
        return out

    # ---------- write into the labels Excel file ----------
    def write_labels(self, labels_path, predictions=None, out_path=None, backup=True):
        """Fill Condition, SV1-SV4 and BP1-BV1 for every case in the labels file.
        The file keeps its layout; missing columns are added after the last column.
        With backup=True the untouched file is copied once to <name>_original.xlsx."""
        import openpyxl
        pred = self.predictions if predictions is None else predictions
        assert pred is not None, "run predict() first"
        path = labels_path.rstrip("/\\")
        out_path = out_path or path
        if backup and os.path.abspath(out_path) == os.path.abspath(path):
            stem, ext = os.path.splitext(path)
            if not os.path.exists(stem + "_original" + ext):
                shutil.copy2(path, stem + "_original" + ext)

        wb = openpyxl.load_workbook(path)
        ws = wb.worksheets[0]
        # header cells: "Case..." and the target names, searched in the first 30 rows
        targets = ["Condition"] + SV_COLS + LOC_COLS
        pos, case_pos = {}, None
        for r in range(1, min(ws.max_row, 30) + 1):
            for c in range(1, ws.max_column + 1):
                v = ws.cell(r, c).value
                if not isinstance(v, str):
                    continue
                v = v.strip().lower()
                if case_pos is None and v.startswith("case"):
                    case_pos = (r, c)
                for t in targets:
                    if v == t.lower() and t not in pos:
                        pos[t] = (r, c)
        assert case_pos is not None, "no 'Case' column found in the labels file"
        hdr_row = max([case_pos[0]] + [r for r, _ in pos.values()])
        for t in targets:                                    # add missing columns
            if t not in pos:
                c = ws.max_column + 1
                ws.cell(case_pos[0], c).value = t
                pos[t] = (case_pos[0], c)

        written, not_in_data = [], []
        for r in range(hdr_row + 1, ws.max_row + 1):
            v = ws.cell(r, case_pos[1]).value
            if v is None:
                continue
            m = re.search(r"(\d+)", str(v).split(".")[0] if isinstance(v, float) else str(v))
            if not m:
                continue
            cid = int(m.group(1))
            if cid not in pred.index:
                not_in_data.append(cid)
                continue
            p = pred.loc[cid]
            ws.cell(r, pos["Condition"][1]).value = p["label_condition"]
            for sv in SV_COLS:
                ratio = p["opening_ratio_%"] if p["failed_valve"] == sv else 100
                ws.cell(r, pos[sv][1]).value = int(ratio) if ratio != "-" else 100
            for loc in LOC_COLS:
                ws.cell(r, pos[loc][1]).value = "Yes" if p["bubble_location"] == loc else "No"
            written.append(cid)
        wb.save(out_path)

        not_in_file = sorted(set(pred.index) - set(written))
        print(f"Labels written for {len(written)} cases -> {os.path.abspath(out_path)}")
        if not_in_data:
            print(f"  {len(not_in_data)} cases in the labels file have no data file (left empty): {not_in_data[:10]}")
        if not_in_file:
            print(f"  {len(not_in_file)} data files have no row in the labels file: {not_in_file[:10]}")
        return out_path

    # ---------- everything in one call ----------
    def run(self, data_dir, labels_path, out_csv="predictions_test.csv", out_labels=None, verbose=True):
        pred = self.predict(data_dir)
        pred.to_csv(out_csv)
        self.write_labels(labels_path, pred, out_path=out_labels)
        if verbose:
            print(f"{len(pred)} recordings | models: {self.use} | unknown detector: "
              f"{self.B['task2']['unknown_detector_name']}")
            print("Predictions table saved to:", os.path.abspath(out_csv))
            print("\nCondition:", pred["label_condition"].value_counts().to_dict())
            display(pred)
        return pred
