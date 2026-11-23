# Data-Driven Prognostics and Health Management for Liquid Propulsion Feed Systems

A machine-learning pipeline for fault diagnosis and severity assessment of a spacecraft
propulsion feed system, built from multi-channel pressure time-series data.

Course project for **ME F321: Data Mining in Mechanical Sciences**, BITS Pilani, Pilani Campus.

| | |
|---|---|
| **Authors** | Lakshya Jagnani (2023B2A40912P), Vikhyaat Sharma (2023B2A40936P)
| **Instructor** | Dr. Madhurjya Dev Choudhury |
| **Repository** | [tensorlj/Data-Driven-Prognostics-and-Health-Management-for-Liquid-Propulsion-Feed-Systems](https://github.com/tensorlj/Data-Driven-Prognostics-and-Health-Management-for-Liquid-Propulsion-Feed-Systems) |
| **Report** | [`Project_Report.pdf`](Project_Report.pdf) · **Slides** [`_Data_Mining_Project_PPT.pptx.pdf`](_Data_Mining_Project_PPT.pptx.pdf) · **Brief** [`MEF321_ProjectDescription_2026.pdf`](MEF321_ProjectDescription_2026.pdf) |

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Methodology / Workflow](#methodology--workflow)
5. [Detailed Explanation of Each Stage](#detailed-explanation-of-each-stage)
6. [Project Structure and File Guide](#project-structure-and-file-guide)
7. [Technologies and Libraries Used](#technologies-and-libraries-used)
8. [Experiments, Evaluation and Results](#experiments-evaluation-and-results)
9. [Outputs and Visualizations](#outputs-and-visualizations)
10. [References](#references)

---

## Overview

Prognostics and Health Management (PHM) matters for spacecraft propulsion because incipient
faults and unit-to-unit variability make simple threshold monitoring unreliable. In the
benchmark rig modelled here, water pressurised to **2.0 MPa** is delivered through a branched
manifold to four thruster solenoid valves (**SV1–SV4**), and transient pressure is recorded by
seven transducers (**P1–P7**) at **1 kHz for 1.2 s**. Each run performs three successive
valve open–close cycles of 400 ms, producing water-hammer spikes followed by damped acoustic
ringing.

Two physical fault mechanisms are present in the data:

- **Bubble contamination** — a localised gas bubble at one of eight feedline nodes
  (`BV1`, `BP1`–`BP7`). Bubbles change the local speed of sound and shift reflection points,
  subtly distorting the pressure traces.
- **Solenoid valve fault** — a valve stuck partially open (an opening ratio strictly between
  0% and 100%), which throttles flow and alters pressure behaviour.

The test set additionally contains **unknown anomalies** and data from a **previously unseen
spacecraft unit (No. 4)**, so the pipeline must both generalise across units and refuse to
force novel faults into a known class.

This repository implements the full cascade as reusable Python classes: signal slicing →
feature engineering → five diagnostic tasks → prediction on new recordings → an HTML
dashboard. All five tasks are implemented, trained, evaluated and saved.

## Objectives

The pipeline solves five tasks in sequence, each operating only on recordings that the
previous task routed to it:

| # | Task | Type | Scope | Output |
|---|------|------|-------|--------|
| 1 | **Anomaly detection** | Binary classification | All recordings | Normal / Abnormal |
| 2 | **Fault-type classification** | 3-way (2 classes + novelty gate) | Abnormal recordings | Bubble / SV fault / Unknown |
| 3 | **Bubble localisation** | 8-class classification | Bubble recordings | `BP1`–`BP7`, `BV1` |
| 4 | **Failed-valve identification** | 4-class classification | SV-fault recordings | `SV1`–`SV4` |
| 5 | **Fault severity estimation** | Regression | SV-fault recordings | Opening ratio (%) of the failed valve |


## Dataset

Source: a public benchmark for fault diagnosis of numerical spacecraft propulsion systems
(see [References](#references)). The copy used here lives under `Dataset/`.

**Signal files** — each `CaseXXX.csv` has 8 `float64` columns (`TIME`, `P1`–`P7`) and
**1201 rows**, spanning `TIME` = 0.000–1.200 s in 0.001 s steps. The pipeline reads the
first **1200** rows (`n_rows=1200`), giving exactly three 400-sample cycles.

| Split | Files | Labels | Contents |
|-------|-------|--------|----------|
| **Train** | `Case001.csv`–`Case177.csv` (177) | `labels.xlsx` | Spacecraft 1–3, 59 runs each (35 normal + 24 abnormal) |
| **Test** | `Case178.csv`–`Case223.csv` (46) | `labels_spacecraft.xlsx` | 23 runs from Spacecraft 1 + 23 from the unseen Spacecraft 4, including unknown anomaly types |

**Verified training-label distribution** (via `read_label_table`):
`Normal` = 105, `Fault` (SV fault) = 48, `Anomaly` (bubble) = 24 → 177 total.

**Label schema** (`labels.xlsx`, two header rows that the loader detects automatically by
searching for the `Condition` cell):

| Column(s) | Meaning |
|---|---|
| `Case#` | Recording id, joined to `CaseXXX.csv` by zero-padded number |
| `Spacecraft#` | Unit 1–4 |
| `Condition` | `Normal` / `Anomaly` (bubble) / `Fault` (SV fault) |
| `SV1`–`SV4` | Opening ratio in %; `100` = healthy, `<100` = the failed valve and its severity |
| `BP1`–`BP7`, `BV1` | `Yes` at the bubble location, otherwise `No` |

The class-defining structure is **implicit in these numbers**, and each task derives its own
target from them: Task 4 takes `argmin` over the `SV*` columns (asserting exactly one valve
is `<100`), Task 5 takes that valve's percentage, and Task 3 takes the single `Yes` column.

**Measured actuation timing** (independent check on `Case001.csv`, sensor `P4`): pressure
collapses from 2.0 MPa at row 12, and water-hammer peaks (≈5.28 MPa) occur at rows
107–112, 507–512 and 907–912 — i.e. a 400-row period. The windows `[0,100)`, `[400,500)`,
`[800,900)` therefore cover the valve-open flow phase of each cycle and stop short of the
closing water hammer, which is what the "valve-opening window" naming refers to. (The course
brief states a nominal open-at-100-ms / close-at-300-ms schedule; the recordings' own time
base is offset from that, so the window indices follow the measured transitions.)

> **Note on `Dataset/test/`:** `labels_spacecraft.xlsx` in this repository **contains the
> pipeline's written predictions** — `fault_predictor.py` fills in `Condition`, `SV1`–`SV4`
> and `BP1`–`BV1`, adding those columns since the original file had only `Case#` and
> `Spacecraft#`. The untouched original is preserved as
> `labels_spacecraft_original.xlsx`. Ground truth for the test set is not distributed, so
> all metrics in this README come from the held-out split of the *training* data.

---

## Methodology / Workflow

**Execution order of the modules:** `sensor_analysis` (optional, exploratory) →
`feature_engineering` → `fault_tasks` → `fault_predictor` → `dashboard`.
`feature_engineering.py` is the only module imported by another (`fault_predictor.py`
re-instantiates it to guarantee identical feature computation at inference time).

---

## Detailed Explanation of Each Stage

### Stage 0 — Exploratory Data Analysis

**File:** [`.venv/sensor_analysis.py`](.venv/sensor_analysis.py) · **Driven from:** `.venv/file.ipynb` (cell 3)

*Why:* before trusting any model, confirm the data is uniform and find out which part of the
signal actually carries fault information.

| Class | Input | What it does | Output |
|---|---|---|---|
| `DatasetInspector` | A folder of CSVs | Counts files, reads headers only (fast), checks every file has identical columns in identical order, builds a per-file missing-value summary | Console report + `DataFrame` |
| `SignalExplorer` | One recording `DataFrame` | `describe()`/`info()` overview, per-sensor line plots, correlation heatmap (seaborn, with a matplotlib fallback), box plots | Matplotlib figures |
| `FFTAnalyzer` | One recording, optional `t_start_ms`/`t_end_ms` segment | Hanning-windowed, DC-removed, amplitude-normalised `rfft`; reports N, f<sub>s</sub>, Nyquist and frequency resolution; warns if `TIME` is unevenly spaced; extracts top-*n* distinct local peaks; `scipy.signal.spectrogram`; `overlay()` compares two recordings on a shared frequency axis | Spectra, peak table, spectrogram, normal-vs-abnormal overlay |

*Findings that shaped the design:* all 223 files share one schema with **zero missing
values**; every run shows the three-cycle pulse structure with sharp drops on opening and
water-hammer spikes on closing (up to 5.22 MPa on `P4` in nominal `Case008`); and the FFT
overlay separates abnormal from normal runs clearly, with the dominant peak shifting from
**≈66 Hz to ≈63 Hz**. That frequency sensitivity is why spectral features are included
alongside statistical ones.

### Stage 1 — Signal slicing

**File:** [`.venv/feature_engineering.py`](.venv/feature_engineering.py) → `FeatureEngineering.load()`, `.slice_windows()`

*Why:* a full 1.2 s recording mixes steady state, opening transient, water hammer and
ringdown. Restricting features to the most diagnostic regime raises accuracy and improves
transfer to unseen spacecraft.

Three slicing schemes were evaluated (report §3.1):

| Scheme | Window of each 400 ms cycle | Content |
|---|---|---|
| Case 1 | Full cycle, 0–400 ms | Steady state + opening + closing |
| **Case 2 (selected)** | **First 100 ms** | **Valve-open flow phase** |
| Case 3 | Last 300 ms | Closure, water-hammer surge, damped ringing |

**Case 2 won** on diagnostic performance and cross-spacecraft generalisation, so each
recording contributes **3 windows** of 100 samples × 7 sensors — row ranges
`(0,100)`, `(400,500)`, `(800,900)`, configurable via the `windows` constructor argument.

`load()` performs the integrity gate before any slicing, and fails loudly rather than
silently skipping bad data:

- locates the `Condition` header cell to find the real header row, then parses `Case#`;
- asserts every `Condition` value is recognised and every `Case#` is unique;
- asserts each referenced `CaseXXX.csv` exists, has all 7 sensor columns, trims to exactly
  (1200, 7), and contains **no NaNs**;
- checks `TIME` spacing is 0.001 s within `1e-6` and reports any offending recordings.

**Inputs:** training CSV folder + `labels.xlsx` → **Outputs:** `self.recordings`
(`{case_id: (1200, 7) array}`), `self.label_of` (`{case_id: 0|1}`), `self.windows`
(`{(case_id, window_idx): (100, 7) array}`).

### Stage 2 — Feature engineering

**File:** [`.venv/feature_engineering.py`](.venv/feature_engineering.py) → `_stat_features()`, `_fft_features()`, `window_features()`, `extract_features()`

*Why:* classical classifiers need fixed-length vectors, and physically interpretable
time- and frequency-domain descriptors generalise better on 177 recordings than a raw
100 × 7 waveform would.

**17 features are computed per sensor per window** — 11 statistical + 6 spectral:

| Group | Features |
|---|---|
| Amplitude / energy | `rms`, `mean`, `var`, `p2p`, `energy` |
| Distribution shape | `kurt`, `skew` (via `scipy.stats`, NaN-guarded) |
| Dimensionless shape factors | `crest` = peak/rms, `shape` = rms/mean\|x\|, `clearance` = peak/(mean√\|x\|)², `impulse` = peak/mean\|x\| |
| FFT peaks | `fpeak1`–`fpeak3` (frequencies), `apeak1`–`apeak3` (amplitudes) |

Implementation details that matter: all shape factors go through a guarded division
(`0.0` when the denominator is < 1e-12) so flat segments cannot produce `inf`; the FFT is
Hanning-windowed, mean-subtracted and amplitude-scaled by `2/Σw`; the 0 Hz bin is skipped and
peaks are taken as **distinct local maxima** (`scipy.signal.find_peaks`) rather than the top-3
bins of one broad lobe, which would otherwise return three neighbouring points of the same
peak.

**Outputs:** `features_sliced.csv` — verified **531 rows × 88 columns** = (177 recordings ×
3 windows) × (4 metadata + 84 feature columns), with metadata
`recording_id, window, label, label_name`.

### Stage 3 — Feature selection

**Files:** feature list applied in [`.venv/feature_engineering.py`](.venv/feature_engineering.py) (`drop_features`); persisted to `saved_models/feature_settings.json`

*Why:* 119 features (17 × 7) over 177 recordings invites overfitting, and several features are
near-duplicates of each other.

A two-step selection was performed (report §3.3):

1. **Relevance — mutual information** against the normal/abnormal label. Lowest-scoring
   features were dropped: `fpeak1` (0.0605), `fpeak2` (0.1288), `fpeak3` (0.1970). The
   retained set includes `kurt` (0.4025), `apeak2` (0.4022) and `apeak3` (0.3956).
2. **Redundancy — pairwise correlation.** `crest` was removed for near-perfect correlation
   with `impulse` (|r| = 0.9991) alongside a lower MI score (0.3158); `rms` was likewise
   dropped as collinear.

Net effect: **119 → 84 features** (12 per sensor × 7 sensors), while leaving the Task 1
results unchanged. The 12 retained types are
`mean, var, kurt, skew, p2p, energy, shape, clearance, impulse, apeak1, apeak2, apeak3`.

> **Implemented vs. analysed:** the *outcome* of this analysis is what the code enforces —
> `drop_features=("rms", "fpeak1", "fpeak2", "fpeak3", "crest")` is a constructor default,
> and the 5 types are filtered out inside `window_features()`. The mutual-information and
> correlation computations themselves are **not** part of the committed modules or notebooks;
> the scores above are quoted from the report. Re-deriving them is listed under
> [Future Scope](#limitations-and-future-scope).

### Stage 4 — Model training and evaluation

**File:** [`.venv/fault_tasks.py`](.venv/fault_tasks.py)

*Why:* the five tasks share one evaluation protocol, so they share one base class; only the
target definition and splitting policy differ per task.

#### Shared protocol — `BaseTask`

| Step | Method | Detail |
|---|---|---|
| Target | `target()` | Subclass-specific `{recording_id: y}`, derived from the labels table |
| Split | `make_split()` | **By recording**, never by window; `prepare()` asserts no recording appears in two sets |
| Feature pruning | `prepare()` | Columns with ≈zero variance **on the training split** (std ≤ 1e-12) are dropped |
| Scaling | `prepare()` | `StandardScaler` **fitted on training windows only**, then applied to val/test |
| Fit | `fit()` | All 7 supervised models; plus 3 unsupervised scorers when `unsupervised = True` |
| Aggregate | `majority_vote()` / `post()` | Per-recording majority vote over its 3 windows |
| Score | `evaluate()` | Recording-level accuracy, F1, precision, recall + confusion matrix; binary tasks use class 1 as positive, multi-class uses macro averaging |
| Persist | `bundle()` | Saves chosen models **with** their scaler, their exact feature-column list and class names |

**Model zoo** (`make_supervised`, `make_regressors`, `make_unsupervised`):

- *Supervised:* `DecisionTree` (depth 5, balanced), `LogReg` (C=10, balanced, 1000 iters),
  `NaiveBayes` (uniform priors), `KNN(k=3)`, `ANN` (`MLPClassifier`, hidden 64→32, ReLU,
  α=1e-3), `RandomForest` (300 trees, balanced), `GradBoost` (100 stages, lr 0.1, depth 3).
- *Unsupervised novelty scorers*, each fitted on class-0 windows only, thresholded at
  mean + 3σ of its training scores: `PCAScorer` (reconstruction error at 95% retained
  variance), `KMeansScorer` (distance to nearest of k=3 centres), `DBSCANScorer` (distance to
  nearest core point, ε = 90th percentile of k-distances, with leave-self-out scoring for
  core points so training scores aren't biased to zero).
- *Regressors (Task 5):* `LinearRegression`, `Ridge` (α=10), `DecisionTreeRegressor`
  (depth 5), plus a train-mean baseline.

#### Per-task specifics

| Class | Target derivation | Split policy |
|---|---|---|
| `Task1NormalAbnormal` | `Condition == Normal` → 0, else 1 | Stratified 60/20/20 over all 177 recordings |
| `Task2FaultType` | `Fault` → 1 (SV fault), `Anomaly` → 0 (bubble) | **Inherits Task 1's split**, filtered to abnormal recordings — so a recording never moves between sets across tasks |
| `Task3BubbleLocation` | Index of the single `Yes` among `BP1`–`BP7`, `BV1` | Per location: 1 → val, 1 → test, rest → train (asserts ≥3 recordings per location) |
| `Task4FailedValve` | Index of the single `SV*` column `< 100` | Stratified 60/20/20 over 48 recordings |
| `Task5OpeningRatio` | That valve's opening percentage | Stratified **70/30, no validation set** |

**Task 2's two-stage novelty gate** is the most intricate part of the pipeline:

- *Stage 2a — Unknown detector.* A scorer (default `KMeans(k=3)`) is fitted on known abnormal
  training windows. Its threshold is mean + 3σ of **out-of-fold** scores from a
  `GroupKFold(n_splits=5)` grouped **by recording**, so each window is scored by a detector
  that never saw its own recording. This matters: in-sample scores are systematically too low,
  and calibrating on them would flag many known recordings as Unknown. The final gate
  (threshold **14.238**) is then refit on all known abnormal windows.
- *Stage 2b — Classifier.* Windows judged "known" are labelled Bubble vs SV fault.
- *Voting.* `vote3()` aggregates over three outcomes and resolves **ties to Unknown** —
  a deliberately conservative choice for novelty rejection.
- Because the training data contains no unknown faults, **every Unknown on val/test is by
  definition an error**; the `test_unknown` column counts known recordings wrongly flagged.

**Task 5 details:** the failed valve is one-hot encoded (4 extra columns) and concatenated
with the 84 pressure features → **88 inputs**. The target is scaled to 0–1 for fitting and
rescaled by 100 for reporting. Window predictions are **averaged** (not voted) per recording,
clipped to the observed range, and `snap()`-ped to the nearest of the four discrete levels
(0/25/50/75%) for accuracy and F1 — while MAE and R² are computed on the **unsnapped**
continuous prediction.

**Outputs** (`TaskTrainer.save()` → `saved_models/`): `task1.joblib`…`task5.joblib`
(models + scaler + feature list + class names), `feature_settings.json` (the exact windowing
and feature configuration, so inference cannot drift), and `training_results.joblib`
(picklable metrics, split membership and tables for the dashboard).

`TaskTrainer._check_labels()` runs first and asserts that every recording in the feature
matrix has a labels row and that the binary feature labels agree with the labels file.

### Stage 5 — Inference on new recordings

**File:** [`.venv/fault_predictor.py`](.venv/fault_predictor.py) → `FaultPredictor`

*Why:* the test set must be scored by a cascade that applies each task only to the recordings
routed to it, with **no training and no label access**.

1. Load the five bundles and `feature_settings.json`, then rebuild a `FeatureEngineering`
   instance **from those saved settings** — the mechanism that guarantees identical features
   at train and inference time. Each requested model name is asserted to exist in its bundle.
2. `features()` globs `*.csv`, recovers each `recording_id` from digits in the filename,
   validates sensor columns and row count, and computes the same 84 features per window.
   It also asserts two files cannot claim the same case number.
3. `predict()` runs the cascade, writing `"-"` wherever a task does not apply:
   Task 1 → abnormal subset → Task 2 (Unknown recordings stop here) → bubble subset to
   Task 3, SV-fault subset to Task 4 → Task 5 using **Task 4's predicted valve** as its
   one-hot input.
4. `_X()` re-asserts that every feature column a bundle was trained on exists in the new data
   before transforming with the saved scaler.
5. `write_labels()` fills the Excel file in place using `openpyxl`: it finds header cells
   within the first 30 rows, appends any missing target columns, writes
   `Normal`/`Anomaly`/`Fault`/`Unknown` to `Condition`, sets the failed valve's cell to the
   predicted ratio (others `100`), marks `Yes`/`No` across `BP1`–`BV1`, and backs the original
   up once to `<name>_original.xlsx`. It reports cases present in only one of the two sources
   rather than failing.

Default models: **ANN** for Tasks 1–4 (override per task via `use=`, choosing among `ANN`,
`GradBoost`, `LogReg`), **Ridge** for Task 5.

**Outputs:** `predictions_test.csv` (one row per recording: `condition`, `fault_type`,
`bubble_location`, `failed_valve`, `opening_ratio_%`, `label_condition`) — verified at
**46 rows**, one per test recording — plus the filled `labels_spacecraft.xlsx`.

### Stage 6 — Dashboard

**File:** [`.venv/dashboard.py`](.venv/dashboard.py) → `Dashboard`


| Tab | Contents |
|---|---|
| **Overview** | Headline tiles (recordings, windows, sensors, features, class counts, sampling rate) + the saved-model results table |
| **Dataset** | Recordings per class, bubble recordings per location, failed valve × opening ratio crosstab, recordings per spacecraft, example waveforms with the feature windows shaded, mean amplitude spectra per class |
| **Features** | \|Cohen's *d*\| heatmap (sensor × feature type) for normal vs abnormal separation, top-12 features, 2-component PCA scatter of all windows, class means of the top features |
| **Model results** | Per task: split counts, the full 10-model table, accuracy/F1 bar chart, confusion matrices of the saved models |
| **Performance** | Plain-language glossary of every metric used, saved-model summary table, test-F1-per-task chart, and an explicit small-sample caveat |
| **Test predictions** | Prediction counts per task, predicted opening ratio per valve, and the full 46-recording table (only when predictions are supplied) |

It can run **without retraining** by reading `saved_models/training_results.joblib` and
`predictions_test.csv` — though raw-signal plots require a `pipe` on which `run()`/`load()`
was called, since recordings are not pickled.

---

## Project Structure and File Guide

> ### ⚠️ Read this first: the source code lives in `.venv/`
>
> All Python modules, notebooks and generated artifacts are committed inside a directory named
> **`.venv/`**. Despite the name, **it is not a virtual environment** — it contains no
> `Scripts/`, `Lib/` or `pyvenv.cfg`, only project source. Two consequences:
>
> - **Do not run `python -m venv .venv` or `uv venv` in the project root** — it would write
>   into the same directory as the source code.
> - Many tools and `.gitignore` templates exclude `.venv/` by default, so treat it as the
>   source root rather than ignorable output.
>
> Renaming it to `src/` would be an improvement, but this README documents the repository
> **as it stands** so that every path below resolves. If you do rename it, the only changes
> needed are the directory you `cd` into — the modules import each other by bare module name
> and resolve `saved_models/` relative to the current working directory.

```
Course_Project/
├── README.md                                 # This file
├── pyproject.toml                            # Project metadata (incomplete — see Setup)
│
├── Project_Report.pdf                        # 5-page report: methodology, results, references
├── _Data_Mining_Project_PPT.pptx.pdf         # Slides: per-task result tables, confusion matrices
├── MEF321_ProjectDescription_2026.pdf        # Course brief: rig description, tasks, deliverables
├── Statement of contributions.pdf            # Team contribution statement
│
├── Dataset/
│   ├── train/
│   │   ├── data/Case001.csv … Case177.csv    # 177 training recordings, 1201 × (TIME, P1–P7)
│   │   └── labels.xlsx                       # Condition, SV1–SV4 %, BP1–BP7/BV1 Yes/No
│   └── test/
│       ├── data/Case178.csv … Case223.csv    # 46 test recordings (Spacecraft 1 + unseen 4)
│       ├── labels_spacecraft.xlsx            # ⚠️ contains WRITTEN PREDICTIONS
│       └── labels_spacecraft_original.xlsx   # Untouched original (Case# + Spacecraft# only)
│
└── .venv/                                    # ⚠️ SOURCE ROOT (not a virtualenv)
    ├── sensor_analysis.py                    # Stage 0 — EDA classes
    ├── feature_engineering.py                # Stages 1–3 — slicing + features (+ drop list)
    ├── fault_tasks.py                        # Stage 4 — Tasks 1–5 training & evaluation
    ├── fault_predictor.py                    # Stage 5 — cascade inference on new data
    ├── dashboard.py                          # Stage 6 — self-contained HTML dashboard
    │
    ├── testing.ipynb                         # ✅ CANONICAL 5-cell pipeline driver
    ├── file.ipynb                            # Exploratory notebook (pre-refactor)
    │
    ├── features_sliced.csv                   # Generated — 531 × 88 feature matrix
    ├── predictions_test.csv                  # Generated — 46 test predictions
    ├── dashboard.html                        # Generated — the dashboard
    ├── split*.json                            # Stale artifacts from file.ipynb (unused)
    └── saved_models/
        ├── task1.joblib … task4.joblib       # ANN + GradBoost + LogReg, scaler, fcols, classes
        ├── task5.joblib                      # Ridge + levels + valve names + target scale
        ├── feature_settings.json             # Windows, sensors, fs, n_rows, drop_features
        └── training_results.joblib           # Metrics, split membership, tables (dashboard input)
```

### What each file is for, and when you need it

| File | Contains | Used at | Connects to |
|---|---|---|---|
| **`feature_engineering.py`** | `FeatureEngineering` — label parsing, integrity assertions, window slicing, the 17-feature extractor, the `drop_features` list, plus inspection helpers (`show`, `label_counts`, `recording_table`, `summary_by_label`) | Stages 1–3; **always first** | The only module imported by another — `fault_predictor.py` rebuilds it from `feature_settings.json`. Writes `features_sliced.csv`, consumed by `fault_tasks.py` and `dashboard.py` |
| **`fault_tasks.py`** | `read_label_table`, the model zoo, the three novelty scorers, `BaseTask` + `Task1`–`Task5`, and `TaskTrainer` (`run_all`, `save`, `show`, `saved_models_table`) | Stage 4 — training | Takes a `FeatureEngineering` instance; writes everything in `saved_models/` |
| **`fault_predictor.py`** | `FaultPredictor` — loads bundles, recomputes features, runs the 1→2→3/4→5 cascade, writes the Excel labels file | Stage 5 — inference | Reads `saved_models/`; imports `feature_engineering`; writes `predictions_test.csv` + `labels_spacecraft.xlsx` |
| **`dashboard.py`** | `Dashboard` — 6 tabs, matplotlib style tokens, Cohen's *d* and PCA diagnostics, HTML/CSS page builder | Stage 6 — reporting | Accepts live `pipe`/`trainer`/`predictions`, or reads `training_results.joblib` + `predictions_test.csv` for a no-retrain rebuild |
| **`sensor_analysis.py`** | `DatasetInspector`, `SignalExplorer`, `FFTAnalyzer` | Stage 0 — EDA | Standalone; imports nothing from the other modules. Its findings justify the slicing and spectral features |
| **`testing.ipynb`** | **Start here.** 5 cells: paths → imports → train+save → predict → dashboard | Running the whole project | Drives all four modules in order |
| **`file.ipynb`** | The development notebook: EDA, then Task 1–5 logic written inline before being refactored into classes, plus save/predict prototypes | Historical reference | Superseded by the modules; its per-task cells are the ancestors of `fault_tasks.py`. Uses **absolute Windows paths** that must be edited to run |
| **`split*.json`** | Recording-id splits (`split`, `split_bubble`, `split_valve`, `split_ratio`) dumped by `file.ipynb` | — | **Not read by any module.** Splits are now regenerated deterministically from `seed=42`. Safe to ignore or delete |
| **`feature_settings.json`** | `windows`, `sensors`, `fs`, `n_rows`, `drop_features` | Stage 5 | The train/inference contract — prevents feature drift between training and prediction |
| **`training_results.joblib`** | `tasks` (per-task metrics, tables, split membership), `labels`, `feature_csv` path, `feature_settings` | Stage 6 | Lets the dashboard rebuild with no retraining |

**Where to look if you want to…**

| Goal | File and entry point |
|---|---|
| Change which part of the signal is used | `feature_engineering.py` → `windows=` constructor arg |
| Add or remove a feature | `feature_engineering.py` → `_stat_features()` / `_fft_features()`, and `drop_features` |
| Add a candidate model | `fault_tasks.py` → `make_supervised()` / `make_regressors()` |
| Change which models get saved | `fault_tasks.py` → `SAVE_CLASSIFIERS` / `SAVE_REGRESSOR`, or a task's `save_models` |
| Re-tune the Unknown gate | `fault_tasks.py` → `Task2FaultType.fit()`, `thr_3sigma()`, or `TaskTrainer(unknown_detector=…)` |
| Add a sixth task | Subclass `BaseTask`, implement `target()` + `make_split()`, register in `TaskTrainer.run_all()` |
| Pick a different model at inference | `FaultPredictor(use={"task1": "GradBoost", …})` |
| Change the words written to Excel | `fault_predictor.py` → `CONDITION_WORD` |
| Add a dashboard panel | `dashboard.py` → a `_tabname()` method + an entry in `build()`'s `tabs` list |

---

## Technologies and Libraries Used

| Library | Used for |
|---|---|
| **numpy** | Array maths, FFT (`np.fft.rfft`), Hanning window, statistics |
| **pandas** | Label tables, feature matrix, grouping/voting, CSV & Excel I/O |
| **scipy** | `stats.kurtosis`, `stats.skew`, `signal.find_peaks`, `signal.spectrogram` |
| **scikit-learn** | `StandardScaler`, `PCA`, `KMeans`, `DBSCAN`, `NearestNeighbors`, `DecisionTree*`, `LogisticRegression`, `GaussianNB`, `KNeighborsClassifier`, `MLPClassifier`, `RandomForest`, `GradientBoosting`, `LinearRegression`, `Ridge`, `train_test_split`, `GroupKFold`, metrics |
| **matplotlib** | All figures, confusion matrices, dashboard charts |
| **joblib** | Model-bundle and results serialisation |
| **openpyxl** | Writing predictions into the labels workbook in place (via `pandas.read_excel` too) |
| **seaborn** | Correlation heatmap in EDA — *optional*, falls back to matplotlib |
| **IPython / Jupyter** | Notebook drivers; `display()` degrades to `print()` outside notebooks |

Every module guards its notebook-only imports, so **all four can be imported and used from
plain Python**, not just inside Jupyter.

## Experiments, Evaluation and Results

### Protocol

- Splits are made **at recording level** (60/20/20; Task 5 uses 70/30 with no validation
  set), stratified by class, with `seed = 42` and an assertion that no recording leaks across
  sets.
- Features are z-scored with a scaler **fitted on training windows only**.
- Every metric is **recording-level**: the 3 window predictions are combined by majority vote
  (mean for Task 5's regression) *before* scoring.
- Binary tasks report F1/precision/recall for the positive class; multi-class tasks report
  macro averages.

All figures below were regenerated from `saved_models/training_results.joblib` and confirmed
by a fresh end-to-end training run.

### Task 1 — Normal vs Abnormal · 36 test recordings

Split: 106 train (63 normal / 43 abnormal), 35 val, 36 test (21 / 15).

| Model | Type | Val acc % | Val F1 | **Test acc %** | **Test F1** | Precision | Recall |
|---|---|---|---|---|---|---|---|
| PCA | unsupervised | 94.3 | 0.933 | 97.2 | 0.968 | 0.938 | 1.000 |
| K-Means (k=3) | unsupervised | 85.7 | 0.783 | 97.2 | 0.966 | 1.000 | 0.933 |
| DBSCAN | unsupervised | 85.7 | 0.783 | 97.2 | 0.966 | 1.000 | 0.933 |
| Decision Tree | supervised | 88.6 | 0.833 | 88.9 | 0.846 | 1.000 | 0.733 |
| **Logistic Regression** ✔ | supervised | 91.4 | 0.880 | **94.4** | **0.929** | 1.000 | 0.867 |
| Naive Bayes | supervised | 88.6 | 0.833 | 97.2 | 0.966 | 1.000 | 0.933 |
| KNN (k=3) | supervised | 88.6 | 0.833 | 100.0 | 1.000 | 1.000 | 1.000 |
| **ANN** ✔ | supervised | 91.4 | 0.880 | **94.4** | **0.929** | 1.000 | 0.867 |
| Random Forest | supervised | 85.7 | 0.783 | 100.0 | 1.000 | 1.000 | 1.000 |
| **Gradient Boosting** ✔ | supervised | 88.6 | 0.833 | **94.4** | **0.929** | 1.000 | 0.867 |

✔ = carried forward into the saved pipeline.

**Why not KNN or Random Forest, despite 100% test accuracy?** They score *lower* on
validation (88.6% and 85.7%) than on test — a val→test gap in the wrong direction, which on a
36-recording test set signals luck rather than skill. ANN, Logistic Regression and Gradient
Boosting were selected on validation stability, a small val–test gap, and architectural
diversity (a neural net, a linear model and an ensemble, whose errors are less likely to
correlate). All three reach **94.4% accuracy, F1 = 0.929, with perfect precision** — no
healthy recording is ever flagged as faulty, and the 2 errors are missed abnormal runs.

Notably, the purely **unsupervised** detectors are competitive here (97.2%), which is
practically valuable: anomaly detection on this rig does not strictly require labelled faults.

### Task 2 — Bubble vs SV fault vs Unknown · 15 test recordings

Split (inherited from Task 1, abnormal only): 43 train (16 bubble / 27 SV), 14 val, 15 test
(5 / 10). Positive class = SV fault. Unknown detector = `KMeans(k=3)`, threshold 14.238.

| Model | Type | Val acc % | **Test acc %** | **Test F1** | Precision | Recall | Wrongly Unknown |
|---|---|---|---|---|---|---|---|
| PCA | unsupervised | 100.0 | 93.3 | 0.952 | 0.909 | 1.00 | 0 |
| K-Means (k=3) | unsupervised | 42.9 | 73.3 | 0.750 | 1.000 | 0.60 | 0 |
| DBSCAN | unsupervised | 28.6 | 66.7 | 0.667 | 1.000 | 0.50 | 0 |
| Decision Tree | supervised | 100.0 | 100.0 | 1.000 | 1.000 | 1.00 | 0 |
| **Logistic Regression** ✔ | supervised | 100.0 | **100.0** | **1.000** | 1.000 | 1.00 | 0 |
| Naive Bayes | supervised | 85.7 | 100.0 | 1.000 | 1.000 | 1.00 | 0 |
| KNN (k=3) | supervised | 100.0 | 100.0 | 1.000 | 1.000 | 1.00 | 0 |
| **ANN** ✔ | supervised | 100.0 | **100.0** | **1.000** | 1.000 | 1.00 | 0 |
| Random Forest | supervised | 100.0 | 100.0 | 1.000 | 1.000 | 1.00 | 0 |
| **Gradient Boosting** ✔ | supervised | 100.0 | **100.0** | **1.000** | 1.000 | 1.00 | 0 |

The two fault mechanisms are **almost perfectly separable** — all seven supervised models
reach 100% on test, and six of them on validation too. Just as importantly, the
**`Wrongly Unknown` column is 0 everywhere**: the out-of-fold-calibrated novelty gate does not
misfire on known faults, which is exactly what it was tuned to avoid. Its ability to catch
*genuinely* novel anomalies cannot be measured here, because the training distribution
contains none (see [Limitations](#limitations-and-future-scope)).

### Task 3 — Bubble location, 8 classes · 8 test recordings

Split: 24 bubble recordings, exactly **one per location** in each of train / val / test.
Macro-averaged metrics.

| Model | Val acc % | **Test acc %** | **Test F1** | Precision | Recall |
|---|---|---|---|---|---|
| Decision Tree | 87.5 | 87.5 | 0.833 | 0.812 | 0.875 |
| **Logistic Regression** ✔ | 100.0 | **100.0** | **1.000** | 1.000 | 1.000 |
| Naive Bayes | 100.0 | 100.0 | 1.000 | 1.000 | 1.000 |
| KNN (k=3) | 100.0 | 100.0 | 1.000 | 1.000 | 1.000 |
| **ANN** ✔ | 100.0 | **100.0** | **1.000** | 1.000 | 1.000 |
| Random Forest | 100.0 | 100.0 | 1.000 | 1.000 | 1.000 |
| **Gradient Boosting** ✔ | 100.0 | **100.0** | **1.000** | 1.000 | 1.000 |

**8 / 8 test recordings localised correctly** by six of seven models — bubble position leaves
a very distinctive multi-sensor signature. Read with care: with **one recording per class per
split** (8 train, 8 val, 8 test), a single error moves accuracy by 12.5 points. This is the
least statistically robust result in the project despite being the most perfect-looking.

### Task 4 — Failed valve SV1–SV4 · 10 test recordings

Split: 48 SV-fault recordings → 28 train (7 per valve), 10 val, 10 test. Macro-averaged.

| Model | Val acc % | **Test acc %** | **Test F1** | Precision | Recall |
|---|---|---|---|---|---|
| Decision Tree | 80.0 | 80.0 | 0.792 | 0.875 | 0.833 |
| **Logistic Regression** ✔ | 100.0 | 80.0 | 0.775 | 0.833 | 0.833 |
| Naive Bayes | 90.0 | 70.0 | 0.675 | 0.750 | 0.708 |
| KNN (k=3) | 100.0 | 70.0 | 0.667 | 0.750 | 0.708 |
| **ANN** ✔ | 90.0 | **80.0** | **0.792** | 0.875 | 0.833 |
| Random Forest | 90.0 | 80.0 | 0.792 | 0.875 | 0.833 |
| **Gradient Boosting** ✔ | 90.0 | 80.0 | 0.775 | 0.833 | 0.833 |

**The hardest classification task**, topping out at **80%** (ANN, F1 = 0.792). The report
attributes the residual confusion mainly to **SV1 vs SV2** — plausibly because those two
valves sit at hydraulically similar positions on the branched manifold, so a partial blockage
at either produces a comparable pressure signature at the seven sensor locations. Note the
val→test drop for Logistic Regression and KNN (100% → 80%/70%), a reminder of how little a
10-recording test set can resolve.

### Task 5 — Opening ratio of the failed valve · 15 test recordings

Regression over 48 SV-fault recordings → 33 train / 15 test (no validation set). Inputs = 84
pressure features + 4 one-hot valve columns = **88**. Window predictions are averaged per
recording; accuracy/F1 are computed after snapping to the nearest level (0/25/50/75%), while
MAE stays on the continuous prediction.

| Model | **Test acc %** | **Test F1** | Precision | Recall | **MAE (pp)** |
|---|---|---|---|---|---|
| Linear Regression | 53.3 | 0.476 | 0.467 | 0.500 | 15.4 |
| **Ridge (α=10)** ✔ | **73.3** | **0.717** | 0.875 | 0.729 | **8.0** |
| Decision Tree | 66.7 | 0.653 | 0.774 | 0.667 | 6.7 |
| Baseline (train mean) | 20.0 | 0.083 | 0.050 | 0.250 | 25.9 |

**Ridge was selected** at 73.3% accuracy and 8.0 pp MAE — against 20.0% / 25.9 pp for the
mean baseline, so the features carry real severity information. Ridge's α=10 regularisation
roughly halves Linear Regression's error (15.4 → 8.0 pp), the expected outcome with 88
correlated inputs and only 33 training recordings. The Decision Tree achieves a slightly lower
MAE (6.7) but a worse snapped accuracy, meaning its errors are smaller on average yet more
often land across a level boundary.

### Summary — the saved pipeline

| Task | Saved model(s) | Selected model | Test result | Test recordings |
|---|---|---|---|---|
| 1 · Normal vs abnormal | ANN, GradBoost, LogReg | **ANN** | 94.4% acc, F1 0.929, precision 1.000 | 36 |
| 2 · Bubble / SV fault / Unknown | ANN, GradBoost, LogReg | **Logistic Regression** | 100% acc, F1 1.000, 0 wrongly Unknown | 15 |
| 3 · Bubble location | ANN, GradBoost, LogReg | **Gradient Boosting** | 100% acc (8/8), macro F1 1.000 | 8 |
| 4 · Failed valve | ANN, GradBoost, LogReg | **ANN** | 80.0% acc, macro F1 0.792 | 10 |
| 5 · Opening ratio | Ridge | **Ridge** | 73.3% acc, F1 0.717, MAE 8.0 pp | 15 |

All three classifiers are saved per task, so the selected model can be swapped at inference
time without retraining (`FaultPredictor(use=…)`).

### Predictions on the 46 unseen test recordings

Produced by the saved cascade with default models and written to `predictions_test.csv` and
`Dataset/test/labels_spacecraft.xlsx`. Example rows:

| recording_id | condition | fault_type | bubble_location | failed_valve | opening_ratio_% | label_condition |
|---|---|---|---|---|---|---|
| 178 | Abnormal | Bubble | BP2 | – | – | Anomaly |
| 179 | Abnormal | SV fault | – | SV2 | 25 | Fault |
| 180 | Normal | – | – | – | – | Normal |
| 181 | Abnormal | SV fault | – | SV4 | 50 | Fault |
| 184 | Abnormal | **Unknown** | – | – | – | Unknown |

The cascade does flag **Unknown** cases on this set (e.g. `Case184`), consistent with the
brief's statement that unknown anomalies are mixed into the test data. **These predictions are
unscored** — no public ground truth accompanies the test split, so their accuracy is not
reported anywhere in this README.

### Key conclusions

1. Restricting features to the **first 100 ms after each valve-opening command** was the most
   informative slicing strategy of the three evaluated.
2. Two-step feature selection cut **119 → 84** features per window while leaving Task 1's
   results identical — redundancy removed at no accuracy cost.
3. **Anomaly detection is the most robust task** (94.4%, perfect precision), and even
   unsupervised detectors reach 97.2% — labelled faults are not strictly required for it.
4. **Bubble vs valve fault is nearly perfectly separable** (100% on validation and test for
   all three selected classifiers).
5. Difficulty increases with diagnostic specificity: detection (94.4%) → mechanism (100%) →
   location (100%, tiny sample) → **which valve (80%)** → **how severe (73.3%)**. Identifying
   and quantifying the specific failed valve is where the remaining headroom lies.

## Outputs and Visualizations

| Artifact | Produced by | Contents |
|---|---|---|
| `.venv/features_sliced.csv` | `FeatureEngineering.extract_features()` | 531 × 88 feature matrix (177 recordings × 3 windows) |
| `.venv/saved_models/task{1..5}.joblib` | `TaskTrainer.save()` | Fitted models + scaler + feature-column list + class names |
| `.venv/saved_models/feature_settings.json` | `TaskTrainer.save()` | The train/inference feature contract |
| `.venv/saved_models/training_results.joblib` | `TaskTrainer.save()` | All metrics, tables and split membership |
| `.venv/predictions_test.csv` | `FaultPredictor.run()` | 46 test recordings × all five task outputs |
| `Dataset/test/labels_spacecraft.xlsx` | `FaultPredictor.write_labels()` | Predictions in the submission schema |
| `.venv/dashboard.html` | `Dashboard.build()` | Self-contained 6-tab dashboard |

**Figures available interactively:** per-sensor waveforms, correlation heatmaps and box plots
(`SignalExplorer`); FFT spectra, peak tables, spectrograms and normal-vs-abnormal overlays
(`FFTAnalyzer`); per-task confusion matrix grids (`BaseTask.plot_confusions`,
`TaskTrainer.show`); and everything in the dashboard tabs listed in
[Stage 6](#stage-6--dashboard).

## Limitations and Future Scope

### Limitations (as implemented)

- **Small test sets dominate the uncertainty.** Test splits hold 8–36 recordings, so single
  errors move scores by several points — 12.5 pp for Task 3's 8 recordings, 10 pp for Task 4's
  10. The 100% scores on Tasks 2 and 3 should be read as "no errors on a small sample",
  not as a precise accuracy estimate. Confidence intervals are not computed.
- **The unknown-anomaly capability is untested end-to-end.** Training data contains no novel
  faults, so the gate can only be shown *not to misfire* on known ones (`test_unknown` = 0).
  Its true detection rate on genuine novelties is unmeasured.
- **No cross-spacecraft validation of generalisation.** Splits are stratified by class, not
  grouped by spacecraft unit, so training and test sets share units 1–3. The claim of
  transfer to the unseen unit 4 rests on the unscored test-set predictions.
- **Peak-frequency features were dropped on Task 1 relevance alone.** `fpeak1`–`fpeak3` were
  removed using mutual information against the *normal/abnormal* label, yet the observed
  66 → 63 Hz spectral shift may well carry information specific to bubble localisation and
  opening-ratio estimation — the two weakest tasks.
- **The feature-selection analysis is not reproducible from this repository.** Only the
  resulting `drop_features` list is in the code; the MI and correlation computations behind it
  are reported but not committed.
- **Task 5 is trained only on four discrete levels** (0/25/50/75%), while the brief states the
  true opening ratio is drawn continuously from 0–100%. Snapped accuracy therefore overstates
  performance on genuinely continuous severities.
- **No hyperparameter search.** All model settings are fixed literals in `make_supervised()`
  and `make_regressors()`; no grid/random search or nested cross-validation was run.
- **Operational rough edges:** source lives in `.venv/`; notebooks hard-code absolute Windows
  paths; `pyproject.toml` is incomplete and names the wrong Python version; and inference
  mutates the dataset's labels workbook in place by default.


## References

1. *A Framework for Data-Driven Fault Diagnosis of Numerical Spacecraft Propulsion Systems* —
   International Journal of Prognostics and Health Management.
   <https://papers.phmsociety.org/index.php/ijphm/article/view/3866>
2. *Anomaly Detection in Spacecraft Propulsion System using Time Series Classification based
   on K-NN* — PHM Asia Pacific.
   <https://papers.phmsociety.org/index.php/phmap/article/view/3675>