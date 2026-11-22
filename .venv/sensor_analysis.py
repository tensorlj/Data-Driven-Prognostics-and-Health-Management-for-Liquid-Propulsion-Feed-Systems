"""
sensor_analysis.py
Classes for the pressure-sensor project (columns: TIME, P1..P7; TIME in seconds).

    DatasetInspector : column check and missing-value check across a folder of CSVs
    SignalExplorer   : overview, line plots, correlation heatmap, box plots for one dataframe
    FFTAnalyzer      : FFT spectra, dominant peaks, spectrogram, normal-vs-abnormal overlay
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks

try:
    import seaborn as sns
except ImportError:          # heatmap falls back to matplotlib
    sns = None

try:
    from IPython.display import display
except ImportError:          # plain Python: just print
    def display(x):
        print(x)


# ----------------------------------------------------------------------
class DatasetInspector:
    """File-level checks over every CSV in a folder."""

    def __init__(self, path):
        self.path = path
        self.files = sorted(f for f in os.listdir(path) if f.lower().endswith(".csv"))
        self.columns_by_file = {}
        self.missing_summary = None
        self.missing_by_column = {}

    def count_files(self):
        print(f"Number of files: {len(self.files)}\n")
        return len(self.files)

    def list_columns(self, verbose=True):
        """Read only the headers (fast) and store/print the columns of each file."""
        for f in self.files:
            cols = list(pd.read_csv(os.path.join(self.path, f), nrows=0).columns)
            self.columns_by_file[f] = cols
            if verbose:
                print(f"{f}: {len(cols)} columns")
                print(f"   {cols}\n")
        return self.columns_by_file

    def check_columns(self, verbose=True):
        """True if every file has the same columns in the same order."""
        if not self.files:
            print("No CSV files found.")
            return False
        if not self.columns_by_file:
            self.list_columns(verbose=verbose)
        ref_file = self.files[0]
        ref_cols = self.columns_by_file[ref_file]
        all_same = True
        for f, cols in self.columns_by_file.items():
            if cols != ref_cols:
                all_same = False
                missing, extra = set(ref_cols) - set(cols), set(cols) - set(ref_cols)
                print(f"MISMATCH in {f}")
                if missing:
                    print(f"   Missing columns: {missing}")
                if extra:
                    print(f"   Extra columns:   {extra}")
                if not missing and not extra:
                    print("   Same columns but in a different order")
        if all_same:
            print("All files have identical columns (same names and order).")
        else:
            print("\nFiles do NOT all have the same columns.")
        return all_same

    def missing_values(self, verbose=True):
        """Per-file missing-value summary; returns a DataFrame."""
        rows, by_col = [], {}
        for f in self.files:
            d = pd.read_csv(os.path.join(self.path, f))
            total_cells = d.shape[0] * d.shape[1]
            total_missing = int(d.isnull().sum().sum())
            rows.append({
                "file": f,
                "rows": d.shape[0],
                "columns": d.shape[1],
                "total_missing": total_missing,
                "missing_%": round(100 * total_missing / total_cells, 2) if total_cells else 0,
                "rows_with_missing": int(d.isnull().any(axis=1).sum()),
            })
            cm = d.isnull().sum()
            by_col[f] = cm[cm > 0]
        self.missing_summary = pd.DataFrame(rows)
        self.missing_by_column = by_col

        if verbose:
            display(self.missing_summary)
            bad = self.missing_summary[self.missing_summary["total_missing"] > 0]
            print(f"\nFiles checked: {len(self.missing_summary)}")
            print(f"Files with missing values: {len(bad)}")
            print(f"Total missing values in dataset: {self.missing_summary['total_missing'].sum()}")
            for f, cm in by_col.items():
                if len(cm) > 0:
                    print(f"\n{f}: missing values per column")
                    print(cm.to_string())
            if len(bad) == 0:
                print("\nNo missing values found in any file.")
        return self.missing_summary

    def run_all(self):
        self.count_files()
        self.list_columns()
        self.check_columns()
        self.missing_values()


# ----------------------------------------------------------------------
class SignalExplorer:
    """Basic exploration of one recording (a dataframe)."""

    def __init__(self, df, name=""):
        self.df = df
        self.name = name
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()

    @classmethod
    def from_file(cls, path, file_name):
        return cls(pd.read_csv(os.path.join(path, file_name)), name=file_name)

    def overview(self):
        print(f"\nShape: {self.df.shape}")
        print("\nFirst 5 rows:")
        display(self.df.head())
        print("\nData types / missing values:")
        self.df.info()
        print("\nSummary statistics:")
        display(self.df.describe())
        print("\nNumeric columns:", self.numeric_cols)

    def plot_lines(self):
        self.df[self.numeric_cols].plot(subplots=True, figsize=(14, 2.5 * len(self.numeric_cols)),
                                        title=f"Line plots - {self.name}")
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self):
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numeric columns for a correlation heatmap.")
            return
        corr = self.df[self.numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        if sns is not None:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        else:
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr)), corr.columns, rotation=90)
            plt.yticks(range(len(corr)), corr.columns)
        plt.title(f"Correlation heatmap - {self.name}")
        plt.show()

    def plot_boxplots(self):
        self.df[self.numeric_cols].plot(kind="box", figsize=(14, 6), rot=45,
                                        title=f"Box plots - {self.name}")
        plt.show()

    def run_all(self):
        self.overview()
        self.plot_lines()
        self.plot_heatmap()
        self.plot_boxplots()


# ----------------------------------------------------------------------
class FFTAnalyzer:
    """
    Frequency-domain analysis of one recording.

    t_start_ms / t_end_ms : optional segment in milliseconds (end inclusive), e.g. 100, 400.
                            Leave both as None to analyse the whole recording.
    time_unit             : unit of the TIME column, "s" (default) or "ms".
    """

    def __init__(self, df, name="", time_col="TIME", t_start_ms=None, t_end_ms=None, time_unit="s"):
        self.name, self.time_col = name, time_col
        self.t_start_ms, self.t_end_ms = t_start_ms, t_end_ms
        self.to_ms = 1000.0 if time_unit == "s" else 1.0

        d = df
        if t_start_ms is not None or t_end_ms is not None:
            t_ms = df[time_col] * self.to_ms
            lo = -np.inf if t_start_ms is None else t_start_ms - 1e-6   # tolerance for float time values
            hi = np.inf if t_end_ms is None else t_end_ms + 1e-6
            d = df[(t_ms >= lo) & (t_ms <= hi)]
        self.seg = d.reset_index(drop=True)
        if len(self.seg) < 8:
            raise ValueError(f"Segment has only {len(self.seg)} rows - check the TIME unit and limits.")

        numeric = self.seg.select_dtypes(include="number").columns.tolist()
        self.sensor_cols = [c for c in numeric if c != time_col]
        self.freqs, self.fft_results = None, {}

    @classmethod
    def from_file(cls, path, file_name, **kwargs):
        return cls(pd.read_csv(os.path.join(path, file_name)), name=file_name, **kwargs)

    # ---- computation ----
    def compute(self, verbose=True):
        t = self.seg[self.time_col].values
        dt = np.diff(t)
        dt_mean = np.mean(dt)
        self.N = len(self.seg)
        dt_s = dt_mean * self.to_ms / 1000.0
        self.fs_hz = 1.0 / dt_s

        if verbose:
            seg_txt = (f"{self.t_start_ms}-{self.t_end_ms} ms" if self.t_start_ms is not None
                       or self.t_end_ms is not None else "full recording")
            print(f"Segment: {seg_txt}  |  Samples (N): {self.N}")
            print(f"Mean sampling interval: {dt_mean:.6f}  ->  fs = {self.fs_hz:.2f} Hz")
            print(f"Nyquist frequency (max analyzable): {self.fs_hz / 2:.2f} Hz")
            print(f"Frequency resolution: {self.fs_hz / self.N:.4f} Hz")
        if np.std(dt) / dt_mean > 0.01:
            print("WARNING: TIME is not evenly spaced. Consider resampling/interpolating first.")

        window = np.hanning(self.N)
        self.freqs = np.fft.rfftfreq(self.N, d=dt_s)
        self.fft_results = {}
        for col in self.sensor_cols:
            x = self.seg[col].values.astype(float)
            x = x - np.mean(x)                                   # remove DC offset
            X = np.fft.rfft(x * window)
            self.fft_results[col] = (2.0 / np.sum(window)) * np.abs(X)
        return self

    def _need_compute(self):
        if self.freqs is None:
            self.compute(verbose=False)

    # ---- outputs ----
    def _segment_label(self):
        if self.t_start_ms is None and self.t_end_ms is None:
            return ""
        return f" ({self.t_start_ms}-{self.t_end_ms} ms)"

    def plot_spectra(self):
        self._need_compute()
        n = len(self.sensor_cols)
        fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, self.sensor_cols):
            ax.plot(self.freqs, self.fft_results[col])
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"FFT amplitude spectrum{self._segment_label()} - {self.name}")
        axes[-1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()

    def peak_table(self, n_peaks=3):
        """Top n distinct spectral peaks per sensor (local maxima, 0 Hz skipped)."""
        self._need_compute()
        rows = []
        for col in self.sensor_cols:
            a, f = self.fft_results[col][1:], self.freqs[1:]
            pk, _ = find_peaks(a)
            top = pk[np.argsort(a[pk])[::-1][:n_peaks]]
            row = {"sensor": col}
            for i in range(n_peaks):
                if i < len(top):
                    row[f"peak{i+1}_freq_Hz"] = round(float(f[top[i]]), 2)
                    row[f"peak{i+1}_amp"] = round(float(a[top[i]]), 4)
                else:
                    row[f"peak{i+1}_freq_Hz"], row[f"peak{i+1}_amp"] = np.nan, np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_spectrogram(self, sensor=None, nperseg=None):
        self._need_compute()
        sensor = sensor or self.sensor_cols[0]
        segmented = self.t_start_ms is not None or self.t_end_ms is not None
        if nperseg is None:
            nperseg = min(64 if segmented else 256, self.N // 4)
        x = self.seg[sensor].values
        f_s, t_s, Sxx = spectrogram(x - x.mean(), fs=self.fs_hz, nperseg=nperseg)
        t0_ms = self.seg[self.time_col].iloc[0] * self.to_ms
        plt.figure(figsize=(14, 5))
        plt.pcolormesh(t0_ms + t_s * 1000, f_s, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
        plt.colorbar(label="Power (dB)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Spectrogram of {sensor}{self._segment_label()} - {self.name}")
        plt.show()

    def run_all(self, sensor=None):
        self.compute()
        self.plot_spectra()
        display(self.peak_table())
        self.plot_spectrogram(sensor)

    # ---- comparison ----
    def overlay(self, other, label_self="normal", label_other="abnormal"):
        """Overlay this spectrum on another analyzer's (same segment => same N and frequency axis)."""
        self._need_compute()
        other._need_compute()
        if self.N != other.N:
            raise ValueError(f"Different N ({self.N} vs {other.N}); use the same segment for both.")
        n = len(self.sensor_cols)
        fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, self.sensor_cols):
            ax.plot(self.freqs, self.fft_results[col], label=label_self)
            ax.plot(other.freqs, other.fft_results[col], label=label_other, alpha=0.8)
            ax.set_ylabel(col)
            ax.grid(alpha=0.3)
        axes[0].legend()
        axes[0].set_title(f"FFT overlay{self._segment_label()}")
        axes[-1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()
