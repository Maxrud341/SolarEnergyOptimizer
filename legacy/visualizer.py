import csv
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from matplotlib.figure import Figure


DATA_DIR = Path(__file__).resolve().parent / "data"


def normalize_label(name: str) -> str:
    """Convert column names to readable legend labels."""
    cleaned = name.replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in cleaned.split())


def detect_separator(csv_path: str | Path):
    """Detect CSV separator for CSV files in this project."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except csv.Error:
        return ";" if ";" in sample else ","


def detect_time_column(csv_path: str | Path):
    """Load CSV and find the most suitable timestamp column."""
    path = Path(csv_path)
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep)

    candidates = [
        "datetime",
        "timestamp",
        "date_time",
        "time",
        "DatumCas",
        "DatumCasOTE",
        "date",
        "Date",
        "DateTime",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            time_col = candidate
            break
    else:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                sample_values = df[col].dropna().head(10)
                if not sample_values.empty:
                    parsed = pd.to_datetime(sample_values, errors="coerce")
                    if parsed.notna().sum() >= max(1, len(sample_values) // 2):
                        time_col = col
                        break
        else:
            raise ValueError(f"No valid datetime-like column found in {path.name}.")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df, time_col


def get_data_columns(df: pd.DataFrame, time_col: str):
    """Return all numeric columns except the time column."""
    numeric_cols = []
    for col in df.columns:
        if col == time_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def get_day_labels(df: pd.DataFrame, time_col: str):
    """Return sorted unique dates from the dataset."""
    if df.empty or time_col not in df.columns:
        return []
    return sorted(df[time_col].dt.strftime("%Y-%m-%d").unique().tolist())


def resample_to_5min(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Resample dataframe to regular 5-minute intervals with linear interpolation."""
    if df.empty or time_col not in df.columns:
        return df.copy()

    df_indexed = df.copy()
    df_indexed[time_col] = pd.to_datetime(df_indexed[time_col], errors="coerce")
    df_indexed = df_indexed.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    if len(df_indexed) > 1:
        diffs = df_indexed.index.to_series().diff().dropna()
        median_minutes = diffs.median().total_seconds() / 60.0
        if median_minutes <= 5.1:
            # Already 5-min or sub-5-min resolution
            return df_indexed.reset_index()

    # Resample to 5-minute grid
    numeric_cols = df_indexed.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return df_indexed.reset_index()

    resampled = df_indexed[numeric_cols].resample("5min").interpolate(method="linear").ffill().bfill()
    return resampled.reset_index()


def filter_day_data(df: pd.DataFrame, time_col: str, day_label: str, force_5min: bool = True):
    """Return a subset for a single day, resampled to 5-minute intervals if requested."""
    if df.empty or time_col not in df.columns:
        return df.copy()
    day_df = df[df[time_col].dt.strftime("%Y-%m-%d") == day_label].copy()
    if force_5min and len(day_df) > 1:
        day_df = resample_to_5min(day_df, time_col)
    return day_df


def find_matching_or_closest_date(input_str: str, available_dates: list[str]) -> str | None:
    """Find exact, prefix, or closest matching date in available_dates."""
    if not available_dates:
        return None
    cleaned = str(input_str).strip()
    if not cleaned:
        return available_dates[0]

    # 1. Exact match
    if cleaned in available_dates:
        return cleaned

    # 2. Prefix match (e.g. "2025-06" matches first day in June 2025)
    for d in available_dates:
        if d.startswith(cleaned):
            return d

    # 3. Parse user date format (YYYY-MM-DD, DD.MM.YYYY, YYYY/MM/DD, etc.)
    dayfirst = "." in cleaned
    parsed = pd.to_datetime(cleaned, errors="coerce", dayfirst=dayfirst)
    if pd.isna(parsed):
        parsed = pd.to_datetime(cleaned, errors="coerce")

    if not pd.isna(parsed):
        iso_date = parsed.strftime("%Y-%m-%d")
        if iso_date in available_dates:
            return iso_date

        # If date is out of sample, find the closest date
        target_dt = parsed.date()
        date_objs = [pd.to_datetime(d).date() for d in available_dates]
        closest_idx = min(range(len(date_objs)), key=lambda i: abs((date_objs[i] - target_dt).days))
        return available_dates[closest_idx]

    return None


class CSVTimeVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Time Visualizer")
        self.geometry("1280x760")
        self.minsize(980, 620)

        self.configure(bg="#0f172a")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#0f172a")
        self.style.configure("TLabel", background="#0f172a", foreground="#e2e8f0")
        self.style.configure("TButton", background="#1e293b", foreground="#f8fafc")
        self.style.configure("TCombobox", fieldbackground="#111827", background="#111827", foreground="#f8fafc")

        self.data_dir = DATA_DIR
        found_files = []
        if self.data_dir.exists():
            found_files.extend(list(self.data_dir.glob("*.csv")))
        root_dir = Path(__file__).resolve().parent
        for root_csv in root_dir.glob("*.csv"):
            if not any(f.name.lower() == root_csv.name.lower() for f in found_files):
                found_files.append(root_csv)
        self.csv_files = sorted(found_files, key=lambda p: (0 if "trading_log" in p.name.lower() else 1, p.name.lower()))
        self.current_df = pd.DataFrame()
        self.current_time_col = ""
        self.current_day = ""
        self.current_columns = []
        self.series_vars = {}

        # Default to trading_log_v2.csv or datasource.csv if available
        default_index = 0
        for i, p in enumerate(self.csv_files):
            if p.name.lower() in ("trading_log_v2.csv", "datasource.csv", "maly.csv"):
                default_index = i
                break
        self.default_file_index = default_index

        self.build_ui()

        if self.csv_files:
            self.load_csv(self.csv_files[self.default_file_index])
        else:
            self.show_empty_state("No CSV files found in the data folder.")

    def build_ui(self):
        self.main_frame = tk.Frame(self, bg="#0f172a")
        self.main_frame.pack(fill="both", expand=True, padx=16, pady=16)

        self.sidebar = tk.Frame(self.main_frame, bg="#111827", padx=16, pady=16, bd=0)
        self.sidebar.pack(side="left", fill="y")

        self.canvas_frame = tk.Frame(self.main_frame, bg="#0f172a")
        self.canvas_frame.pack(side="right", fill="both", expand=True)

        self.file_label = tk.Label(self.sidebar, text="CSV file", font=("Segoe UI", 11, "bold"), bg="#111827", fg="#f8fafc")
        self.file_label.pack(anchor="w", pady=(0, 8))

        file_values = [path.name for path in self.csv_files]
        self.file_var = tk.StringVar(value=file_values[self.default_file_index] if file_values else "")
        self.file_combo = ttk.Combobox(self.sidebar, textvariable=self.file_var, values=file_values, state="readonly", width=28)
        self.file_combo.pack(fill="x", pady=(0, 12))
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_change)

        self.resample_var = tk.BooleanVar(value=True)
        self.resample_chk = tk.Checkbutton(
            self.sidebar,
            text="5-minute intervals (5m)",
            variable=self.resample_var,
            command=self.refresh_plot,
            bg="#111827",
            fg="#38bdf8",
            selectcolor="#0f172a",
            activebackground="#111827",
            activeforeground="#38bdf8",
            font=("Segoe UI", 9, "bold"),
            pady=2,
        )
        self.resample_chk.pack(anchor="w", pady=(0, 12))

        self.day_label = tk.Label(self.sidebar, text="Day: --", font=("Segoe UI", 11, "bold"), bg="#111827", fg="#f8fafc")
        self.day_label.pack(anchor="w", pady=(0, 4))

        # Jump to date controls
        jump_frame = tk.Frame(self.sidebar, bg="#111827")
        jump_frame.pack(fill="x", pady=(0, 8))

        self.jump_label = tk.Label(jump_frame, text="Jump to date (YYYY-MM-DD):", font=("Segoe UI", 9), bg="#111827", fg="#94a3b8")
        self.jump_label.pack(anchor="w", pady=(0, 3))

        input_row = tk.Frame(jump_frame, bg="#111827")
        input_row.pack(fill="x", pady=(0, 4))

        self.day_var = tk.StringVar()
        self.day_combo = ttk.Combobox(input_row, textvariable=self.day_var, values=[], width=16)
        self.day_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.day_combo.bind("<<ComboboxSelected>>", self.on_date_selected)
        self.day_combo.bind("<Return>", self.on_date_entered)

        self.jump_btn = tk.Button(input_row, text="Go", command=self.jump_to_date, bg="#0ea5e9", fg="#ffffff", bd=0, padx=10, pady=3, font=("Segoe UI", 9, "bold"), cursor="hand2")
        self.jump_btn.pack(side="right")

        nav_row = tk.Frame(self.sidebar, bg="#111827")
        nav_row.pack(fill="x", pady=(0, 12))

        self.prev_day_btn = tk.Button(nav_row, text="◀ Previous", command=self.previous_day, bg="#1e293b", fg="#f8fafc", bd=0, padx=10, pady=8, cursor="hand2")
        self.prev_day_btn.pack(side="left", padx=(0, 8), fill="x", expand=True)

        self.next_day_btn = tk.Button(nav_row, text="Next ▶", command=self.next_day, bg="#2563eb", fg="#f8fafc", bd=0, padx=10, pady=8, cursor="hand2")
        self.next_day_btn.pack(side="left", fill="x", expand=True)

        self.series_title = tk.Label(self.sidebar, text="Visible series", font=("Segoe UI", 11, "bold"), bg="#111827", fg="#f8fafc")
        self.series_title.pack(anchor="w", pady=(12, 8))

        self.series_frame = tk.Frame(self.sidebar, bg="#111827")
        self.series_frame.pack(fill="both")

        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor="#0f172a")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#0f172a")
        self.ax.grid(color="#334155", linewidth=0.7, alpha=0.7)
        self.ax.tick_params(colors="#cbd5e1")
        for spine in self.ax.spines.values():
            spine.set_color("#334155")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def on_file_change(self, event=None):
        selected_name = self.file_var.get()
        if not selected_name:
            return
        for p in self.csv_files:
            if p.name == selected_name:
                self.load_csv(p)
                return
        csv_path = self.data_dir / selected_name
        self.load_csv(csv_path)

    def on_date_selected(self, event=None):
        self.jump_to_date()

    def on_date_entered(self, event=None):
        self.jump_to_date()

    def jump_to_date(self):
        days = get_day_labels(self.current_df, self.current_time_col)
        if not days:
            return
        target_str = self.day_var.get()
        matched = find_matching_or_closest_date(target_str, days)
        if matched:
            self.current_day = matched
            self.day_var.set(matched)
            self.update_day_label()
            self.refresh_plot()

    def load_csv(self, csv_path: Path):
        try:
            df, time_col = detect_time_column(csv_path)
        except Exception as exc:  # pragma: no cover - GUI error path
            self.show_empty_state(f"Failed to load file: {exc}")
            return

        self.current_df = df
        self.current_time_col = time_col
        self.current_columns = get_data_columns(df, time_col)
        self.current_day = ""

        days = get_day_labels(df, time_col)
        if days:
            self.current_day = days[0]
            self.day_combo["values"] = days
            self.day_var.set(self.current_day)
        else:
            self.day_combo["values"] = []
            self.day_var.set("")

        self.populate_series_controls()
        self.update_day_label()
        self.refresh_plot()

    def populate_series_controls(self):
        for widget in self.series_frame.winfo_children():
            widget.destroy()

        self.series_vars = {}
        for column in self.current_columns:
            # Key series to show by default
            is_key_metric = column in (
                "house_consumption_kwh",
                "load_kwh",
                "pv_kwh",
                "buy_kwh",
                "sell_kwh",
                "discharge_kwh",
                "charge_kwh",
                "soc_kwh",
                "spot_price_czk_per_kwh",
                "cost_buy_czk",
            )
            default_checked = is_key_metric or len(self.current_columns) <= 8
            var = tk.BooleanVar(value=default_checked)
            var.trace_add("write", lambda *args, var=var: self.refresh_plot())
            self.series_vars[column] = var

            checkbox = tk.Checkbutton(
                self.series_frame,
                text=normalize_label(column),
                variable=var,
                bg="#111827",
                fg="#ecfeff",
                selectcolor="#0f172a",
                activebackground="#111827",
                activeforeground="#f8fafc",
                anchor="w",
                justify="left",
                pady=4,
            )
            checkbox.pack(anchor="w", fill="x")

    def update_day_label(self):
        if not self.current_df.empty:
            days = get_day_labels(self.current_df, self.current_time_col)
            if days and self.current_day in days:
                current_index = days.index(self.current_day)
                self.day_label.config(text=f"Day {current_index + 1}/{len(days)}: {self.current_day}")
                return
        self.day_label.config(text="Day: -- | 1 day")

    def previous_day(self):
        days = get_day_labels(self.current_df, self.current_time_col)
        if not days:
            return
        current_index = days.index(self.current_day) if self.current_day in days else 0
        new_index = max(0, current_index - 1)
        self.current_day = days[new_index]
        self.day_var.set(self.current_day)
        self.update_day_label()
        self.refresh_plot()

    def next_day(self):
        days = get_day_labels(self.current_df, self.current_time_col)
        if not days:
            return
        current_index = days.index(self.current_day) if self.current_day in days else 0
        new_index = min(len(days) - 1, current_index + 1)
        self.current_day = days[new_index]
        self.day_var.set(self.current_day)
        self.update_day_label()
        self.refresh_plot()

    def show_empty_state(self, message: str):
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", color="#f8fafc", fontsize=12)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def refresh_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#0f172a")
        self.ax.grid(color="#334155", linewidth=0.7, alpha=0.7)
        self.ax.tick_params(colors="#cbd5e1")

        for spine in self.ax.spines.values():
            spine.set_color("#334155")

        if self.current_df.empty or not self.current_day:
            self.show_empty_state("No data available for the selected period.")
            return

        force_5min = getattr(self, "resample_var", tk.BooleanVar(value=True)).get()
        day_df = filter_day_data(self.current_df, self.current_time_col, self.current_day, force_5min=force_5min)
        if day_df.empty:
            self.show_empty_state(f"No values for {self.current_day}.")
            return

        selected = [column for column in self.current_columns if self.series_vars.get(column, tk.BooleanVar(value=True)).get()]
        if not selected:
            self.ax.text(0.5, 0.5, "Select at least one series to display.", ha="center", va="center", color="#f8fafc", fontsize=12)
            self.ax.set_axis_off()
            self.canvas.draw_idle()
            return

        for column in selected:
            if column in day_df.columns:
                self.ax.plot(
                    day_df[self.current_time_col],
                    day_df[column],
                    label=normalize_label(column),
                    linewidth=1.8,
                )

        mode_text = "5-minute resolution" if force_5min else "native resolution"
        self.ax.set_title(f"{self.current_day} • {mode_text} ({len(day_df)} steps/day)", color="#f8fafc", fontsize=13)
        self.ax.set_xlabel("Time (HH:MM)", color="#f8fafc")
        self.ax.set_ylabel("Value", color="#f8fafc")

        if len(day_df) > 1:
            self.ax.set_xlim(day_df[self.current_time_col].min(), day_df[self.current_time_col].max())
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            self.ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
            self.ax.grid(which="minor", color="#1e293b", linestyle=":", linewidth=0.5, alpha=0.5)

        self.ax.legend(loc="upper right", frameon=False, labelcolor="#f8fafc")
        self.ax.tick_params(axis="x", rotation=25)
        self.figure.tight_layout()
        self.canvas.draw_idle()


def main():
    app = CSVTimeVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
