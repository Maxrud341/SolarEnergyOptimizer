import csv
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


def filter_day_data(df: pd.DataFrame, time_col: str, day_label: str):
    """Return a subset for a single day."""
    if df.empty or time_col not in df.columns:
        return df.copy()
    return df[df[time_col].dt.strftime("%Y-%m-%d") == day_label].copy()


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
        self.csv_files = sorted(self.data_dir.glob("*.csv")) if self.data_dir.exists() else []
        self.current_df = pd.DataFrame()
        self.current_time_col = ""
        self.current_day = ""
        self.current_columns = []
        self.series_vars = {}

        self.build_ui()

        if self.csv_files:
            self.load_csv(self.csv_files[0])
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
        self.file_var = tk.StringVar(value=file_values[0] if file_values else "")
        self.file_combo = ttk.Combobox(self.sidebar, textvariable=self.file_var, values=file_values, state="readonly", width=28)
        self.file_combo.pack(fill="x", pady=(0, 16))
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_change)

        self.day_label = tk.Label(self.sidebar, text="Day: --", font=("Segoe UI", 11, "bold"), bg="#111827", fg="#f8fafc")
        self.day_label.pack(anchor="w", pady=(0, 6))

        nav_row = tk.Frame(self.sidebar, bg="#111827")
        nav_row.pack(fill="x", pady=(0, 12))

        self.prev_day_btn = tk.Button(nav_row, text="Previous day", command=self.previous_day, bg="#1e293b", fg="#f8fafc", bd=0, padx=10, pady=8)
        self.prev_day_btn.pack(side="left", padx=(0, 8))

        self.next_day_btn = tk.Button(nav_row, text="Next day", command=self.next_day, bg="#2563eb", fg="#f8fafc", bd=0, padx=10, pady=8)
        self.next_day_btn.pack(side="left")

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
        csv_path = self.data_dir / selected_name
        self.load_csv(csv_path)

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
        self.populate_series_controls()
        self.update_day_label()
        self.refresh_plot()

    def populate_series_controls(self):
        for widget in self.series_frame.winfo_children():
            widget.destroy()

        self.series_vars = {}
        for column in self.current_columns:
            var = tk.BooleanVar(value=True)
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
            if days:
                current_index = days.index(self.current_day) if self.current_day in days else 0
                self.day_label.config(text=f"Day: {days[current_index]} | 1 day")
                return
        self.day_label.config(text="Day: -- | 1 day")

    def previous_day(self):
        days = get_day_labels(self.current_df, self.current_time_col)
        if not days:
            return
        current_index = days.index(self.current_day) if self.current_day in days else 0
        new_index = max(0, current_index - 1)
        self.current_day = days[new_index]
        self.update_day_label()
        self.refresh_plot()

    def next_day(self):
        days = get_day_labels(self.current_df, self.current_time_col)
        if not days:
            return
        current_index = days.index(self.current_day) if self.current_day in days else 0
        new_index = min(len(days) - 1, current_index + 1)
        self.current_day = days[new_index]
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

        day_df = filter_day_data(self.current_df, self.current_time_col, self.current_day)
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
            self.ax.plot(
                day_df[self.current_time_col],
                day_df[column],
                label=normalize_label(column),
                linewidth=2,
            )

        self.ax.set_title(f"{self.current_day} • 1 day view", color="#f8fafc", fontsize=13)
        self.ax.set_xlabel("Time", color="#f8fafc")
        self.ax.set_ylabel("Value", color="#f8fafc")

        if len(day_df) > 1:
            self.ax.set_xlim(day_df[self.current_time_col].min(), day_df[self.current_time_col].max())

        self.ax.legend(loc="upper right", frameon=False, labelcolor="#f8fafc")
        self.ax.tick_params(axis="x", rotation=25)
        self.figure.tight_layout()
        self.canvas.draw_idle()


def main():
    app = CSVTimeVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
