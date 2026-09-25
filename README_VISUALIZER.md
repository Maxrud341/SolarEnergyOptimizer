# CSV Time Visualizer

Simple dark-theme interface for viewing time series data from CSV files.

## Features
- Reads CSV files from the `data` folder
- Detects a timestamp column automatically (`datetime`, `timestamp`, `DatumCas`, etc.)
- Plots all numeric series on a time axis
- Lets you hide/show individual series with checkboxes
- Shows one day at a time with `Previous day` and `Next day` controls
- Updates the chart automatically whenever the file, day or toggles change

## Running
```bash
python visualizer.py
```

## Notes
- The project CSV files use `;` as separator.
- The visualization is optimized for daily views and 5-minute measurement data.
