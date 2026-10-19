import os
import tempfile
import unittest

import pandas as pd

from visualizer import (
    detect_time_column,
    filter_day_data,
    find_matching_or_closest_date,
    get_data_columns,
    resample_to_5min,
)


class VisualizerDataTests(unittest.TestCase):
    def test_detect_time_column_and_numeric_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "sample.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "datetime;consumption_kwh;production_kwh;status\n"
                    "2025-05-20 00:00:00;1.0;2.0;ok\n"
                    "2025-05-20 00:05:00;1.5;2.5;ok\n"
                    "2025-05-21 00:00:00;2.0;3.0;ok\n"
                )

            df, time_col = detect_time_column(csv_path)
            self.assertEqual(time_col, "datetime")
            columns = get_data_columns(df, time_col)
            self.assertEqual(columns, ["consumption_kwh", "production_kwh"])

    def test_filter_day_data_returns_selected_day(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "sample.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "datetime;consumption_kwh;production_kwh\n"
                    "2025-05-20 00:00:00;1.0;2.0\n"
                    "2025-05-20 00:05:00;1.5;2.5\n"
                    "2025-05-21 00:00:00;2.0;3.0\n"
                )

            df, time_col = detect_time_column(csv_path)
            daily = filter_day_data(df, time_col, "2025-05-20")
            self.assertEqual(len(daily), 2)
            self.assertEqual(daily[time_col].dt.date.iloc[0].isoformat(), "2025-05-20")

    def test_find_matching_or_closest_date(self):
        available = ["2025-05-20", "2025-05-21", "2025-06-01", "2025-06-15"]
        
        # Exact match
        self.assertEqual(find_matching_or_closest_date("2025-05-21", available), "2025-05-21")
        
        # Prefix match
        self.assertEqual(find_matching_or_closest_date("2025-06", available), "2025-06-01")
        
        # Format parsing (DD.MM.YYYY)
        self.assertEqual(find_matching_or_closest_date("20.05.2025", available), "2025-05-20")
        
        # Closest date when out of sample
        self.assertEqual(find_matching_or_closest_date("2025-05-22", available), "2025-05-21")
        self.assertEqual(find_matching_or_closest_date("2025-06-10", available), "2025-06-15")
        
        # Empty/None checks
        self.assertIsNone(find_matching_or_closest_date("2025-05-20", []))
        self.assertEqual(find_matching_or_closest_date("", available), "2025-05-20")

    def test_resample_to_5min(self):
        # 1-hour interval data (2 points) -> resampled to 5-min interval (13 points: 00:00 to 01:00)
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2025-05-20 00:00:00", "2025-05-20 01:00:00"]),
            "consumption_kwh": [1.2, 2.4],
            "production_kwh": [0.0, 1.2],
        })
        resampled = filter_day_data(df, "datetime", "2025-05-20", force_5min=True)
        self.assertEqual(len(resampled), 13)
        self.assertAlmostEqual(resampled["consumption_kwh"].iloc[0], 1.2)
        self.assertAlmostEqual(resampled["consumption_kwh"].iloc[-1], 2.4)
        # Check 5-minute spacing
        time_diffs = resampled["datetime"].diff().dropna().dt.total_seconds() / 60.0
        self.assertTrue((time_diffs == 5.0).all())


if __name__ == "__main__":
    unittest.main()
