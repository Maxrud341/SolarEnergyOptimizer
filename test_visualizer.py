import os
import tempfile
import unittest

from visualizer import detect_time_column, filter_day_data, get_data_columns


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


if __name__ == "__main__":
    unittest.main()
