"""
Script to read all CSV files from /data folder and create variables for each column.
Variable names are in English based on column headers with predefined mapping.
"""

import pandas as pd
from pathlib import Path


# Predefined mapping: original column name -> English variable name
COLUMN_MAPPING = {
    # ceny.csv - prices
    "DatumCasOTE": "datetime_ote",
    "CenaEUR": "price_eur",
    "CenaCZK": "price_czk",
    
    # Maly.csv - energy data
    "ID_zaznamu": "record_id",
    "ID_ZDROJE": "source_id",
    "DatumCas": "datetime",
    "SoC": "state_of_charge",
    "P_PV": "solar_power",
    "P_HOME": "home_power",
    "P_HOME_L1": "home_power_l1",
    "P_HOME_L2": "home_power_l2",
    "P_HOME_L3": "home_power_l3",
    "P_EPS": "eps_power",
    "P_GRID": "grid_power",
    "P_GRID_L1": "grid_power_l1",
    "P_GRID_L2": "grid_power_l2",
    "P_GRID_L3": "grid_power_l3",
    "P_BAT": "battery_power",
    "PVenergy": "solar_energy",
    "PVenergy_T": "solar_energy_total",
    "ToBAT": "to_battery",
    "ToBAT_T": "to_battery_total",
    "FromBAT": "from_battery",
    "FromBAT_T": "from_battery_total",
    "SELL": "energy_sold",
    "SELL_T": "energy_sold_total",
    "BUY": "energy_bought",
    "BUY_T": "energy_bought_total",
    "Consumed": "energy_consumed",
    "Consumed_T": "energy_consumed_total",
    "GRID_LIMIT": "grid_limit",
    
    # predpoved.csv - weather forecast
    "Teplota": "temperature",
    "Oblacnost": "cloudiness",
    "Vitr": "wind_speed",
    "Smer": "wind_direction",
    "Vlhkost": "humidity",
    
    # sun.csv - solar production
    "Vyroba": "solar_production",
    "Suma": "total",
}


def get_english_name(original_name: str, file_prefix: str) -> str:
    """Get English variable name from original column name."""
    if original_name in COLUMN_MAPPING:
        return f"{file_prefix}_{COLUMN_MAPPING[original_name]}"
    else:
        return f"{file_prefix}_{original_name.lower()}"


def read_all_csv_files():
    """Read all CSV files from data folder and store each column in a separate variable."""
    data_dir = Path("data")
    all_data = {}
    
    for csv_file in sorted(data_dir.glob("*.csv")):
        file_name = csv_file.stem
        df = pd.read_csv(csv_file, sep=';')
        
        file_data = {}
        
        for column in df.columns:
            var_name = get_english_name(column, file_name)
            file_data[var_name] = df[column].tolist()
        
        all_data[file_name] = file_data
        print(f"\n=== {csv_file.name} ===")
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Variable prefix: {file_name}")
        print(f"Rows: {len(df)}")
    
    return all_data


def create_global_variables(all_data):
    """Create global variables for each column from all CSV files."""
    for file_name, columns in all_data.items():
        for var_name, col_data in columns.items():
            globals()[var_name] = col_data
            print(f"Created: {var_name} ({len(col_data)} items)")


def main():
    print("Reading CSV files from /data folder...\n")
    
    all_data = read_all_csv_files()
    
    print("\n--- Creating variables ---")
    create_global_variables(all_data)
    
    print("\n=== Summary ===")
    total_vars = sum(len(columns) for columns in all_data.values())
    print(f"Total files processed: {len(all_data)}")
    print(f"Total variables created: {total_vars}")
    
    return all_data


if __name__ == "__main__":
    all_data = main()
