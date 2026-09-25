"""
Create datasource with 5-minute intervals.
All values in kWh for consistent measurement units.
"""

import pandas as pd
from pathlib import Path


BATTERY_SIZE_KWH = 0.5  # Battery capacity in kWh


def load_source_data():
    """Load source data from Maly.csv."""
    data_dir = Path("data")
    
    maly = pd.read_csv(data_dir / "Maly.csv", sep=';')
    maly['DatumCas'] = pd.to_datetime(maly['DatumCas'])
    
    print(f"Maly: {len(maly)} records (5-minute intervals)")
    
    return maly


def process_to_5min(maly):
    """Process data to 5-minute intervals in kWh."""
    
    print("\nProcessing Maly.csv...")
    
    # CONSUMPTION: W -> kWh per 5 min
    maly['consumption_kwh'] = maly['P_HOME'].abs() * 5 / 60 / 1000
    
    # PRODUCTION: W -> kWh per 5 min (positive values only)
    maly['production_kwh'] = maly['P_PV'].clip(lower=0) * 5 / 60 / 1000
    
    # BATTERY: SOC (%) -> kWh (using BATTERY_SIZE_KWH)
    maly['battery_kwh'] = maly['SoC'].clip(0, 100) / 100 * BATTERY_SIZE_KWH
    
    # Format datetime
    maly['datetime'] = maly['DatumCas'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Final dataset (all in kWh)
    datasource = maly[['datetime', 'consumption_kwh', 'production_kwh', 
                        'battery_kwh']].copy()
    datasource = datasource.dropna().sort_values('datetime').reset_index(drop=True)
    
    print(f"\nDataset: {len(datasource)} rows x {len(datasource.columns)} columns")
    
    return datasource


def save_datasource(df, output_path):
    """Save dataset to CSV."""
    df.to_csv(output_path, sep=";", index=False)
    print(f"\n✓ Saved: {output_path}")


def main():
    print("=" * 70)
    print("CREATING DATASOURCE (5-min intervals, all in kWh)")
    print("=" * 70)
    print(f"Battery size: {BATTERY_SIZE_KWH} kWh")
    
    # Load source data
    maly = load_source_data()
    
    # Process to 5-minute intervals
    datasource = process_to_5min(maly)
    
    # Save dataset
    output_path = Path("data") / "datasource.csv"
    save_datasource(datasource, output_path)
    
    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS (all values in kWh):")
    print("=" * 70)
    print(datasource[['consumption_kwh', 'production_kwh', 'battery_kwh']].describe())
    
    print("\n" + "=" * 70)
    print("Sample data (first 5 rows):")
    print("=" * 70)
    print(datasource.head().to_string())
    
    print("\n" + "=" * 70)
    print("COLUMNS:")
    print("=" * 70)
    for col in datasource.columns:
        print(f"  • {col}")


if __name__ == "__main__":
    main()