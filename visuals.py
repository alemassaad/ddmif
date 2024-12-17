import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

# Step 1: Load and combine all test iteration files
def combine_iteration_files(folder_path):
    combined_df = pd.DataFrame()
    files = sorted(glob.glob(os.path.join(folder_path, "test_iteration_*.csv")))
    last_AUM = 1.0
    last_SPX = 1.0
    
    for i, file in enumerate(files):
        df = pd.read_csv(file, parse_dates=["caldt"])
        df = df[["caldt", "AUM", "AUM_SPX"]].copy()
        
        # Scale AUM to continue from the previous iteration
        df["AUM"] = df["AUM"] * last_AUM
        df["AUM_SPX"] = df["AUM_SPX"] * last_SPX
        
        # Update the last AUM for scaling the next iteration
        last_AUM = df["AUM"].iloc[-1]
        last_SPX = df["AUM_SPX"].iloc[-1]
        
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df

# Step 2: Plot cumulative AUM for strategy and market as dots
def plot_aum_curves_as_dots(combined_df):
    plt.figure(figsize=(12, 6))
    plt.scatter(combined_df["caldt"], combined_df["AUM"], label="Strategy AUM", color='blue', s=10)
    plt.scatter(combined_df["caldt"], combined_df["AUM_SPX"], label="Market AUM (SPX)", color='red', s=10)
    
    plt.title("Cumulative AUM: Strategy vs Market (Dots)")
    plt.xlabel("Date")
    plt.ylabel("AUM")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Step 3: Main Execution
if __name__ == "__main__":
    folder_path = "results"  # Update to your directory containing the test_iteration_*.csv files
    combined_df = combine_iteration_files(folder_path)
    plot_aum_curves_as_dots(combined_df)
