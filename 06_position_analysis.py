"""
Created on Sat Aug 10 23:59:14 2024

@author: Talal Fakhoury 
NUID: 002909476
DS2000 Programming with Data
Description of assignment: Analysis on which position performanced the best
historically
"""

# Import Statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the paths for each position file, using tuples to handle multiple 
# files for the same position
position_files = {
    'C': ('all_positions_C_2023.csv', 'all_positions_C_2016.csv', 
          'all_positions_C_2019.csv'),
    '1B': ('all_positions_1B_2023.csv', 'all_positions_1B_2016.csv', 
           'all_positions_1B_2019.csv'),
    '2B': ('all_positions_2B_2023.csv', 'all_positions_2B_2016.csv', 
           'all_positions_2B_2019.csv'),
    'SS': ('all_positions_SS_2023.csv', 'all_positions_SS_2016.csv', 
           'all_positions_SS_2019.csv'),
    '3B': ('all_positions_3B_2023.csv', 'all_positions_3B_2016.csv', 
           'all_positions_3B_2019.csv'),
    'LF': ('all_positions_LF_2023.csv', 'all_positions_LF_2016.csv', 
           'all_positions_LF_2019.csv'),
    'CF': ('all_positions_CF_2023.csv', 'all_positions_CF_2016.csv', 
           'all_positions_CF_2019.csv'),
    'RF': ('all_positions_RF_2023.csv', 'all_positions_RF_2016.csv', 
           'all_positions_RF_2019.csv'),
    'DH': ('all_positions_DH_2023.csv', 'all_positions_DH_2016.csv', 
           'all_positions_DH_2019.csv')
}

FILENAME1 = 'Final_Stats.csv'

# Load the main stats file
full_statss = pd.read_csv(FILENAME1)

# Insert a new column for positions if it doesn't already exist
if "Positions" not in full_statss.columns:
    full_statss.insert(1, column="Positions", value="")

for position, files in position_files.items():
    for file in files:
        # Attempt to open the file by reading the first line
        with open(file) as f:
            f.readline()
        
        # Load the file into a DataFrame
        position_data = pd.read_csv(file)

        # Ensure that the columns are properly named for matching
        if 'last_name, first_name' in position_data.columns:
            # Use the existing 'last_name, first_name' column directly
            full_statss.loc[full_statss['last_name, first_name'].isin\
                            (position_data['last_name, first_name']), 
                            'Positions'] = position
        else:
            print(f"'last_name, first_name' column not found in {file}. \
Please check the column names.")

# Save the updated DataFrame back to CSV
full_statss.to_csv('Updated_Final_Stats.csv', index=False)

# Load the updated stats file
FILENAME = 'Updated_Final_Stats.csv'
full_stats = pd.read_csv(FILENAME)

# Define the offensive metrics to analyze
offensive_metrics = ['batting_avg', 'slg_percent', 'on_base_percent', 
                     'on_base_plus_slg', 'home_run', 'b_rbi']

# Group by position and calculate the average for each offensive metric
position_stats = full_stats.groupby('Positions')[offensive_metrics].mean()

# Multiply specific metrics by 100 to convert them to percentage format
position_stats[['batting_avg', 'slg_percent', 'on_base_percent', 
                'on_base_plus_slg']] *= 100

# Print the adjusted averages in percentage for each position
print("Averages for Each Position (Percentage Scale):")
print(position_stats.to_string())  # Print full table


def plot_radar(data, categories, title):
    """
    Plot a radar chart for the given data.
    
    Parameters:
    - data: Array-like structure containing the values for each category.
    - categories: List of category names corresponding to the data.
    - title: Title for the radar chart.
    
    Returns:
    - None: The function displays the radar chart.
    """
    num_vars = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Repeat the first value at the end to close the circle
    data = np.concatenate((data, [data[0]]))
    angles += angles[:1]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color='skyblue', alpha=0.25)
    ax.plot(angles, data, color='skyblue', linewidth=2)

    # Labels for each category
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Title
    plt.title(title, size=15, y=1.1)
    plt.show()

def main():

    # Plot radar chart for each position
    for index, row in position_stats.iterrows():
        plot_radar(row.values, offensive_metrics, f'{index} Offensive \
Metrics Radar')
        
    # Rank each position by each metric
    ranked_positions = position_stats.rank(ascending=False, method='min')
    
    # Sum ranks across all metrics to determine the overall ranking
    ranked_positions['total_rank'] = ranked_positions.sum(axis=1)
    
    # Sort by total rank
    best_positions = ranked_positions.sort_values(by='total_rank')
    
    # Print the positions sorted by their total rank
    print("\nPositions ranked by total rank (lower is better):")
    print(best_positions.to_string())
    
    # Identify the best position
    best_position = best_positions.index[0]
    print(f"\nThe best offensive position is {best_position} with a total \
rank score of {best_positions['total_rank'].iloc[0]:.0f}.")

if __name__ == "__main__":
    main()