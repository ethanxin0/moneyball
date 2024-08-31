"""
Author        : Josh Longo
Course        : DS2500: Intermediate Programming with Data
Filename      : 01_predictive_analysis.py
Creation date : Tue Aug  6 01:26:18 2024

Description   : Analysis #1 of project work that creates a predictive model
using multiple linear regression to predict BA, OBP, SLG
"""

# Main Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as mtick

# Sklearn imports for regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# append system path to import utils functions
import sys
sys.path.append("../week3")
import utils

def get_filenames(dirname, ext = ".csv"):
    ''' given the name of a directory (string), return a list
    of paths to all the ****.ext files in that directory'''
    filenames = []
    files = os.listdir(dirname)
    for file in files:
        if file.endswith(ext):
            filenames.append(dirname + "/" + file)
    return filenames

def act_pred_scat_plot(y_test, predictions, rmse, y):
    '''This function accepts the series for y_test and a numpy array of 
    predictions, rmse for the data being predicted, and which data is being 
    predicted in a string, then plots an actual vs predicted scatter plot.'''
    # Plot the figure    
    plt.figure(figsize=(12, 8), dpi=300)
    sns.set_theme(style="whitegrid")
    scatter = sns.scatterplot(x=y_test, y=predictions, color='#2a9d8f')
    # Regression line
    sns.regplot(x=y_test, y=predictions, scatter=False, color='#e76f51', 
                line_kws={"linewidth": 1})
    # Reference line (y=x) - perfect prediction line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         linestyle='--', color='black', linewidth=1)
    plt.xlabel(f'Actual {y} (Decimal)', fontsize=13)
    plt.ylabel(f'Predicted {y} (Decimal)', fontsize=13)
    plt.title(f'Actual vs. Predicted {y} for MLB Player Offensive Data \
(2015-2023, Excluding 2020)', fontsize=17)
    # Add RMSE annotation
    plt.text(0.05, 0.95, f'RMSE: {rmse}', fontsize=12, ha='left', 
             va='center', transform=scatter.transAxes, 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    # Format the y-axis labels to show 3 decimal places
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    y_format = y.replace(" ", "_")
    utils.save_plot_to_subdir(f'{y_format}_accuracy_plot.png')
    plt.show()
    
def train_predict_model(df, independent_vars, dependent_var):
    """Train a linear regression model and evaluate it by calculating RMSE.
    Parameters - df: DataFrame containing the data, independent_vars: List of 
    column name(s) to be used as independent variables, dependent_var: Column 
    name to be used as the dependent variable.
    Returns - model: Trained linear regression model, rmse: root mean square 
    error, coefficients: Coefficients of the trained linear regression model, 
    X_test: Test set of independent variables, y_test: Test set actual values,
    predictions: Predicted values for the test set."""
    # Ensure independent_vars is a list
    if isinstance(independent_vars, str):
        independent_vars = [independent_vars]
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[independent_vars], 
                                            df[dependent_var], random_state=0)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(utils.mse(list(predictions), list(y_test)))
    coefficients = model.coef_
    return model, rmse, coefficients, X_test, y_test, predictions

def main():
    
    # Define the directory containing the CSV files
    dirname = 'batting_stats'
    
    # Get the list of CSV filenames in the directory
    files = get_filenames(dirname, ext=".csv")
    
    # Read all the CSV files into DataFrames
    dataframes = utils.read_multiple_csv(files)
    
    # dataframe with all batting data for qualified batters (>500 pa's)
    df_concat = pd.concat(dataframes, ignore_index=True)
    
    # Establish features for x for batting average, OBP, OPS
    ind_v_ba = ['single', 'k_percent', 'slg_percent','exit_velocity_avg', 
                'launch_angle_avg', 'whiff_percent', 'swing_percent', 
                'linedrives_percent', 'popups_percent', 'meatball_percent', 
                'barrel_batted_rate', 'sweet_spot_percent', 
                'solidcontact_percent', 'hard_hit_percent',
                'swing_percent', 'oz_swing_miss_percent']
    ind_v_obp = ['k_percent', 'bb_percent', 'babip','exit_velocity_avg', 
                'launch_angle_avg', 'sweet_spot_percent','barrel_batted_rate', 
                'whiff_percent', 'solidcontact_percent', 'meatball_percent', 
                'sweet_spot_percent', 'hard_hit_percent',
                'z_swing_miss_percent', 'oz_contact_percent']
    ind_v_slg = ['isolated_power', 'barrel_batted_rate', 'bb_percent',
                 'exit_velocity_avg', 'launch_angle_avg', 'sweet_spot_percent',
                 'barrel_batted_rate','solidcontact_percent', 
                 'sweet_spot_percent', 'hard_hit_percent','oz_contact_percent',
                 'linedrives_percent', 'flyballs_percent']
    
    # Train the model and return the model, rmse, coefficients, X_test, 
    # y_test, and predictions
    model_ba, rmse_ba, coef_ba, X_test_ba, y_test_ba, predictions_ba = \
        train_predict_model(df_concat, ind_v_ba, "batting_avg")
        
    model_obp, rmse_obp, coef_obp, X_test_obp, y_test_obp, predictions_obp = \
        train_predict_model(df_concat, ind_v_obp, "on_base_percent")
    model_slg, rmse_slg, coef_slg, X_test_slg, y_test_slg, predictions_slg = \
        train_predict_model(df_concat, ind_v_slg, "slg_percent")
    
    # Print RMSE for batting average, OBP, and OPS predictions
    print(f"RMSE for batting average: {rmse_ba}")
    print(f"RMSE for OBP: {rmse_obp}")
    print(f"RMSE for SLG: {rmse_slg}")

    # Plot the actual versus predicted batting averages, OBP, OPS
    act_pred_scat_plot(y_test_ba, predictions_ba, rmse_ba, 'Batting Average')
    act_pred_scat_plot(y_test_obp, predictions_obp, rmse_obp, 'OBP')
    act_pred_scat_plot(y_test_slg, predictions_slg, rmse_slg, 'Slugging %')

if __name__ == "__main__":
    main()
