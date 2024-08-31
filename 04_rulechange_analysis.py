"""
author: ethan xin

filename: 04_rulechange_analysis.py

description: This analysis looks at the affect of rule changes made in the MLB
on offensive metrics
"""

# Import statements
import pandas as pd
import numpy as np
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from textblob.blob import TextBlob
from bs4 import BeautifulSoup
from utils import read_csv, normalize, mse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def scrape_team_hitting_stats(years):
    """
    parameters: years: list, # of years users would like to scrape

    this function scrapes mlb offensive team data from foxsports.com, 
    values are updated daily

    returns: dataframe
    """
    # initlialize DF
    all_data = pd.DataFrame()
    # loop thorugh every year
    for year in years:
        # open link to specified year using f-string
        url = f"https://www.foxsports.com/mlb/team-stats?season={year}" \
            "&category=batting"
        html = urlopen(url)
        bs = BeautifulSoup(html, 'html.parser')
        
        # find all tables
        tables = bs.find_all("div")
        lst = []
        for table in tables:
            # find all tr classes
            rows = table.find_all("tr")
            for row in rows:
               cells = row.find_all(["td", "th", "a"])
               cell_texts = [cell.get_text(strip=True) for cell in cells]
               lst.append(cell_texts)
        # appening statistics to respective column
        df_team_stats = pd.DataFrame()
        lst_cleaned = [[cell for cell in row if cell != ""] for row in lst]
        df_team_stats["team"] = pd.Series(lst_cleaned[i][1] for i in 
                                          range(1, 31))
        df_team_stats["games"] = pd.Series(lst_cleaned[i][3] for i in 
                                           range(1, 31))
        df_team_stats["avg"] = pd.Series(lst_cleaned[i][16] for i in 
                                         range(1, 31))
        df_team_stats["hits"] = pd.Series(lst_cleaned[i][6] for i in 
                                          range(1, 31))
        df_team_stats["runs"] = pd.Series(lst_cleaned[i][7] for i in 
                                          range(1, 31))
        df_team_stats["doubles"] = pd.Series(lst_cleaned[i][8] for i in 
                                             range(1, 31))
        df_team_stats["hr"] = pd.Series(lst_cleaned[i][10] for i in 
                                        range(1, 31))
        df_team_stats["rbi"] = pd.Series(lst_cleaned[i][11] for i in 
                                         range(1, 31))
        df_team_stats["steals"] = pd.Series(lst_cleaned[i][12] for i in 
                                            range(1, 31))
        df_team_stats["caught_steals"] = pd.Series(lst_cleaned[i][13] for i in 
                                                   range(1, 31))
        df_team_stats["so"] = pd.Series(lst_cleaned[i][15] for i in 
                                        range(1, 31))
        df_team_stats["year"] = year
        #combine and return
        all_data = pd.concat([all_data, df_team_stats], ignore_index=True)
    return all_data

def format_df(df):
    """
    parameters: df: dataframe

    this function formats the dataframe into columns that are
    neccesary for calculations for both plots

    returns: df_total: dataframe

    """
    # convert cols to numeric
    df['steals'] = pd.to_numeric(df['steals'], errors='coerce')
    df['caught_steals'] = pd.to_numeric(df['caught_steals'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['runs'] = pd.to_numeric(df['runs'], errors='coerce')

    # calculate attempted steals
    df['attempted_steals'] = df['steals'] + df['caught_steals']

    # aggregate data by year
    df_total = df.groupby('year').agg({'runs': 'sum','games': 'sum',
                                      'attempted_steals': 'sum'}).reset_index()

    # calculate runs per game and attempted steals per game
    df_total["runs_per_game"] = df_total["runs"] / df_total["games"] * 2
    df_total["attempt_per_game"] = \
        df_total["attempted_steals"] / df_total["games"] * 2

    return df_total

def plot_1(df):
    '''Plots a bar graph to show attempted steals per game before and after
    the pitch clock rule was implemented'''
    # plotting, axising, labeling
    years = df['year']
    colors = ['mediumseagreen' if year in [2023, 2024] else 'dodgerblue' if 
              year in [2025,2026] else 'salmon' for year in years]
    
    # Create a bar plot without the label parameter
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x=years, height=df["attempt_per_game"], color=colors)
    
    # Define the unique labels
    unique_labels = {
        'mediumseagreen': 'Post-Pitch-Clock',
        'dodgerblue': 'Predictions-With-Pitch-Clock',
        'salmon': 'Pre-Pitch-Clock'}
    
    # Create custom handles and labels for the legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color
               in unique_labels]
    labels = [unique_labels[color] for color in unique_labels]
    
    # Add the legend manually
    plt.legend(handles, labels, fontsize=9)
    plt.xlabel("Year")
    plt.ylabel("Attempted Steals per game")
    plt.title("Attempted steals per game before and after pitch clock \
implementation (Regular Season)")
    plt.savefig("steals_per_game.png")
    plt.show()

    # Calculate correlation between attempt_per_game and 'year'
    df_focused = df[df["year"].isin([2021, 2022, 2023, 2024])]
    correlation = df_focused[["attempt_per_game"]].corrwith(df["year"])
    
    return correlation

def plot_2(df):
    '''Plots a bar graph to show the runs per game before and after the 
    pitch clock was implemented'''
    # define colors and labels
    years = df['year']
    colors = ['green' if year in [2023, 2024] else 'red' for year in years]
    
    # Create a bar plot without the label parameter
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x=years, height=df["runs_per_game"], color=colors)
    
    # Define the unique labels
    unique_labels = {
        'green': 'Post-Pitch-Clock',
        'red': 'Pre-Pitch-Clock' }
    
    # Create custom handles and labels for the legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in 
               unique_labels]
    labels = [unique_labels[color] for color in unique_labels]
    
    # Add the legend manually
    plt.legend(handles, labels, fontsize=8)
    plt.xlabel("Year")
    plt.ylabel("Runs per game")
    plt.title("Runs per game before and after pitch clock implementation \
(Regular Season)")
    plt.ylim(bottom=0, top=13)
    plt.savefig("runs_pitch_clock.png")
    plt.show()

    # Calculate correlation between runs_per_game and 'year'
    df_focused = df[df["year"].isin([2021, 2022, 2023, 2024])]
    correlation = df_focused[["runs_per_game"]].corrwith(df_focused["year"]) 
    return correlation
    
def analysis(df):
    """
    parameters: df: dataframe

    this function predicts the amount of stolen bases in the next two years

    returns df: dataframe, used for plotting
            r2_score, float, the r^2 value of the predictions of 
            linear regression RMSE, float, RMSE of the predictions
    """
    # intialize vals for regression        
    x = df[["year"]]
    y = df["attempt_per_game"]

    # initialize and fit model
    model = LinearRegression()
    model.fit(x, y)
    
    # make predictions for 2025 and 2026
    y_pred = model.predict(x)
    future_years = pd.DataFrame({'year': [2025, 2026]})
    predictions = model.predict(future_years)

    # combine predictoins and inputted df for plotting
    future_years['attempt_per_game'] = predictions
    df = pd.concat([future_years, df], ignore_index = True)
    df = df.sort_values(by='year', ascending=True)
    df = df.reset_index()
    return df, r2_score(y,y_pred), mse(y_pred,y) ** 0.5

def main():
    
    df = scrape_team_hitting_stats([2019,2020,2021,2022,2023,2024])
    df = format_df(df)

    # plotting predictions and statistics for plot 1
    df_pred, r2_steals, rmse_steals = analysis(df)
    corr1 = plot_1(df_pred)
    print("Correlation from years 2021-2024 and steal", corr1)
    print(f"The r^2 score for this predictive regression is {r2_steals} and \
the RMSE is {rmse_steals}\n")

    # plotting plot two
    corr2 = plot_2(df)
    print("\n\n\n\nCorrelation from years 2021-2024 and", corr2)
    
if __name__ == "__main__":
    main()