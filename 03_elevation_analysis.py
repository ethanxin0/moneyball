"""
author: ethan xin

filename: 03_elevation_analysis.py

description: this file finds the correlation between elevation and 
            offesnive metrics, specifically HR, Doubles Runs and OPS
"""

# Import statements
import pandas as pd
from pybaseball import team_game_logs
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt

def scrape_elevation(url):
    """
    parameters: url: string
    
    this function gathers stadium elevation and team name by
    webscrabeing a url and returns a dataframe

    returns: df_elevation: dataframe
    """
    # open the webpage
    html = urlopen(url)
    bs = BeautifulSoup(html, 'html.parser')
    
    # find all tables
    tables = bs.find_all("table")
    lst = []
    # loop through every table in website
    for table in tables:
         # find all tr classes
        rows = table.find_all("tr")
    for row in rows:    
            # for every tr class find sub td and th
           cells = row.find_all(["td", "th"])
           cell_texts = [cell.get_text(strip=True) for cell in cells]
           lst.append(cell_texts)    
    # create df and append each val to respective column
    df_elevation= pd.DataFrame()
    df_elevation["stadium"] = pd.Series(lst[i][0] for i in range(4, 35))
    df_elevation["elevation"] = pd.Series(lst[i][1] for i in range(4, 35))
    df_elevation = df_elevation[df_elevation["stadium"] != ""]
    # reset index and return the df
    df_elevation = df_elevation.reset_index(drop=True)
    return df_elevation

def correlation(df_elevation, team, years, home_elevation):
    """
    parameters: df_elevation: dataframe, elevation of each stadium
                team: string, MLB team abbrevation
                years: list, how many years user wants the correlation between
                home_elevation: int, the home stadium elevation of the 
                team user wants to pull correlation of
                                
    this function returns the correlation between elevation and 
    key offensive metrics, compatibale for every mlb team thorugh 
    custom periods of time

    returns: dataframe and correlation coefficient
    """
    # initialize an empty list to store dataframes
    dfs = []
    
    # loop through each year and get the data from pybaseball
    for year in years:
        df_year = team_game_logs(year, team)
        df_year['Year'] = year 
        # append lists to empty lists
        dfs.append(df_year) 
    
    # combine all dataframes
    team_logs = pd.concat(dfs, ignore_index=True)

    # manually replacing abbreviations for teams to join later on
    team_logs["Opp"] = team_logs["Opp"].replace('SDP', 'SD')
    team_logs["Opp"] = team_logs["Opp"].replace('WSN', 'WAS')
    team_logs["Opp"] = team_logs["Opp"].replace('KCR', 'KC')
    team_logs["Opp"] = team_logs["Opp"].replace('SFG', 'SF')
    team_logs["Opp"] = team_logs["Opp"].replace('LAA', 'ANA')
    team_logs["Opp"] = team_logs["Opp"].replace('CHW', 'CWS')
    team_logs["Opp"] = team_logs["Opp"].replace('TBR', 'TB')

    # filtering out home and away
    df_away = team_logs[team_logs["Home"] == False]
    # if away game, join the opponent abbrev on the elevation df on 
    # that away stadium
    df_away = pd.merge(df_away, df_elevation, left_on='Opp',
                       right_on='stadium')

    # if home, we add the user inputted value as the home stadium elevation
    df_home = team_logs[team_logs["Home"] == True]
    df_home['elevation'] = home_elevation

    # convert to float and combine both dfs
    df_home["elevation"] = df_home["elevation"].astype(float)
    df_away["elevation"] = df_away["elevation"].astype(float)
    df_final = pd.concat([df_home, df_away], ignore_index=True)

    # find correlation between key offensive metrics and return the df
    correlation = df_final[["R", "2B", "HR", "OPS"]].\
        corrwith(df_final['elevation'])
    return df_final, correlation

def plot(df_final):
    """
    This function plots a jitter scatter plot of the rockies 2023
    offensive metrics vs elevation of all 162 games played in the season. 
    We used jitter plots due to the fact that the x vars, elevation, 
    are very clumped """
    # add jitter to elevation values
    df_final = df_final[df_final["Year"] == 2023]
    jitter = np.random.uniform(-2500, 200, size=len(df_final))
    df_final['elevation_jittered'] = df_final['elevation'] + jitter
    
    # plot with jitter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_final, x='elevation_jittered', y='R', 
                    alpha=0.7, color='b')
    plt.title('Elevation vs Runs Scored for Rockies Games from 2011-2023 \
(exlcuding 2020) (With Jitter)')
    plt.xlabel('Elevation (feet)')
    plt.ylabel('Runs Scored')
    plt.grid()
    plt.savefig("elevation_jittered.png")
    plt.show()

def main():
    
    # call function to make dataframe
    df_elevation = scrape_elevation\
        ("https://baseballjudgments.tripod.com/id62.html")
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 
             2018, 2019, 2021, 2022, 2023]
    
    # find correlation
    df_col, corr_col = correlation(df_elevation, "COL", years, 5183)
    df_nyy, corr_nyy = correlation(df_elevation, "NYY", years, 54)
    print(f"Correlation between elevation and offensive metrics for"
          f"the Rockies from 2011- 2023 (excluding 2020)\n",corr_col)
    print(f"Correlation between elevation and offensive metrics for"
          f"the Yankees from 2011- 2023 (excluding 2020)\n", corr_nyy)
    plot(df_col)
    
if __name__ == "__main__":
    main()