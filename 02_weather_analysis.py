"""
Author        : Josh Longo
Course        : DS2500: Intermediate Programming with Data
Filename      : 02_weather_analysis.py
Creation date : Wed Aug  7 03:39:21 2024

Description   : Relationship between weather metrics and offensive baseball
statistics.
"""

# Import Statements
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import numpy as np
from datetime import datetime, timedelta
import pytz
import statsmodels.api as sm

# Baseball Scraping Libraries
from pybaseball import team_game_logs
import statsapi

# Imports for graphing
import statistics 
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.ticker as mtick

# append system path to import utils functions
import sys
sys.path.append("../week3")
import utils

# Establish global variables
YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
BOS_ID = 111

def get_schedule_utc_minus_4(start_date, end_date, team):
    # Fetch the schedule data using the statsapi.schedule function
    schedule = statsapi.schedule(start_date=start_date, end_date=end_date, 
                                 team=team)

    # Define the UTC timezone and the desired UTC-4 timezone
    utc_timezone = pytz.utc
    utc_minus_4_timezone = pytz.timezone('America/New_York')
    
    # Iterate through each game in the schedule and convert the datetime
    for game in schedule:
        if 'game_datetime' in game:
            # Convert the game_datetime string to a datetime object
            game_datetime = datetime.strptime(game['game_datetime'], 
                                              '%Y-%m-%dT%H:%M:%SZ')
            # Set the timezone to UTC
            game_datetime = utc_timezone.localize(game_datetime)
            # Convert the datetime to UTC-4
            game['game_datetime'] = game_datetime.astimezone\
                (utc_minus_4_timezone).strftime('%Y-%m-%d %H:%M:%S')
    return schedule

def get_filtered_schedule(year, team_id):
    """Fetch and filter the schedule data for a given year and team."""
    sched = pd.DataFrame(get_schedule_utc_minus_4(f'01/01/{year}', 
                                                  f'12/31/{year}', team_id))
    
    # Filter for regular season home games that are not doubleheader games
    sched = sched[sched['home_id'] == team_id]
    sched = sched[sched['game_type'] == 'R']
    sched = sched[sched['doubleheader'] == 'N']
    sched = sched[sched['status'] != 'Postponed']
    return sched

def combine_schedules(years, team_id):
    """Combine schedules across multiple years into a single DataFrame."""
    all_years_data = []
    for year in years:
        sched1 = get_filtered_schedule(year, team_id)
        all_years_data.append(sched1)
        
    # Combine all the yearly DataFrames into a single DataFrame
    final_sched = pd.concat(all_years_data, ignore_index=True)
    return final_sched

def remove_incorrectly_labeled_games(df):
    """Remove games that are incorrectly labeled."""
    # Drop the games that are incorrectly labeled
    df = df[(df['game_datetime'] != '2023-07-22 14:10:00') &
            (df['game_datetime'] != '2019-08-22 13:05:00')]
    return df

def process_team_game_logs(years, team):
    """Process game logs for a team across multiple years."""
    data_frames = []
    for year in years:
        df = team_game_logs(year, team)
        df['year'] = year
        data_frames.append(df)
    
    # Concatenate all DataFrames into one
    df_concat = pd.concat(data_frames, ignore_index=True)
    return df_concat

def clean_game_logs(df):
    """Clean and process the game logs DataFrame."""
    # Filter for home games and create a copy
    df = df[df['Home'] == True].copy()
    
    # Clean the 'Date' column by removing doubleheader notations and anomalies
    df['Date'] = df['Date'].str.replace(r'\s*\(.*\)', '', regex=True).str.\
        replace('susp', '', regex=True).str.strip()
    
    # Replace hyphen with a space and combine with the year
    df['Date'] = df['year'].astype(str) + ' ' + df['Date']
    
    # Convert to datetime format and reformat to 'YYYY-MM-DD'
    df['Date'] = pd.to_datetime(df['Date'], format='%Y %b %d', errors='coerce')
    df['game_date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Remove rows with duplicated dates
    df = df.drop_duplicates(subset=['game_date'], keep=False)
    return df

def calculate_ba(df):
    '''Calculate team batting average for game log df.'''
    return df['H'] / df['AB']

def calculate_obp(df):
    '''Calculate team on base percentage for game log df.'''
    numerator = df['H'] + df['BB'] + df['HBP']
    denominator = df['AB'] + df['BB'] + df['HBP'] + df['SF']  
    return numerator / denominator

def calculate_slg(df):
    '''Calculate team slugging percentage for game log df.'''
    singles = df['H'] - df['2B'] - df['3B'] - df['HR']
    total_bases = singles + 2 * df['2B'] + 3 * df['3B'] + 4 * df['HR']
    return total_bases / df['AB']

def calculate_ops(df):
    '''Calculate on base plus slugging for game log df.'''
    return df['rOBP'] + df['rSLG']

def calculate_weather_averages(m_df, weather_df):
    '''Accepting the merged game log df and the weather df, calculate the 
    average weather metrics for each game over a 3-hour window after the 
    start time. Returns df with average weather metrics'''
    weather_averages = []
    for index, row in m_df.iterrows():
        start_time = row['start_time_rounded']
        # 3 hour span is equal to 4 rows of data
        end_time = start_time + pd.Timedelta(hours=4)
        
        # Filter weather data for the 3-hour time span after the start time
        weather_window = weather_df[(weather_df['date'] >= start_time) & 
                                    (weather_df['date'] < end_time)]
        
        # Calculate the averages
        avg_temperature = weather_window['temperature_2m'].mean()
        avg_humidity = weather_window['relative_humidity_2m'].mean()
        avg_apparent_temp = weather_window['apparent_temperature'].mean()
        avg_precipitation = weather_window['precipitation'].mean()
        avg_wind_speed = weather_window['wind_speed_10m'].mean()
    
        # Append the averages to the list
        weather_averages.append({
            'game_id': row['game_id'],
            'avg_temperature': avg_temperature,
            'avg_humidity': avg_humidity,
            'avg_apparent_temp': avg_apparent_temp,
            'avg_precipitation': avg_precipitation,
            'avg_wind_speed': avg_wind_speed
        })
    weather_averages_df = pd.DataFrame(weather_averages)
    return weather_averages_df

def corr_scat_plot(x, y, x_name, y_name, color=None):
    '''Create a plot to show relationship between a weather metric 
    and offensive baseball metric.'''
    # Calculate correlation coefficient and r^2
    r = statistics.correlation(x, y)
    r_sq = r**2
    if color is None:
        sns.set_palette('muted')
        color = sns.color_palette()[0]
    # Create scatter plot with regression line
    plt.figure(figsize=(11,7), dpi=300)
    sns.set_style("whitegrid")
    scatter = sns.regplot(x=x, y=y, color=color)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{y_name} vs. {x_name} for Boston Red Sox Home Games \
(2013-2023, Excluding 2020)')
    # Format the y-axis labels to show 3 decimal places
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # Create a custom legend for r and r^2
    legend_text = f'$r = {r:.4f}$\n$r^2 = {r_sq:.4f}$'
    # Add r and r squared annotation
    plt.text(0.04, 0.90, legend_text, fontsize=12, ha='left', 
             va='center', transform=scatter.transAxes, 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    y_format = y_name.replace(" ", "_").replace("%", "percent").\
        replace("(", "").replace(")", "").replace("Decimal", "")
    utils.save_plot_to_subdir(f'{y_format}corr_plot.png')
    plt.show()

def run_regression(df, x_vars, y_vars):
    """
    Accepts dataframe of all data, a list of x variables and a y variable,
    then runs an OLS regression, prints summary, and returns the model."""
    X = df[x_vars]
    y = df[y_vars] 

    # Add constant to independent variables matrix to represent the intercept
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    print(f"Results for {y_vars}:")
    print(model.summary())
    return model, X, y
    
def plot_coefficients(model, X, y_var):
    """
    Plots the coefficients of the regression model.
    """
    # Plot the coefficients
    coefficients = model.params[1:]  # Exclude the constant term
    features = X.columns[1:]  # Exclude the constant term
    plt.bar(features, coefficients)
    plt.ylabel('Coefficient Value')
    plt.title(f'Coefficient Values for {y_var}')
    plt.show()
    
def full_analysis(f_game_df, x_vars, y_vars):
    """
    Runs the full analysis for each dependent variable specified.
    """
    for y_var in y_vars:
        model, X, y = run_regression(f_game_df, x_vars, y_var)
        plot_coefficients(model, X, y_var)

def main():
    
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    
    # Make sure all required weather variables are listed here. The order of 
    # variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
    	"latitude": 42.346268,
    	"longitude": -71.095764,
    	"start_date": "2013-04-08",
    	"end_date": "2023-10-02",
    	"hourly": ["temperature_2m", "relative_humidity_2m", 
                "apparent_temperature", "precipitation", "wind_speed_10m"],
    	"temperature_unit": "fahrenheit",
    	"wind_speed_unit": "mph",
    	"precipitation_unit": "inch",
    	"timezone": "America/New_York"
    }
    responses = openmeteo.weather_api(url, params=params)
    
    # Process first location.
    response = responses[0]
    
    # Process hourly data. The order of variables needs to be the same as 
    # requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(2).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
    
    # Create hourly data with timezone info removed
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ).tz_convert("America/New_York").tz_localize(None)
    }
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["apparent_temperature"] = hourly_apparent_temperature
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    
    # Create weather df
    weather_df = pd.DataFrame(data = hourly_data)
    
    # Combine schedules for all years except 2020
    final_sched = combine_schedules(YEARS, team_id=BOS_ID)
    
    # Remove incorrectly labeled games
    final_sched = remove_incorrectly_labeled_games(final_sched)

    # Process team game logs for all years
    team_logs = process_team_game_logs(YEARS, "BOS")
    
    # Clean and process the concatenated DataFrame
    team_logs = clean_game_logs(team_logs)
    
    # merge the two dataframes and select the desired columns
    m_df = pd.merge(final_sched, team_logs, on='game_date')
    
    # Round the start time to the nearest hour to get in line with weather df
    m_df['start_time_rounded'] = pd.to_datetime\
        (m_df['game_datetime']).dt.round('h')
    
    # Calculate the accurate metrics using table data
    m_df['rBA'] = calculate_ba(m_df)
    m_df['rOBP'] = calculate_obp(m_df)
    m_df['rSLG'] = calculate_slg(m_df)
    m_df['rOPS'] = calculate_ops(m_df)
    
    # Create a df to store game weather averages
    weather_averages_df = calculate_weather_averages(m_df, weather_df)
    
    # Merge the weather averages back into the merged game log df
    f_game_df = pd.merge(m_df, weather_averages_df, on='game_id')
    
    # Establish the x and y vars for analysis
    x_vars = ['avg_temperature', 'avg_humidity', 'avg_wind_speed']
    y_vars = ['rBA', 'rOBP', 'rOPS', 'rSLG']
    
    # run the regression analysis and plot act vs pred. and coefficients
    full_analysis(f_game_df, x_vars, y_vars)
    
    # Focus on only rOPS/rSLG and avg tempature, and create correlation graphs
    corr_scat_plot(f_game_df['avg_temperature'], f_game_df['rOPS'], 
                    'Avg Temperature (°F)', 'On-Base Plus Slugging (Decimal)', 
                    '#FF4500')
    corr_scat_plot(f_game_df['avg_temperature'], f_game_df['rSLG'], 
                    'Avg Temperature (°F)', 'Slugging % (Decimal)', '#1E90FF')
    
if __name__ == "__main__":
    main()