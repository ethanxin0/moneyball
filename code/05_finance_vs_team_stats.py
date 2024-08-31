# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:18:02 2024

@author: eitan
"""

# Import Statements
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from pybaseball import standings
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def fetch_html(url):
    """
    Fetch HTML content from a given URL.

    Args:
        url (str): The URL to fetch the HTML content from.

    Returns:
        BeautifulSoup: Parsed HTML content.
    """
    response = urlopen(url)  # Fetch the HTML response
    return BeautifulSoup(response, 'html.parser')  # Parse and return HTML

def extract_tables(soup):
    """
    Extract all tables from the HTML content and 
    convert them to a list of lists.
    Args:
        soup (BeautifulSoup): Parsed HTML content.

    Returns:
        list: A list of tables, each as a list of lists containing cell text.
    """
    tables = soup.find_all("table")  # Find all table elements
    table_data_list = []

    for table in tables:
        rows = table.find_all("tr")  # Find all rows in the table
        # Extract text from each cell in each row
        table_data = [[cell.get_text(strip=True) for cell in row.find_all(
            ["td", "th"])] for row in rows]
        table_data_list.append(table_data)  # Append table data to the list

    return table_data_list

def create_dataframes(table_2, table_3):
    """
    Create DataFrames from the second and third tables, and merge 
    them on team names.
    """
    # Create DataFrames from the provided tables
    df_table_2 = pd.DataFrame(table_2[1:], columns=table_2[0])
    df_table_3 = pd.DataFrame(table_3[1:], columns=table_3[0])

    # Merge the DataFrames on the 'Team' column
    df_combined = pd.merge(df_table_2, df_table_3, left_on='Team', 
                           right_on='', how="inner")
    
    # Select relevant columns for revenue and attendance
    df_revenue = df_combined[['Team', 'Market size (M)', 
                              '2023 revenue (M)']].copy()
    df_attendance = df_combined[['Team', '2023 attendance']].copy()

    # Merge the revenue and attendance DataFrames on 'Team'
    return pd.merge(df_revenue, df_attendance, on='Team', how='inner')

def clean_team_names(df):
    """
    Clean and replace team names in the DataFrame.
    """
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure 'team' column exists
    if 'team' not in df.columns:
        raise KeyError("Column 'team' not found in DataFrame")
    
    def replace_team_names(team):
        words = team.split()
        if len(words) > 1:
            last_word = words[-1]
            if df['team'].apply(lambda x: x.split()[-1]).tolist().count(
                    last_word) > 1:
                return ' '.join(words[-2:])
            return last_word
        return team

    # Apply the function to clean 'team' names
    df['team'] = df['team'].apply(replace_team_names)
    # Replace specific team name
    df['team'] = df['team'].replace("A's", "Athletics")

    return df

def scrape_team_stats(url):
    """
    Scrape team stats from the second website and return a DataFrame.
    """
    soup = fetch_html(url)  # Fetch HTML content from the URL
    tables = soup.find_all("div")  # Find all div elements

    lst = []
    for table in tables:
        # Find all rows in each div
        rows = table.find_all("tr") 
        for row in rows:
            # Find all cells in each row
            cells = row.find_all(["td", "th", "a"])  
            # Extract text from cells
            cell_texts = [cell.get_text(strip=True) for cell in cells]  
            lst.append(cell_texts)

    df_team_stats = pd.DataFrame()
    # Remove empty cells
    lst_cleaned = [[cell for cell in row if cell != ""] for row in lst]  

    # Create DataFrame columns from cleaned list
    df_team_stats["team"] = pd.Series(lst_cleaned[i][1] for i in range(1, 31))
    df_team_stats["avg"] = pd.Series(lst_cleaned[i][16] for i in range(1, 31))
    df_team_stats["hits"] = pd.Series(lst_cleaned[i][6] for i in range(1, 31))
    df_team_stats["runs"] = pd.Series(lst_cleaned[i][7] for i in range(1, 31))
    df_team_stats["doubles"] = pd.Series(lst_cleaned[i][8] for i in
                                         range(1, 31))
    df_team_stats["hr"] = pd.Series(lst_cleaned[i][10] for i in range(1, 31))
    df_team_stats["rbi"] = pd.Series(lst_cleaned[i][11] for i in range(1, 31))

    return df_team_stats

def fetch_standings_data():
    """
    Fetch and process standings data using pybaseball library.
    """
    data1 = standings(2023)  # Fetch standings data for 2023
    data1 = pd.concat(data1)  # Concatenate the data into a single DataFrame
    data = data1[['Tm', 'W-L%']].copy()  # Select relevant columns

    def replace_team_names(team):
        words = team.split()
        if len(words) > 1:
            last_word = words[-1]
            if data['Tm'].apply(lambda x: x.split()[-1]).tolist().count(
                    last_word) > 1:
                return ' '.join(words[-2:])
            return last_word
        return team
    # Clean team names
    data.loc[:, 'Tm'] = data['Tm'].apply(replace_team_names) 
    return data

def merge_dataframes(df_final, df_team_stats, data):
    """
    Merge the three DataFrames on their respective team columns and 
    drop unnecessary columns.
    """
    # Standardize column names to lowercase for consistency
    df_final.columns = df_final.columns.str.lower()
    df_team_stats.columns = df_team_stats.columns.str.lower()
    data.columns = data.columns.str.lower()

    # Verify column names
    print("df_final columns:", df_final.columns)
    print("df_team_stats columns:", df_team_stats.columns)
    print("data columns:", data.columns)
    
    # Merge df_final and df_team_stats on 'team'
    merged_df = pd.merge(df_final, df_team_stats, left_on='team', 
                         right_on='team')
    
    # Merge the resulting DataFrame with data on 'team' and 'Tm'
    final_df = pd.merge(merged_df, data, left_on='team', right_on='tm')
    
    # Drop unnecessary columns
    final_df = final_df.drop(columns=['tm', 'team'])
    
    return final_df


def calculate_pearson_coefficients(final_df):
    """
    Normalize specified columns and calculate Pearson coefficients.
    """
    # Remove dollar signs and commas, then convert to numeric
    for column in ['hr', 'hits', 'w-l%', 'market size (m)', '2023 revenue (m)', 
                   '2023 attendance']:
        if final_df[column].dtype == 'object':
            final_df[column] = final_df[column].str.replace('$', '').\
                str.replace(',', '').astype(float)

    # Columns to normalize
    columns_to_normalize = ['hr', 'hits', 'w-l%']

    # Normalize specific columns
    scaler = MinMaxScaler()
    final_df[columns_to_normalize] = scaler.fit_transform(final_df[
        columns_to_normalize])

    return final_df

def plot_correlation_matrix(final_df):
    """
    Calculate the Pearson correlation coefficient and plot the bottom-left
    3x3 section of the matrix.
    """
    # Select specific columns for correlation calculation
    columns_to_correlate = ['hr', 'hits', 'w-l%', 'market size (m)', 
                            '2023 revenue (m)', '2023 attendance']
    selected_df = final_df[columns_to_correlate]

    # Calculate Pearson correlation coefficient
    correlation_matrix = selected_df.corr()

    # Slice the bottom-left 3x3 section of the correlation matrix
    bottom_left_3x3 = correlation_matrix.loc[
        ['market size (m)', '2023 revenue (m)', '2023 attendance'], 
        ['hr', 'hits', 'w-l%']
    ]

    # Create and display a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(bottom_left_3x3, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap (2023)')
    plt.savefig('corr_matrix_heatmap.png')
    plt.show()

    return correlation_matrix


def plot_regressions(final_df):
    """
    Create scatter plots with regression lines for each pair of independent 
    and dependent variables.
    """
    independent_vars = ['hr', 'hits', 'w-l%']
    dependent_vars = ['market size (m)', '2023 revenue (m)', '2023 attendance']

    # Create scatter plots with regression lines for each variable pair
    for independent_var in independent_vars:
        for dependent_var in dependent_vars:
            plot = sns.lmplot(x=independent_var, y=dependent_var, 
                              data=final_df)
            plt.title(f'Scatter Plot with Regression Line: {independent_var} \
vs {dependent_var}')
            plot.savefig(f'{independent_var}_vs_{dependent_var}.png')
            plt.show()

def main():
    # First website
    url_1 = "https://www.blessyouboys.com/2024/4/19/24134946/the-business" \
        "-of-baseball-2024-edition"
    soup_1 = fetch_html(url_1)
    tables_1 = extract_tables(soup_1)

    if len(tables_1) >= 3:
        df_final = create_dataframes(tables_1[1], tables_1[2])
        df_final = clean_team_names(df_final)
        print("Final DataFrame from first website:\n", df_final)
    
    # Second website
    url_2 = "https://www.foxsports.com/mlb/team-stats?season=2023&" \
        "category=batting"
    df_team_stats = scrape_team_stats(url_2)
    df_team_stats = clean_team_names(df_team_stats)
    print("Team Stats DataFrame from second website:\n", df_team_stats)

    # Standings data using pybaseball
    standings_data = fetch_standings_data()
    print("Standings DataFrame from pybaseball:\n", standings_data)

    # Merge all three DataFrames
    final_df = merge_dataframes(df_final, df_team_stats, standings_data)
    print("Final Combined DataFrame:\n", final_df)

    # Calculate Pearson coefficients
    final_df = calculate_pearson_coefficients(final_df)

    # Plot correlation matrix
    correlation_matrix = plot_correlation_matrix(final_df)
    print("Correlation Matrix:\n", correlation_matrix)
    
    # Create scatter plots with regression lines
    plot_regressions(final_df)

if __name__ == "__main__":
    main()