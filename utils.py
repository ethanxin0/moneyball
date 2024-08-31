"""
Author        : Josh Longo
Course        : DS2500: Intermediate Programming with Data
Filename      : utils.py
Creation date : Tue Jul 16 14:15:39 2024

Description   : Utility Function
"""

import csv
import os
import shutil
import matplotlib.pyplot as plt
import statistics
import math
import pandas as pd


FILENAME = "data/speed_restrictions.csv"

def read_csv(filename):
    '''Reads the CSV file and returns a list of lists.
    '''
    data = []
    
    with open(filename, 'r') as infile:
        csvfile = csv.reader(infile)
        for row in csvfile:
            data.append(row)  
    return data

def lst_to_dct(lst):
    '''creates a dictionary from a 2d list, where
    the keys are the header (1st item in the list),
    and the values are columns.
    '''
    # create a dictionary
    dct = {}
    
    # headers from the list to dictionary keys
    for i in range(len(lst[0])):
        header = lst[0][i] 
        dct[header] = [] # start with an empty list
        
        # let's populate the list I guess
        for row in lst[1:]:
            # skipped the first list since it was
            # just the headers
            # now pick up element i and add to dct
            dct[header].append(row[i])
    
    return dct

def get_filenames(dirname, ext = ".csv"):
    ''' given the name of a directory (string), return a list
    of paths to all the ****.ext files in that directory'''
    filenames = []
    files = os.listdir(dirname)
    for file in files:
        if file.endswith(ext):
            filenames.append(dirname + "/" + file)
    return filenames

def clean_numeric(s):
    ''' given a string with extra characters $ or , or %, remove them
    and return the value as a float'''
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.replace(",", "")
    return float(s)

def median(old_lst):
    '''Finds the median of the data in my list. This function will NOT 
    update the old list but instead create a copy'''
    lst = old_lst.copy()
    lst.sort()
    mid = len(lst) // 2
    return lst[mid]

def str_to_int(lst):
    '''Convert a list of str to list of int'''
    return [int(i) for i in lst]

def int_to_str(lst):
    '''Convert a list of ints to list of strs'''
    return [str(i) for i in lst]

def filter_by_line(filter_value, data_lst, other_lst):
    '''
    Filters other_lst data down by filter value, checking
    using data_lst
    
    Parameters
    -------
    filter_value: str 
    The value to look for
    
    data_lst: list of strings
    The column of data which has the filter_value that needs
    to be checked
    
    other_lst: list of obj
    The data to filter and return.

    Returns
    -------
    list of filtered data from other_list.

    '''
    return [other_lst[i] for i in range(len(data_lst)) if 
            filter_value.lower() in data_lst[i].lower()]

def moving_avg(num_lst, window=2):
    '''parameters: list of numbers, optionally an int (the window for 
    the moving average, default to 2) 
    returns: a new list, containing the moving average values of the original,
    according to the given window'''
    new_lst = [sum(num_lst[i:i + window]) / window for i in 
               range(len(num_lst) - window + 1)]
    return new_lst

def organize_files(my_dir):
    '''A function that, given the name of directory, creates subdirectories 
    based on file extensions, and moves the existing files accordingly.'''
    for file in os.listdir(my_dir):
        full_file_path = os.path.join(my_dir, file)
        # Check if the path is an actual file, not a directory
        if os.path.isfile(full_file_path):
            # Split the file into the filename and extension
            filename, ext = os.path.splitext(file)
            # Remove the leading dot from the extension
            ext = ext[1:]
            if ext:  # Only proceed if there's an extension
                # Create a path to the subdirectory based on the extension
                ext_dir = os.path.join(my_dir, ext)
                # Create the subdirectory if it doesn't exist
                if not os.path.exists(ext_dir):
                    os.mkdir(ext_dir)
                # Move the file to the subdirectory
                shutil.move(full_file_path, os.path.join(ext_dir, file))
                
def save_plot_to_subdir(filename):
    '''Save the current plot to the specified filename and organize it into a 
    subdirectory based on its extension.
    filename: string, the name of the file to save the plot as'''
    # Extract the extension from the filename
    file, ext = os.path.splitext(filename)
    # Remove the leading dot from the extension
    ext_dir = ext[1:]
    if not os.path.exists(ext_dir):
        os.makedirs(ext_dir)
    # Save the plot in the subdirectory
    plt.savefig(os.path.join(ext_dir, filename))
    
def normalize(some_lst):
    '''Normalize the list and returns a list of normalized values.'''
    normalized = []
    min_val = min(some_lst)
    max_val = max(some_lst)
    
    for item in some_lst:
        normalized_val = (item - min_val) / (max_val - min_val)
        normalized.append(normalized_val)
    return normalized

def mse(pred_vals, act_vals):
    '''Accepts two lists of numbers that are equal length. One list is a list
    of predicted values while the other is a list of actual values. Returns 
    the Mean Square Error: average of (predicted - actual)2 for every 
    predicted/actual pair'''
    result = 0
    for i in range(len(pred_vals)):
        result += (pred_vals[i] - act_vals[i]) ** 2
    return result / len(pred_vals)

def haversine(lat1, long1, lat2, long2, earth_radius = 6371000):
    """Calculate the Haversine distance between two points on the Earth.
    Parameters - lat1 (float): Latitude of the first point in degrees, 
    long1 (float): Longitude of the first point in degrees, 
    lat2 (float): Latitude of the second point in degrees, 
    long2 (float): Longitude of the second point in degrees, 
    earth_radius (float): Radius of the Earth in meters. Default is 6371000 m.
    Returns: float: Haversine distance between the two points in meters."""
    ɸ1, ɸ2 = math.radians(lat1), math.radians(lat2)
    λ1, λ2 = math.radians(long1), math.radians(long2)
    Δɸ = ɸ2 - ɸ1
    Δλ = λ2 - λ1
    a = a = math.sin(Δɸ / 2)**2 + math.cos(ɸ1) * math.cos(ɸ2) * math.sin(Δλ / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    hav_dist = c * earth_radius
    return hav_dist

def read_multiple_csv(files):
    '''Reads multiple CSV files into a list of DataFrames.'''
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)
    return dataframes

def shallow_cloning(lst):
    new_lst = []
    for item in lst:
        new_lst.append(item)

def main():
    
    # Testing out my functions
    lst_data = read_csv(FILENAME)
    dct_data = lst_to_dct(lst_data)
    # print(dct_data)
    
    lst_of_ints = [5, 3, 6, 1, 7, 5, 6 , 9, 8]
    
    print(f"Old list is {lst_of_ints}.")
    print(f"Median is {median(lst_of_ints)}")
    print(f"After calculating median, list is {lst_of_ints}")

if __name__ == "__main__":
    main()