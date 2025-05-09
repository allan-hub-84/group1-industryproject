# Data Manipulation and Analysis
import numpy as np  
import pandas as pd 

# Data Visualization
import matplotlib.pyplot as plt 
import seaborn as sns  
import plotly.express as px

# Common variable declaration

feature_descriptions = {
    "State": "State FIPS Code",
    "Sex": "Sex of Respondent",
    "GeneralHealth": "Would you say that in general your health is:",
    "PhysicalActivities": "Did you participate in any physical activities or exercises in the past month?",
    "SleepHours": "On average, how many hours of sleep do you get in a 24-hour period?",
    "HadHeartAttack": "Ever told you had a heart attack (myocardial infarction)?",
    "HadAngina": "Ever told you had angina or coronary heart disease?",
    "HadStroke": "Ever told you had a stroke?",
    "HadAsthma": "Ever told you had asthma?",
    "HadSkinCancer": "Ever told you had skin cancer (not melanoma)?",
    "HadCOPD": "Ever told you had COPD, emphysema, or chronic bronchitis?",
    "HadDepressiveDisorder": "Ever told you had a depressive disorder (e.g., major depression, dysthymia)?",
    "HadKidneyDisease": "Ever told you had kidney disease (excluding kidney stones, bladder infection, incontinence)?",
    "HadArthritis": "Ever told you had arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia?",
    "HadDiabetes": "Ever told you had diabetes?",
    "SmokerStatus": "Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker",
    "ECigaretteUsage": "E-cigarette usage: Never used, currently using daily/some days, or used in the past but not currently",
    "ChestScan": "Have you ever had a CT or CAT scan of your chest area?",
    "RaceEthnicityCategory": "Five-level race/ethnicity category",
    "AgeCategory": "Fourteen-level age category",
    "HeightInMeters": "Reported height in meters",
    "WeightInKilograms": "Reported weight in kilograms",
    "BMI": "Body Mass Index (BMI)",
    "AlcoholDrinkers": "Adults who reported having had at least one drink of alcohol in the past 30 days",
}

# Importing the data
def import_csv(file_path):
    """
    Import CSV data into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    data = pd.read_csv(file_path,index_col=None)
    return data

def generate_data_dictionary(df, column_descriptions):
    """
    Generate a data dictionary.

    Parameters:
        df (pd.DataFrame): The DataFrame.
        column_descriptions (dict):  column names with their descriptions.

    Returns:
        pd.DataFrame: Return the data-dictionary for all the columns in the Data-frame.
    """
    
    heart_attack_dict = []

    # Loop through all columns in the DataFrame
    for column in df.columns:
        # Append column details
        heart_attack_dict.append({
            'Column Name': column,
            'Data Type': df[column].dtype,
            'Description': column_descriptions.get(column, 'Description not available'),
            'Unique Values': df[column].nunique(),
            'Missing Values': df[column].isnull().sum(),
            'Distinct Values': df[column].dropna().unique().tolist(),
            'Value Range': (df[column].min(), df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else 'N/A',
        })

    # Convert the list to a DataFrame
    heart_attack_dict_df = pd.DataFrame(heart_attack_dict)
    return heart_attack_dict_df

def define_df_settings():
    """
    Define the necessary data frame settinfs.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 80)

def display_plot(data, column, plot_type='hist', bins=30, xlabel=None, ylabel=None, title=None, title_orientation='None', sort=False):
    """
    Plots various EDA charts for a given column.
    
    Parameters:
    - data: DataFrame containing the data.
    - column: Column name to visualize.
    - plot_type: Type of plot ('bar', 'hist', 'box', 'violin', 'scatter', 'pair', 'correlation'). Default is 'hist'.
    - bins: Number of bins for histograms. Default is 30.
    - xlabel: Custom label for the x-axis.
    - ylabel: Custom label for the y-axis.
    - title: Custom title for the plot.
    - orientation: Histogram orientation ('vertical' or 'horizontal'). Default is 'vertical'.
    - sort: Whether to sort categorical data in ascending order (only applicable for bar plots). Default is False.
    """
    plt.figure(figsize=(10, 5))
    
    if plot_type == 'bar':
        value_counts = data[column].value_counts()
        if sort:
            value_counts = value_counts.sort_index()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        if title_orientation == 'vertical':
            plt.xticks(rotation=90)

    elif plot_type == 'hist':
            plt.hist(data[column], bins=bins, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel(xlabel if xlabel else column,)
            plt.ylabel(ylabel if ylabel else 'Frequency')
            if title_orientation == 'vertical':
                plt.xticks(rotation=90)
    elif plot_type == 'box':
        if orientation == 'horizontal':
            sns.boxplot(x=data[column])
        else:
            sns.boxplot(y=data[column])
    elif plot_type == 'violin':
        if orientation == 'horizontal':
            sns.violinplot(x=data[column])
        else:
            sns.violinplot(y=data[column])
    elif plot_type == 'scatter':
        sns.scatterplot(x=data.index, y=data[column])
    elif plot_type == 'pair':
        sns.pairplot(data)
        plt.show()
        return
    elif plot_type == 'correlation':
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
        return
    else:
        raise ValueError("Invalid plot_type. Choose from 'bar', 'hist', 'box', 'violin', 'scatter', 'pair', or 'correlation'.")
    
    plt.title(title if title else f'{plot_type.capitalize()} plot for {column}')
    plt.show()
