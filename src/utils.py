import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Union

def plot_missing_values_greater_than_0(df):
    # Calculate percentage of missing values by column
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_percentage = missing_percentage[missing_percentage > 0]

    # Plot the percentage of missing values
    plt.figure(figsize=(10, 6))
    bars = missing_percentage.plot(kind='bar', color='skyblue')
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage of Missing Values')
    plt.xticks(rotation=45, ha='right')

    # Add text labels to the bars
    for bar in bars.patches:
        plt.annotate(format(bar.get_height(), '.1f') + '%', 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), 
                    ha='center', 
                    va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()


def create_numeric_eda(df, fig_height, graphs_per_row=3):
    numeric_date_cols = df.select_dtypes(include=['number', 'datetime64']).columns

    # Create histograms for numeric/date variables
    fig, axes = plt.subplots(
        nrows=(len(numeric_date_cols) + graphs_per_row - 1) // graphs_per_row, 
        ncols=graphs_per_row, 
        figsize=(18, fig_height * ((len(numeric_date_cols) + graphs_per_row - 1) // graphs_per_row))
        )
    axes = axes.flatten()  # Flatten to 1D array for easier iteration
    for i, col in enumerate(numeric_date_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)  # Hide unused subplots
    plt.tight_layout()
    plt.show()

def create_factor_eda(df, fig_height, graphs_per_row=3, abbreviate_names: Union[bool, int] = True):
    if isinstance(abbreviate_names, bool): abbvr_length = 10
    else: abbvr_length = abbreviate_names

    object_cols = df.select_dtypes(include='object').columns

    # Create stacked countplots for object variables
    fig, axes = plt.subplots(
        nrows=(len(object_cols) + graphs_per_row - 1) // graphs_per_row, 
        ncols=graphs_per_row, 
        figsize=(18, fig_height * ((len(object_cols) + graphs_per_row - 1) // graphs_per_row))
        )
    axes = axes.flatten()
    for i, col in enumerate(object_cols):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'Countplot of {col} by Species')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        
        # Abbreviate xticks if requested
        if abbvr_length != False:
            xticklabels = [label.get_text()[:abbvr_length] for label in axes[i].get_xticklabels()]
            axes[i].set_xticklabels(xticklabels, rotation=45)
        else:
            axes[i].tick_params(axis='x', rotation=45)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)  # Hide unused subplots
    plt.tight_layout()
    plt.show()


