import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Union
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


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
    _, axes = plt.subplots(
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


def plot_numeric_relationship(data, numeric_var, categorical_var, subplot_position=(1,1,1), abbreviate_names = True):
    """
    Plots the relationship between a numeric variable and a categorical variable using a boxplot within a subplot.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        numeric_var (str): The name of the numeric column.
        categorical_var (str): The name of the categorical column.
        subplot_position (tuple): The position of the subplot (nrows, ncols, index).
        figure_size (tuple): The overall figure size.
    """
    if isinstance(abbreviate_names, bool): abbvr_length = 10
    else: abbvr_length = abbreviate_names
    # Ensure the variable names exist in the DataFrame
    if numeric_var not in data.columns or categorical_var not in data.columns:
        raise ValueError("The specified columns do not exist in the DataFrame")

    # Set the current subplot position
    plt.subplot(*subplot_position)
    
    # Create a boxplot
    sns.boxplot(x=categorical_var, y=numeric_var, data=data, hue=categorical_var)
    plt.title(f'Relationship between {categorical_var} and {numeric_var}')
    plt.xlabel(categorical_var)
    plt.ylabel(numeric_var)
    if abbvr_length != False:
        xticklabels = [label.get_text()[:abbvr_length] for label in plt.gca().get_xticklabels()]
        plt.gca().set_xticklabels(xticklabels)
    plt.xticks(rotation=45, ha='right')


def plot_categorical_relationship(data, target, categorical_var, subplot_position=(1,1,1), abbreviate_names: Union[bool, int] = True):
    """
    Plots the relationship between two categorical variables using a count plot within a subplot.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        cat_var1 (str): The name of the first categorical column.
        cat_var2 (str): The name of the second categorical column.
        subplot_position (tuple): The position of the subplot (nrows, ncols, index).
        figure_size (tuple): The overall figure size.
        abbreviate_names (Union[bool, int]): If True, abbreviate category names to 10 characters. If int, abbreviate to that many characters.
    """
    if isinstance(abbreviate_names, bool): abbvr_length = 10 if abbreviate_names else None
    else: abbvr_length = abbreviate_names

    # Ensure the variable names exist in the DataFrame
    if target not in data.columns or categorical_var not in data.columns:
        raise ValueError("The specified columns do not exist in the DataFrame")

    # Set the current subplot position
    plt.subplot(*subplot_position)
    
    # Create a count plot
    sns.countplot(x=target, hue=categorical_var, data=data)
    plt.title(f'Relationship between {target} and {categorical_var}')
    plt.xlabel(target)
    plt.ylabel('Count')

    # Abbreviate x-tick labels if required
    if abbvr_length != False:
        xticklabels = [label.get_text()[:abbvr_length] for label in plt.gca().get_xticklabels()]
        plt.gca().set_xticklabels(xticklabels)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


def bivariate_w_target_variable(data, numeric_vars: bool=True, target='y', ncols=3, fig_width=15, fig_height_per_row=5):
    """
    Plots multiple relationships in a grid layout.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        numeric_vars (bool): If we want to plot numerical variables or categorical ones.
    """
    if numeric_vars:
        cols = data.select_dtypes(include=['number']).columns
    
    else:
        cols = data.select_dtypes(include=['object']).columns


    num_plots = len(cols)
    nrows = (num_plots + 2) // 3  # Calculate rows needed for 3 plots per row
    ncols = 3

    plt.figure(figsize=(fig_width, fig_height_per_row * nrows))  # Adjust the figure size based on the number of rows

    
    for index, col in enumerate(cols, start=1):
        if numeric_vars: plot_numeric_relationship(data=data, numeric_var=col, categorical_var=target, subplot_position=(nrows, ncols, index))
        else: plot_categorical_relationship(data=data, categorical_var=col, target=target, subplot_position=(nrows, ncols, index))

    plt.tight_layout()
    plt.show()


def imputation_knn(df, column, n_neighbors=3):
    """
    Imputar los valores NaN en una columna específica utilizando K-Nearest Neighbors.

    :param df: DataFrame de pandas.
    :param column: Columna para la cual se imputarán los valores NaN.
    :param n_neighbors: Número de vecinos a considerar en KNN.
    :return: DataFrame con la columna imputada.
    """
    df_column = df[column]

    # Imputación con KNN para la columna 'sex'
    # Convertir 'sex' a valores numéricos temporales para KNN
    label_encoder = LabelEncoder()
    col_encoded = label_encoder.fit_transform(df_column.astype(str))
    col_encoded = col_encoded.reshape(-1, 1)

    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    col_imputed = knn_imputer.fit_transform(col_encoded)

    # Convertir de nuevo los valores imputados a etiquetas originales
    col_imputed = np.round(col_imputed).astype(int)
    col_imputed = label_encoder.inverse_transform(col_imputed.ravel())

    # Asignar los valores imputados a la columna original
    df[column] = col_imputed
    return df


def imputation_random_forest(df, column):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    encoded_columns = {}
    for col in cat_cols:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col].astype(str))
        encoded_columns[col] = label_encoder

    train_df = df[df[column].notna()]
    predict_df = df[df[column].isna()]

    if predict_df.empty:
        return df

    X_train = train_df.drop(columns=[column])
    y_train = train_df[column]

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    X_predict = predict_df.drop(columns=[column])
    y_predict = rf_regressor.predict(X_predict)

    df.loc[df[column].isna(), column] = y_predict

    for col, encoder in encoded_columns.items():
        df[col] = encoder.inverse_transform(df[col].astype(int))

    return df

