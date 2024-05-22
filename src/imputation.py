from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import numpy as np


def imputation_knn(df, test, column, n_neighbors=3, random_state=123):
    """
    Imputar los valores NaN en una columna específica utilizando K-Nearest Neighbors.

    :param df: DataFrame de pandas.
    :param column: Columna para la cual se imputarán los valores NaN.
    :param n_neighbors: Número de vecinos a considerar en KNN.
    :return: DataFrame con la columna imputada.
    """
    df_column = df[column]
    test_column = test[column]

    # Imputación con KNN para la columna 'sex'
    # Convertir 'sex' a valores numéricos temporales para KNN
    label_encoder = LabelEncoder()
    col_encoded = label_encoder.fit_transform(df_column.astype(str))
    test_col_encoded = label_encoder.transform(test_column.astype(str))
    
    col_encoded = col_encoded.reshape(-1, 1)
    test_col_encoded = test_col_encoded.reshape(-1, 1)

    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    col_imputed = knn_imputer.fit_transform(col_encoded)
    test_col_imputed = knn_imputer.transform(test_col_encoded)

    # Convertir de nuevo los valores imputados a etiquetas originales
    col_imputed = np.round(col_imputed).astype(int)
    test_col_imputed = np.round(test_col_imputed).astype(int)

    col_imputed = label_encoder.inverse_transform(col_imputed.ravel())
    test_col_imputed = label_encoder.inverse_transform(test_col_imputed.ravel())

    # Asignar los valores imputados a la columna original
    df[column] = col_imputed
    test[column] = test_col_imputed

    # return df


def imputation_random_forest(df, test, column, random_state=123):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    encoded_columns = {}
    for col in cat_cols:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col].astype(str))
        test[col] = label_encoder.transform(test[col].astype(str))

        encoded_columns[col] = label_encoder

    train_df = df[df[column].notna()]
    test_df = test[test[column].notna()]

    predict_df = df[df[column].isna()]
    test_predict_df = test[test[column].isna()]

    if predict_df.empty:
        return df, test

    X_train = train_df.drop(columns=[column])

    y_train = train_df[column]

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf_regressor.fit(X_train, y_train)

    X_predict = predict_df.drop(columns=[column])
    X_test_predict = test_predict_df.drop(columns=[column])

    y_predict = rf_regressor.predict(X_predict)
    y_test_predict = rf_regressor.predict(X_test_predict)

    df.loc[df[column].isna(), column] = y_predict
    test.loc[test[column].isna(), column] = y_test_predict

    for col, encoder in encoded_columns.items():
        df[col] = encoder.inverse_transform(df[col].astype(int))
        test[col] = encoder.inverse_transform(test[col].astype(int))

    # return df
