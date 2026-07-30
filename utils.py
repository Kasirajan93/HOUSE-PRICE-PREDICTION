# ==========================================
# Utility Functions
# ==========================================

import pandas as pd
import numpy as np


# ==========================================
# Create Input DataFrame
# ==========================================

def create_input_dataframe(user_inputs):
    """
    Convert user inputs dictionary into a pandas DataFrame.
    """

    df = pd.DataFrame([user_inputs])

    return df


# ==========================================
# Preprocess Input Data
# ==========================================

def preprocess_input(df, feature_columns):
    """
    Preprocess user input exactly like the training pipeline.
    """

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df["Total Bathrooms"] = (
        df["Full Bath"] +
        0.5 * df["Half Bath"]
    )

    df["Total Living Area"] = (
        df["Gr Liv Area"] +
        df["Total Bsmt SF"]
    )

    # -----------------------------
    # One-Hot Encoding
    # -----------------------------

    df = pd.get_dummies(df)

    # -----------------------------
    # Match Training Features
    # -----------------------------

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df    