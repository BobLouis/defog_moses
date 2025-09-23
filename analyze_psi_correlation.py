#!/usr/bin/env python3
"""
Analysis script to find mathematical relationship between Ar, Ag, Ab and BestPsi
for VLSI implementation (non-AI approach)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def load_and_analyze_data(csv_path):
    """Load CSV data and perform initial analysis"""
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Check for missing values
    print(f"\nMissing values:")
    print(df[['BestPsi', 'Ar', 'Ag', 'Ab']].isnull().sum())

    # Remove rows with NaN values
    df_clean = df.dropna(subset=['BestPsi', 'Ar', 'Ag', 'Ab'])
    print(f"Dataset shape after removing NaN: {df_clean.shape}")

    print("\nBasic statistics:")
    print(df_clean[['BestPsi', 'Ar', 'Ag', 'Ab']].describe())

    return df_clean

def analyze_correlations(df):
    """Analyze correlations between variables"""
    features = ['Ar', 'Ag', 'Ab']
    target = 'BestPsi'

    # Basic correlation matrix
    corr_matrix = df[features + [target]].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # Individual correlations with BestPsi
    print(f"\nIndividual correlations with {target}:")
    for feature in features:
        corr = df[feature].corr(df[target])
        print(f"{feature}: {corr:.4f}")

    return corr_matrix

def analyze_rgb_patterns(df):
    """Analyze RGB channel patterns and relationships"""
    # Calculate derived features
    df['RGB_Sum'] = df['Ar'] + df['Ag'] + df['Ab']
    df['RGB_Mean'] = df['RGB_Sum'] / 3
    df['RGB_Max'] = df[['Ar', 'Ag', 'Ab']].max(axis=1)
    df['RGB_Min'] = df[['Ar', 'Ag', 'Ab']].min(axis=1)
    df['RGB_Range'] = df['RGB_Max'] - df['RGB_Min']
    df['RG_Ratio'] = df['Ar'] / (df['Ag'] + 1e-6)
    df['RB_Ratio'] = df['Ar'] / (df['Ab'] + 1e-6)
    df['GB_Ratio'] = df['Ag'] / (df['Ab'] + 1e-6)

    # Analyze correlations with derived features
    derived_features = ['RGB_Sum', 'RGB_Mean', 'RGB_Max', 'RGB_Min', 'RGB_Range',
                       'RG_Ratio', 'RB_Ratio', 'GB_Ratio']

    print("\nCorrelations with derived features:")
    for feature in derived_features:
        corr = df[feature].corr(df['BestPsi'])
        print(f"{feature}: {corr:.4f}")

    return df, derived_features

def find_mathematical_relationships(df):
    """Find mathematical relationships suitable for VLSI implementation"""

    # Simple linear relationship analysis
    print("\n=== LINEAR RELATIONSHIP ANALYSIS ===")

    # Test different combinations
    X_simple = df[['Ar', 'Ag', 'Ab']].values
    y = df['BestPsi'].values

    # Simple linear regression
    lr = LinearRegression()
    lr.fit(X_simple, y)
    y_pred_lr = lr.predict(X_simple)
    r2_lr = r2_score(y, y_pred_lr)

    print(f"Linear Regression R²: {r2_lr:.4f}")
    print(f"Coefficients: Ar={lr.coef_[0]:.6f}, Ag={lr.coef_[1]:.6f}, Ab={lr.coef_[2]:.6f}")
    print(f"Intercept: {lr.intercept_:.6f}")
    print(f"Formula: BestPsi = {lr.intercept_:.6f} + {lr.coef_[0]:.6f}*Ar + {lr.coef_[1]:.6f}*Ag + {lr.coef_[2]:.6f}*Ab")

    # Test polynomial features (degree 2)
    print("\n=== POLYNOMIAL RELATIONSHIP ANALYSIS ===")
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_simple)

    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, y)
    y_pred_poly = lr_poly.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)

    print(f"Polynomial (degree 2) R²: {r2_poly:.4f}")
    feature_names = poly.get_feature_names_out(['Ar', 'Ag', 'Ab'])
    print("Polynomial features and coefficients:")
    for i, (name, coef) in enumerate(zip(feature_names, lr_poly.coef_)):
        print(f"  {name}: {coef:.6f}")
    print(f"Intercept: {lr_poly.intercept_:.6f}")

    # Analyze simple ratios and products
    print("\n=== RATIO-BASED ANALYSIS ===")

    # Test simple mathematical operations
    test_features = {
        'Ar_div_255': df['Ar'] / 255.0,
        'Ag_div_255': df['Ag'] / 255.0,
        'Ab_div_255': df['Ab'] / 255.0,
        'Min_RGB_div_255': df[['Ar', 'Ag', 'Ab']].min(axis=1) / 255.0,
        'Max_RGB_div_255': df[['Ar', 'Ag', 'Ab']].max(axis=1) / 255.0,
        'Mean_RGB_div_255': df[['Ar', 'Ag', 'Ab']].mean(axis=1) / 255.0,
    }

    for name, feature in test_features.items():
        corr = feature.corr(df['BestPsi'])
        print(f"{name}: {corr:.4f}")

    return lr, lr_poly, poly, test_features

def generate_vlsi_formula(df, lr, lr_poly, poly):
    """Generate simplified formulas suitable for VLSI implementation"""

    print("\n=== VLSI-SUITABLE FORMULAS ===")

    # Method 1: Simple linear approximation (easy to implement in hardware)
    print("Method 1: Simple Linear Formula")
    print(f"BestPsi ≈ {lr.intercept_:.3f} + {lr.coef_[0]:.6f}*Ar + {lr.coef_[1]:.6f}*Ag + {lr.coef_[2]:.6f}*Ab")

    # Method 2: Normalized version (easier for fixed-point arithmetic)
    ar_norm = lr.coef_[0] / 255.0
    ag_norm = lr.coef_[1] / 255.0
    ab_norm = lr.coef_[2] / 255.0

    print("\nMethod 2: Normalized Formula (for 8-bit RGB values)")
    print(f"BestPsi ≈ {lr.intercept_:.3f} + {ar_norm:.6f}*Ar + {ag_norm:.6f}*Ag + {ab_norm:.6f}*Ab")

    # Method 3: Integer approximation for hardware
    # Scale coefficients to avoid floating point
    scale = 1000000  # Use 6 decimal places precision
    ar_int = int(lr.coef_[0] * scale)
    ag_int = int(lr.coef_[1] * scale)
    ab_int = int(lr.coef_[2] * scale)
    intercept_int = int(lr.intercept_ * scale)

    print(f"\nMethod 3: Integer Formula (scaled by {scale})")
    print(f"BestPsi_scaled = {intercept_int} + {ar_int}*Ar + {ag_int}*Ag + {ab_int}*Ab")
    print(f"BestPsi = BestPsi_scaled / {scale}")

    # Method 4: Lookup table approach
    print("\nMethod 4: Simplified Range-based Approach")

    # Analyze typical ranges and create simple rules
    rgb_ranges = []
    psi_means = []

    # Divide RGB space into ranges
    for ar_range in [(0, 200), (200, 240), (240, 255)]:
        for ag_range in [(0, 200), (200, 240), (240, 255)]:
            for ab_range in [(0, 200), (200, 240), (240, 255)]:
                mask = ((df['Ar'] >= ar_range[0]) & (df['Ar'] < ar_range[1]) &
                       (df['Ag'] >= ag_range[0]) & (df['Ag'] < ag_range[1]) &
                       (df['Ab'] >= ab_range[0]) & (df['Ab'] < ab_range[1]))

                if mask.sum() > 0:
                    mean_psi = df.loc[mask, 'BestPsi'].mean()
                    rgb_ranges.append((ar_range, ag_range, ab_range))
                    psi_means.append(mean_psi)

                    print(f"RGB range Ar:{ar_range}, Ag:{ag_range}, Ab:{ab_range} → BestPsi ≈ {mean_psi:.2f} (n={mask.sum()})")

    return ar_int, ag_int, ab_int, intercept_int, scale

def validate_formulas(df, lr):
    """Validate the accuracy of proposed formulas"""

    print("\n=== FORMULA VALIDATION ===")

    # Test on the dataset
    X = df[['Ar', 'Ag', 'Ab']].values
    y_true = df['BestPsi'].values
    y_pred = lr.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    # Analyze error distribution
    errors = y_true - y_pred
    print(f"Error statistics:")
    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Std error: {np.std(errors):.4f}")
    print(f"  Max error: {np.max(np.abs(errors)):.4f}")

    return y_pred, errors

def main():
    csv_path = '/Users/hongweichen/Documents/實驗室/論文/defog/昱恩交接/img_soft/dataset/SOTS_inout/report/score_optimize_psi_grid.csv'

    # Load and analyze data
    df = load_and_analyze_data(csv_path)

    # Analyze correlations
    corr_matrix = analyze_correlations(df)

    # Analyze RGB patterns
    df, derived_features = analyze_rgb_patterns(df)

    # Find mathematical relationships
    lr, lr_poly, poly, test_features = find_mathematical_relationships(df)

    # Generate VLSI-suitable formulas
    ar_int, ag_int, ab_int, intercept_int, scale = generate_vlsi_formula(df, lr, lr_poly, poly)

    # Validate formulas
    y_pred, errors = validate_formulas(df, lr)

    print("\n=== SUMMARY FOR VLSI IMPLEMENTATION ===")
    print("Recommended approach for hardware implementation:")
    print("1. Use linear regression formula (highest interpretability)")
    print("2. Implement with integer arithmetic to avoid floating point")
    print("3. Use fixed-point representation with appropriate scaling")
    print(f"\nFinal integer formula:")
    print(f"BestPsi_scaled = {intercept_int} + {ar_int}*Ar + {ag_int}*Ag + {ab_int}*Ab")
    print(f"BestPsi = BestPsi_scaled / {scale}")
    print(f"Expected accuracy: R² = {r2_score(df['BestPsi'], y_pred):.4f}")

if __name__ == "__main__":
    main()