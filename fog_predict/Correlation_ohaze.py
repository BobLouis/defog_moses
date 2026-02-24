import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -------- 設定 --------
dataset = "SOTS_inout"
# dataset = "OHaze"

base_dir = f"./dataset/{dataset}"
report_dir = os.path.join(base_dir, "report")

# 輸入檔案：直接吃 fog_estimation_tile.csv
fog_csv = os.path.join(report_dir, "fog_estimation_tile.csv")

# 輸出檔案
output_csv = os.path.join(report_dir, "fog_psi_comparison_tile.csv")
output_plot = os.path.join(report_dir, "fog_psi_scatter.png")


def load_data():
    """讀取 fog_estimation_tile.csv"""
    df = pd.read_csv(fog_csv)

    # 如果你有加 AVERAGE 那行就先丟掉
    if "Image" in df.columns:
        df = df[df["Image"] != "AVERAGE"].copy()
        df["Image"] = df["Image"].astype(str)

    # 確保數值欄位是 float
    numeric_cols = [
        "FogScore", "Psi", "FogScoreMeanTile", "FogScoreStdTile",
        "DrMean", "DevMean", "EdgeMean"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"成功讀取 {len(df)} 筆資料")
    return df


def find_relationship(fog_scores, psi_values):
    """找出 FogScore 與 Psi 的關係式 & correlation"""

    # 1. 線性回歸（順便給你 R / R² = correlation）
    slope, intercept, r_value, p_value, std_err = stats.linregress(fog_scores, psi_values)

    print("\n" + "=" * 60)
    print("線性迴歸分析：Psi = a × FogScore + b")
    print("=" * 60)
    print(f"斜率 (a):           {slope:.6f}")
    print(f"截距 (b):           {intercept:.6f}")
    print(f"相關係數 (R):        {r_value:.6f}")      # correlation
    print(f"決定係數 (R²):       {r_value**2:.6f}")
    print(f"P值:                {p_value:.6e}")
    print(f"標準誤差:            {std_err:.6f}")

    # 預測值
    psi_predicted = slope * fog_scores + intercept

    # MAE / RMSE / MAPE
    mae = np.mean(np.abs(psi_values - psi_predicted))
    rmse = np.sqrt(np.mean((psi_values - psi_predicted) ** 2))
    mape = np.mean(np.abs((psi_values - psi_predicted) / psi_values)) * 100

    print("\n" + "=" * 60)
    print("誤差分析")
    print("=" * 60)
    print(f"MAE (平均絕對誤差):           {mae:.6f}")
    print(f"RMSE (均方根誤差):            {rmse:.6f}")
    print(f"MAPE (平均絕對百分比誤差):    {mape:.2f}%")

    # 2. 額外順便看一下其他型態（你要就看，不要就忽略）
    print("\n" + "=" * 60)
    print("其他可能的關係式")
    print("=" * 60)

    # 二次多項式
    coeffs_2 = np.polyfit(fog_scores, psi_values, 2)
    psi_poly2 = np.polyval(coeffs_2, fog_scores)
    mae_poly2 = np.mean(np.abs(psi_values - psi_poly2))
    r2_poly2 = 1 - (np.sum((psi_values - psi_poly2) ** 2) /
                    np.sum((psi_values - np.mean(psi_values)) ** 2))
    print(
        f"二次多項式: Psi = {coeffs_2[0]:.6f}×Score² + {coeffs_2[1]:.6f}×Score + {coeffs_2[2]:.6f}"
    )
    print(f"  R² = {r2_poly2:.6f}, MAE = {mae_poly2:.6f}")

    # 指數關係
    try:
        # Psi = a × exp(b × Score)
        log_psi = np.log(psi_values)
        slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(fog_scores, log_psi)
        psi_exp = np.exp(intercept_exp + slope_exp * fog_scores)
        mae_exp = np.mean(np.abs(psi_values - psi_exp))
        r2_exp = 1 - (np.sum((psi_values - psi_exp) ** 2) /
                      np.sum((psi_values - np.mean(psi_values)) ** 2))
        print(f"指數關係: Psi = {np.exp(intercept_exp):.6f} × exp({slope_exp:.6f} × Score)")
        print(f"  R² = {r2_exp:.6f}, MAE = {mae_exp:.6f}")
    except Exception:
        print("指數關係: 無法計算")

    # 冪次關係
    try:
        # Psi = a × (Score+1)^b
        log_score = np.log(fog_scores + 1)  # +1 避免 log(0)
        log_psi = np.log(psi_values)
        slope_pow, intercept_pow, r_pow, _, _ = stats.linregress(log_score, log_psi)
        psi_pow = np.exp(intercept_pow) * ((fog_scores + 1) ** slope_pow)
        mae_pow = np.mean(np.abs(psi_values - psi_pow))
        r2_pow = 1 - (np.sum((psi_values - psi_pow) ** 2) /
                      np.sum((psi_values - np.mean(psi_values)) ** 2))
        print(
            f"冪次關係: Psi = {np.exp(intercept_pow):.6f} × (Score+1)^{slope_pow:.6f}"
        )
        print(f"  R² = {r2_pow:.6f}, MAE = {mae_pow:.6f}")
    except Exception:
        print("冪次關係: 無法計算")

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_value ** 2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "psi_predicted": psi_predicted,
    }


def plot_relationship(df, results):
    """繪製 FogScore vs Psi 的散點圖 + 迴歸線"""

    plt.figure(figsize=(12, 8))

    # 散點
    plt.scatter(df["FogScore"], df["Psi"], alpha=0.5, s=30, label="實際數據")

    # 線性迴歸線
    fog_range = np.linspace(df["FogScore"].min(), df["FogScore"].max(), 100)
    psi_fit = results["slope"] * fog_range + results["intercept"]
    plt.plot(
        fog_range,
        psi_fit,
        "r-",
        linewidth=2,
        label=f'線性迴歸: Psi = {results["slope"]:.4f} × Score + {results["intercept"]:.4f}',
    )

    plt.xlabel("FogScore", fontsize=12)
    plt.ylabel("Psi", fontsize=12)
    plt.title(
        f'FogScore vs Psi 關係圖\nR² = {results["r_squared"]:.4f}, MAE = {results["mae"]:.4f}',
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    textstr = (
        f"樣本數: {len(df)}\n"
        f"R  = {results['r_value']:.4f}\n"
        f"R² = {results['r_squared']:.4f}\n"
        f"MAE = {results['mae']:.4f}\n"
        f"RMSE = {results['rmse']:.4f}"
    )
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"\n✅ 散點圖已儲存：{output_plot}")
    # 若要看圖可打開：
    # plt.show()


def save_comparison_results(df, results):
    """儲存 Psi 與迴歸預測值的比較結果"""

    df = df.copy()
    df["PsiPredicted"] = results["psi_predicted"]
    df["Error"] = df["Psi"] - df["PsiPredicted"]
    df["AbsError"] = np.abs(df["Error"])
    df["PercentError"] = (df["AbsError"] / df["Psi"]) * 100

    df_sorted = df.sort_values("AbsError", ascending=False)
    df_sorted.to_csv(output_csv, index=False)
    print(f"✅ 比較結果已儲存：{output_csv}")

    print("\n" + "=" * 60)
    print("預測誤差最大的前 10 張圖片")
    print("=" * 60)
    print(
        df_sorted[
            ["Image", "FogScore", "Psi", "PsiPredicted", "AbsError", "PercentError"]
        ]
        .head(10)
        .to_string(index=False)
    )


def main():
    print("開始分析 FogScore 與 Psi 的關係（correlation）...")
    print("=" * 60)

    # 1. 讀資料
    df = load_data()

    # 2. 基本統計
    print("\n" + "=" * 60)
    print("基本統計")
    print("=" * 60)
    print(f"FogScore 範圍: {df['FogScore'].min():.4f} - {df['FogScore'].max():.4f}")
    print(f"FogScore 平均: {df['FogScore'].mean():.4f}")
    print(f"FogScore 標準差: {df['FogScore'].std():.4f}")

    print(
        f"\nPsi 範圍: {df['Psi'].min():.4f} - {df['Psi'].max():.4f}"
    )
    print(f"Psi 平均: {df['Psi'].mean():.4f}")
    print(f"Psi 標準差: {df['Psi'].std():.4f}")

    # 3. 關係式 & correlation
    results = find_relationship(df["FogScore"].values, df["Psi"].values)

    # 4. 繪圖
    plot_relationship(df, results)

    # 5. 儲存比較結果
    save_comparison_results(df, results)

    print("\n" + "=" * 60)
    print("✅ 分析完成")
    print("=" * 60)
    print("\n建議使用的關係式（線性）：")
    print(f"Psi ≈ {results['slope']:.6f} × FogScore + {results['intercept']:.6f}")
    print(f"Correlation (R) = {results['r_value']:.6f}")


if __name__ == "__main__":
    main()
