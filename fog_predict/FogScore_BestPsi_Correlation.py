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

# 輸入檔案
fog_csv = os.path.join(report_dir, "fog_estimation_simple.csv")
psi_csv = os.path.join(report_dir, "score_optimize_psi_grid.csv")

# 輸出檔案
output_csv = os.path.join(report_dir, "fog_psi_comparison.csv")
output_plot = os.path.join(report_dir, "fog_psi_scatter.png")


def load_and_merge_data():
    """讀取並合併兩個 CSV 檔案"""
    
    # 讀取霧氣評分
    df_fog = pd.read_csv(fog_csv)
    df_fog = df_fog[df_fog['Image'] != 'AVERAGE'].copy()
    df_fog['Image'] = df_fog['Image'].astype(str)
    
    # 讀取 BestPsi
    df_psi = pd.read_csv(psi_csv)
    df_psi = df_psi[df_psi['Image'] != 'AVERAGE'].copy()
    df_psi['Image'] = df_psi['Image'].astype(str)
    
    # 合併資料（根據 Image）
    df_merged = pd.merge(
        df_fog[['Image', 'FogScore', 'DynamicRange', 'AvgLocalDiff', 'AvgDeviation']],
        df_psi[['Image', 'BestPsi', 'PSNR']],
        on='Image',
        how='inner'
    )
    
    print(f"成功合併 {len(df_merged)} 筆資料")
    return df_merged


def find_relationship(fog_scores, psi_values):
    """找出 FogScore 與 BestPsi 的關係式"""
    
    # 1. 線性迴歸
    slope, intercept, r_value, p_value, std_err = stats.linregress(fog_scores, psi_values)
    
    print("\n" + "="*60)
    print("線性迴歸分析：BestPsi = a × FogScore + b")
    print("="*60)
    print(f"斜率 (a):           {slope:.6f}")
    print(f"截距 (b):           {intercept:.6f}")
    print(f"相關係數 (R):        {r_value:.6f}")
    print(f"決定係數 (R²):       {r_value**2:.6f}")
    print(f"P值:                {p_value:.6e}")
    print(f"標準誤差:            {std_err:.6f}")
    
    # 預測值
    psi_predicted = slope * fog_scores + intercept
    
    # 計算 MAE (Mean Absolute Error)
    mae = np.mean(np.abs(psi_values - psi_predicted))
    
    # 計算 RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((psi_values - psi_predicted)**2))
    
    # 計算 MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((psi_values - psi_predicted) / psi_values)) * 100
    
    print("\n" + "="*60)
    print("誤差分析")
    print("="*60)
    print(f"MAE (平均絕對誤差):   {mae:.6f}")
    print(f"RMSE (均方根誤差):    {rmse:.6f}")
    print(f"MAPE (平均絕對百分比誤差): {mape:.2f}%")
    
    # 2. 嘗試其他關係式
    print("\n" + "="*60)
    print("其他可能的關係式")
    print("="*60)
    
    # 二次多項式
    coeffs_2 = np.polyfit(fog_scores, psi_values, 2)
    psi_poly2 = np.polyval(coeffs_2, fog_scores)
    mae_poly2 = np.mean(np.abs(psi_values - psi_poly2))
    r2_poly2 = 1 - (np.sum((psi_values - psi_poly2)**2) / np.sum((psi_values - np.mean(psi_values))**2))
    print(f"二次多項式: Psi = {coeffs_2[0]:.6f}×Score² + {coeffs_2[1]:.6f}×Score + {coeffs_2[2]:.6f}")
    print(f"  R² = {r2_poly2:.6f}, MAE = {mae_poly2:.6f}")
    
    # 指數關係
    try:
        # Psi = a × exp(b × Score)
        log_psi = np.log(psi_values)
        slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(fog_scores, log_psi)
        psi_exp = np.exp(intercept_exp + slope_exp * fog_scores)
        mae_exp = np.mean(np.abs(psi_values - psi_exp))
        r2_exp = 1 - (np.sum((psi_values - psi_exp)**2) / np.sum((psi_values - np.mean(psi_values))**2))
        print(f"指數關係: Psi = {np.exp(intercept_exp):.6f} × exp({slope_exp:.6f} × Score)")
        print(f"  R² = {r2_exp:.6f}, MAE = {mae_exp:.6f}")
    except:
        print("指數關係: 無法計算")
    
    # 冪次關係
    try:
        # Psi = a × Score^b
        log_score = np.log(fog_scores + 1)  # +1 避免 log(0)
        log_psi = np.log(psi_values)
        slope_pow, intercept_pow, r_pow, _, _ = stats.linregress(log_score, log_psi)
        psi_pow = np.exp(intercept_pow) * ((fog_scores + 1) ** slope_pow)
        mae_pow = np.mean(np.abs(psi_values - psi_pow))
        r2_pow = 1 - (np.sum((psi_values - psi_pow)**2) / np.sum((psi_values - np.mean(psi_values))**2))
        print(f"冪次關係: Psi = {np.exp(intercept_pow):.6f} × (Score+1)^{slope_pow:.6f}")
        print(f"  R² = {r2_pow:.6f}, MAE = {mae_pow:.6f}")
    except:
        print("冪次關係: 無法計算")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'psi_predicted': psi_predicted
    }


def plot_relationship(df, results):
    """繪製散點圖和迴歸線"""
    
    plt.figure(figsize=(12, 8))
    
    # 散點圖
    plt.scatter(df['FogScore'], df['BestPsi'], alpha=0.5, s=30, label='實際數據')
    
    # 迴歸線
    fog_range = np.linspace(df['FogScore'].min(), df['FogScore'].max(), 100)
    psi_fit = results['slope'] * fog_range + results['intercept']
    plt.plot(fog_range, psi_fit, 'r-', linewidth=2, 
             label=f'線性迴歸: Psi = {results["slope"]:.4f} × Score + {results["intercept"]:.4f}')
    
    # 標註
    plt.xlabel('FogScore', fontsize=12)
    plt.ylabel('BestPsi', fontsize=12)
    plt.title(f'FogScore vs BestPsi 關係圖\nR² = {results["r_squared"]:.4f}, MAE = {results["mae"]:.4f}', 
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 顯示統計資訊
    textstr = f'樣本數: {len(df)}\nR² = {results["r_squared"]:.4f}\nMAE = {results["mae"]:.4f}\nRMSE = {results["rmse"]:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n✅ 散點圖已儲存：{output_plot}")
    
    # 可選：顯示圖表（如果在互動環境中）
    # plt.show()


def save_comparison_results(df, results):
    """儲存比較結果"""
    
    # 新增預測值和誤差欄位
    df['PsiPredicted'] = results['psi_predicted']
    df['Error'] = df['BestPsi'] - df['PsiPredicted']
    df['AbsError'] = np.abs(df['Error'])
    df['PercentError'] = (df['AbsError'] / df['BestPsi']) * 100
    
    # 排序（依誤差大小）
    df_sorted = df.sort_values('AbsError', ascending=False)
    
    # 儲存
    df_sorted.to_csv(output_csv, index=False)
    print(f"✅ 比較結果已儲存：{output_csv}")
    
    # 顯示誤差最大的前 10 筆
    print("\n" + "="*60)
    print("預測誤差最大的前 10 張圖片")
    print("="*60)
    print(df_sorted[['Image', 'FogScore', 'BestPsi', 'PsiPredicted', 'AbsError', 'PercentError']].head(10).to_string(index=False))


def main():
    print("開始分析 FogScore 與 BestPsi 的關係...")
    print("="*60)
    
    # 1. 讀取並合併資料
    df = load_and_merge_data()
    
    # 2. 基本統計
    print("\n" + "="*60)
    print("基本統計")
    print("="*60)
    print(f"FogScore 範圍: {df['FogScore'].min()} - {df['FogScore'].max()}")
    print(f"FogScore 平均: {df['FogScore'].mean():.2f}")
    print(f"FogScore 標準差: {df['FogScore'].std():.2f}")
    print(f"\nBestPsi 範圍: {df['BestPsi'].min():.2f} - {df['BestPsi'].max():.2f}")
    print(f"BestPsi 平均: {df['BestPsi'].mean():.2f}")
    print(f"BestPsi 標準差: {df['BestPsi'].std():.2f}")
    
    # 3. 找出關係式
    results = find_relationship(df['FogScore'].values, df['BestPsi'].values)
    
    # 4. 繪圖
    plot_relationship(df, results)
    
    # 5. 儲存比較結果
    save_comparison_results(df, results)
    
    print("\n" + "="*60)
    print("✅ 分析完成")
    print("="*60)
    print(f"\n建議使用的關係式：")
    print(f"BestPsi = {results['slope']:.6f} × FogScore + {results['intercept']:.6f}")
    print(f"\n這個關係式可以用來根據霧氣評分預測最佳的 Psi 參數！")


if __name__ == "__main__":
    main()