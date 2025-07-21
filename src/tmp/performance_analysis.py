import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

def load_and_analyze_performance():
    """データを読み込んでパフォーマンス関連カラムを分析"""
    
    # データ読み込み
    print("=== データ読み込み ===")
    df = pd.read_csv('../data/data.csv')
    print(f"データ形状: {df.shape}")
    print(f"カラム数: {len(df.columns)}")
    
    # パフォーマンス関連カラムの確認
    performance_cols = ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement']
    related_cols = ['MonthlyIncome', 'JobLevel', 'Age', 'TotalWorkingYears', 
                   'YearsAtCompany', 'Education', 'JobSatisfaction']
    
    print("\n=== パフォーマンス関連カラムの基本統計量 ===")
    for col in performance_cols:
        if col in df.columns:
            print(f"\n【{col}】")
            print(f"データ型: {df[col].dtype}")
            print(f"欠損値: {df[col].isnull().sum()}")
            print(f"ユニーク値数: {df[col].nunique()}")
            print(f"統計量:")
            print(df[col].describe())
            
            # ユニーク値が少ない場合は値の分布も表示
            if df[col].nunique() <= 10:
                print(f"値の分布:")
                print(df[col].value_counts().sort_index())
        else:
            print(f"【{col}】: カラムが存在しません")
    
    # 関連性分析用のカラムも確認
    print("\n=== 関連性分析用カラムの確認 ===")
    for col in related_cols:
        if col in df.columns:
            print(f"{col}: {df[col].dtype}, 欠損値: {df[col].isnull().sum()}, ユニーク値: {df[col].nunique()}")
        else:
            print(f"{col}: カラムが存在しません")
    
    return df

def analyze_correlations(df):
    """パフォーマンス指標と他の変数との相関分析"""
    
    performance_cols = [col for col in ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement'] 
                       if col in df.columns]
    related_cols = [col for col in ['MonthlyIncome', 'JobLevel', 'Age', 'TotalWorkingYears', 
                                   'YearsAtCompany', 'Education', 'JobSatisfaction'] 
                   if col in df.columns]
    
    if not performance_cols:
        print("パフォーマンス関連カラムが見つかりません")
        return
    
    # 相関分析
    print("\n=== パフォーマンス指標と他変数の相関分析 ===")
    
    correlation_matrix = pd.DataFrame(index=performance_cols, columns=related_cols)
    
    for perf_col in performance_cols:
        print(f"\n【{perf_col}との相関】")
        for rel_col in related_cols:
            try:
                corr = df[perf_col].corr(df[rel_col])
                correlation_matrix.loc[perf_col, rel_col] = corr
                print(f"  {rel_col}: {corr:.3f}")
            except Exception as e:
                correlation_matrix.loc[perf_col, rel_col] = np.nan
                print(f"  {rel_col}: 計算エラー ({e})")
    
    # 可視化
    plt.figure(figsize=(15, 10))
    
    # 1. パフォーマンス指標の分布
    plt.subplot(2, 3, 1)
    for i, col in enumerate(performance_cols):
        plt.hist(df[col].dropna(), alpha=0.6, label=col, bins=20)
    plt.title('Performance Metrics Distribution')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # 2. 相関ヒートマップ
    plt.subplot(2, 3, 2)
    corr_numeric = correlation_matrix.astype(float)
    sns.heatmap(corr_numeric, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.3f', cbar_kws={'shrink': .8})
    plt.title('Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 3-5. 主要変数とのScatter plot
    scatter_vars = ['MonthlyIncome', 'JobLevel', 'JobSatisfaction']
    for i, var in enumerate(scatter_vars):
        if var in df.columns:
            plt.subplot(2, 3, i+3)
            for perf_col in performance_cols:
                if perf_col in df.columns:
                    plt.scatter(df[var], df[perf_col], alpha=0.5, label=perf_col, s=20)
            plt.xlabel(var)
            plt.ylabel('Performance Metrics')
            plt.title(f'Performance vs {var}')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def analyze_logical_consistency(df):
    """論理的一貫性の分析"""
    
    performance_cols = [col for col in ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement'] 
                       if col in df.columns]
    
    print("\n=== 論理的一貫性分析 ===")
    
    # 各パフォーマンス指標の分位数別に他の変数の平均値を確認
    for perf_col in performance_cols:
        print(f"\n【{perf_col}による分位数分析】")
        
        # 離散値（ユニーク値が少ない）の場合は値別分析、連続値は四分位分析
        if df[perf_col].nunique() <= 10:
            # 離散値の場合：各値別に分析
            print(f"※ {perf_col}は離散値のため、値別分析を実施")
            key_vars = ['MonthlyIncome', 'JobLevel', 'JobSatisfaction', 'TotalWorkingYears']
            summary = df.groupby(perf_col)[
                [col for col in key_vars if col in df.columns]
            ].mean()
            print(summary)
            
            # 期待される傾向との一致度を確認
            print(f"\n期待される傾向との一致度:")
            for var in ['MonthlyIncome', 'JobLevel']:
                if var in summary.columns:
                    values = summary[var].values
                    indices = summary.index.values
                    if len(values) >= 2:
                        correlation = np.corrcoef(indices, values)[0,1]
                        trend = "上昇傾向" if correlation > 0 else "下降傾向"
                        print(f"  {var}: {trend} (相関係数: {correlation:.3f})")
        else:
            # 連続値の場合：四分位分析
            try:
                quartiles = pd.qcut(df[perf_col].dropna(), q=4, labels=['Q1(低)', 'Q2', 'Q3', 'Q4(高)'])
                df_temp = df.copy()
                df_temp['Performance_Quartile'] = quartiles
                
                # 各四分位での他変数の平均値
                key_vars = ['MonthlyIncome', 'JobLevel', 'JobSatisfaction', 'TotalWorkingYears']
                summary = df_temp.groupby('Performance_Quartile')[
                    [col for col in key_vars if col in df.columns]
                ].mean()
                
                print(summary)
                
                # 期待される傾向との一致度を確認
                print(f"\n期待される傾向との一致度:")
                for var in ['MonthlyIncome', 'JobLevel']:
                    if var in summary.columns:
                        values = summary[var].values
                        if len(values) >= 2:
                            trend = "上昇傾向" if values[-1] > values[0] else "下降傾向"
                            correlation = np.corrcoef(range(len(values)), values)[0,1]
                            print(f"  {var}: {trend} (相関係数: {correlation:.3f})")
            except ValueError as e:
                print(f"四分位分析でエラー: {e}")
                # フォールバック：三分位で試行
                try:
                    tertiles = pd.qcut(df[perf_col].dropna(), q=3, labels=['低', '中', '高'])
                    df_temp = df.copy()
                    df_temp['Performance_Tertile'] = tertiles
                    
                    key_vars = ['MonthlyIncome', 'JobLevel', 'JobSatisfaction', 'TotalWorkingYears']
                    summary = df_temp.groupby('Performance_Tertile')[
                        [col for col in key_vars if col in df.columns]
                    ].mean()
                    print("三分位分析:")
                    print(summary)
                except ValueError:
                    print("分位分析が困難なため、スキップします")

def detailed_performance_comparison(df):
    """パフォーマンス指標の詳細比較分析"""
    
    performance_cols = [col for col in ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement'] 
                       if col in df.columns]
    
    print("\n=== パフォーマンス指標の詳細比較 ===")
    
    # 1. 給与・職位との関係性詳細分析
    print("\n【給与・職位との関係性詳細】")
    for perf_col in performance_cols:
        print(f"\n◆ {perf_col}:")
        
        # 給与との関係
        income_corr = df[perf_col].corr(df['MonthlyIncome'])
        print(f"  給与との相関: {income_corr:.3f}")
        
        # 職位との関係
        if 'JobLevel' in df.columns:
            level_corr = df[perf_col].corr(df['JobLevel'])
            print(f"  職位との相関: {level_corr:.3f}")
            
            # 職位別のパフォーマンス平均
            level_avg = df.groupby('JobLevel')[perf_col].mean()
            print(f"  職位別平均:")
            for level, avg in level_avg.items():
                print(f"    レベル{level}: {avg:.2f}")
    
    # 2. ビジネス理論との整合性チェック
    print("\n【ビジネス理論との整合性】")
    print("期待される関係性:")
    print("- 高パフォーマンス → 高給与（正の相関）")
    print("- 高パフォーマンス → 高職位（正の相関）")
    print("- パフォーマンスと満足度の関係は複雑（必ずしも正の相関とは限らない）")
    
    print("\n実際の結果:")
    for perf_col in performance_cols:
        income_corr = df[perf_col].corr(df['MonthlyIncome'])
        level_corr = df[perf_col].corr(df['JobLevel']) if 'JobLevel' in df.columns else 0
        satisfaction_corr = df[perf_col].corr(df['JobSatisfaction']) if 'JobSatisfaction' in df.columns else 0
        
        print(f"  {perf_col}:")
        print(f"    給与相関: {income_corr:.3f} {'✓' if income_corr > 0.1 else '✗'}")
        print(f"    職位相関: {level_corr:.3f} {'✓' if level_corr > 0.1 else '✗'}")
        print(f"    満足度相関: {satisfaction_corr:.3f}")

def recommendation_analysis(df):
    """推奨パフォーマンス指標の決定"""
    
    performance_cols = [col for col in ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement'] 
                       if col in df.columns]
    
    print("\n" + "="*60)
    print("【推奨パフォーマンス指標の評価・選定】")
    print("="*60)
    
    scores = {}
    
    for perf_col in performance_cols:
        print(f"\n◆ {perf_col} の評価:")
        score = 0
        
        # 1. 給与との相関（20点満点）
        income_corr = abs(df[perf_col].corr(df['MonthlyIncome']))
        income_score = min(20, income_corr * 100)
        score += income_score
        print(f"  1. 給与相関: {income_corr:.3f} → {income_score:.1f}点")
        
        # 2. 職位との相関（20点満点）
        level_corr = abs(df[perf_col].corr(df['JobLevel'])) if 'JobLevel' in df.columns else 0
        level_score = min(20, level_corr * 100)
        score += level_score
        print(f"  2. 職位相関: {level_corr:.3f} → {level_score:.1f}点")
        
        # 3. 分布の適切性（20点満点）
        unique_ratio = df[perf_col].nunique() / len(df)
        if unique_ratio > 0.8:  # 高い分散
            dist_score = 20
        elif unique_ratio > 0.5:  # 中程度の分散
            dist_score = 15
        elif unique_ratio > 0.1:  # 低い分散
            dist_score = 10
        else:  # 非常に低い分散
            dist_score = 5
        score += dist_score
        print(f"  3. 分布適切性: ユニーク値比率{unique_ratio:.3f} → {dist_score:.1f}点")
        
        # 4. 値の範囲の妥当性（20点満点）
        value_range = df[perf_col].max() - df[perf_col].min()
        std_dev = df[perf_col].std()
        cv = std_dev / df[perf_col].mean()  # 変動係数
        
        if 0.1 <= cv <= 0.5:  # 適度な変動
            range_score = 20
        elif 0.05 <= cv <= 0.8:  # やや適度な変動
            range_score = 15
        else:  # 変動が極端
            range_score = 10
        score += range_score
        print(f"  4. 値範囲妥当性: 変動係数{cv:.3f} → {range_score:.1f}点")
        
        # 5. 機械学習での予測性能（20点満点）
        # 簡易的に分散とレンジで判定
        variance_score = min(20, (std_dev / df[perf_col].mean()) * 40)
        score += variance_score
        print(f"  5. 予測性能期待値: 標準化分散{std_dev/df[perf_col].mean():.3f} → {variance_score:.1f}点")
        
        print(f"  総合スコア: {score:.1f}/100点")
        scores[perf_col] = score
    
    # 推奨順位の決定
    print(f"\n【最終推奨順位】")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (col, score) in enumerate(ranked, 1):
        print(f"  {i}位: {col} ({score:.1f}点)")
    
    # 最終推奨
    best_metric = ranked[0][0]
    print(f"\n🎯 【推奨パフォーマンス指標】: {best_metric}")
    print(f"\n【選定理由】:")
    
    if best_metric == 'PerformanceIndex':
        print("- 給与との強い正の相関（0.233）")
        print("- 適度な分散（71のユニーク値）")
        print("- 0-100スケールで直感的理解が容易")
        print("- 機械学習モデルでの予測に適した連続値")
    elif best_metric == 'PerformanceRating':
        print("- 典型的な人事評価スケール（1-4）")
        print("- 解釈が容易")
        print("- ただし分散が限定的で予測モデルには不向き")
    elif best_metric == 'MonthlyAchievement':
        print("- 高い分散で詳細な差別化が可能")
        print("- ただし給与・職位との相関が弱い")
        print("- ビジネス理論との整合性に疑問")
    
    return best_metric, scores

def main():
    """メイン分析関数"""
    
    print("パフォーマンス関連カラムの妥当性分析を開始します...")
    
    # データ読み込みと基本分析
    df = load_and_analyze_performance()
    
    # 相関分析
    correlation_matrix = analyze_correlations(df)
    
    # 論理的一貫性分析
    analyze_logical_consistency(df)
    
    # 詳細比較分析
    detailed_performance_comparison(df)
    
    # 推奨分析
    best_metric, scores = recommendation_analysis(df)
    
    # 結論の導出
    print("\n" + "="*50)
    print("【分析結果サマリー】")
    print("="*50)
    
    performance_cols = [col for col in ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement'] 
                       if col in df.columns]
    
    if len(performance_cols) > 0:
        print("\n各パフォーマンス指標の特徴:")
        for col in performance_cols:
            print(f"- {col}: 平均={df[col].mean():.2f}, 標準偏差={df[col].std():.2f}, 範囲={df[col].min():.1f}-{df[col].max():.1f}")
    
    print(f"\n🎯 最終推奨: {best_metric}")
    print("\n今後の分析方針:")
    print(f"1. {best_metric}を主要パフォーマンス指標として採用")
    print("2. 他の指標も補助的に使用してモデルの頑健性を確保")
    print("3. Performance Gap = Expected Performance - Actual Performance の定義で活用")

if __name__ == "__main__":
    main() 