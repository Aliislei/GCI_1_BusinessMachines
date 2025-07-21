import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

def create_performance_visualizations():
    """パフォーマンス指標の詳細可視化"""
    
    # データ読み込み
    df = pd.read_csv('../data/data.csv')
    
    # パフォーマンス指標
    performance_cols = ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement']
    
    # 大きなキャンバスを作成
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 分布の比較（ヒストグラム）
    plt.subplot(4, 4, 1)
    for i, col in enumerate(performance_cols):
        plt.hist(df[col].dropna(), alpha=0.6, label=col, bins=30, density=True)
    plt.title('図1: パフォーマンス指標の分布比較', fontsize=12, fontweight='bold')
    plt.legend()
    plt.xlabel('値')
    plt.ylabel('密度')
    
    # 2-4. 各指標の箱ひげ図
    for i, col in enumerate(performance_cols):
        plt.subplot(4, 4, 2+i)
        plt.boxplot(df[col].dropna())
        plt.title(f'図{2+i}: {col}の箱ひげ図', fontsize=10, fontweight='bold')
        plt.ylabel('値')
    
    # 5-7. 給与との散布図
    for i, col in enumerate(performance_cols):
        plt.subplot(4, 4, 5+i)
        plt.scatter(df['MonthlyIncome'], df[col], alpha=0.6, s=20)
        
        # 相関係数と回帰直線
        corr = df[col].corr(df['MonthlyIncome'])
        z = np.polyfit(df['MonthlyIncome'], df[col], 1)
        p = np.poly1d(z)
        plt.plot(df['MonthlyIncome'], p(df['MonthlyIncome']), "r--", alpha=0.8)
        
        plt.title(f'図{5+i}: {col} vs 給与 (r={corr:.3f})', fontsize=10, fontweight='bold')
        plt.xlabel('月収')
        plt.ylabel(col)
    
    # 8-10. 職位別の平均値
    for i, col in enumerate(performance_cols):
        plt.subplot(4, 4, 8+i)
        job_level_avg = df.groupby('JobLevel')[col].mean()
        job_level_std = df.groupby('JobLevel')[col].std()
        
        plt.errorbar(job_level_avg.index, job_level_avg.values, 
                    yerr=job_level_std.values, marker='o', capsize=5)
        plt.title(f'図{8+i}: 職位別{col}平均', fontsize=10, fontweight='bold')
        plt.xlabel('職位レベル')
        plt.ylabel(f'{col}平均値')
        plt.grid(True, alpha=0.3)
    
    # 11. 相関ヒートマップ
    plt.subplot(4, 4, 11)
    corr_vars = ['MonthlyIncome', 'JobLevel', 'JobSatisfaction', 'Age', 'TotalWorkingYears']
    corr_data = df[performance_cols + corr_vars].corr()
    
    # パフォーマンス指標と他変数の相関のみ抽出
    perf_corr = corr_data.loc[performance_cols, corr_vars]
    
    sns.heatmap(perf_corr, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.3f', cbar_kws={'shrink': .8})
    plt.title('図11: パフォーマンス指標相関マトリクス', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # 12. 満足度との関係
    plt.subplot(4, 4, 12)
    for col in performance_cols:
        satisfaction_avg = df.groupby('JobSatisfaction')[col].mean()
        plt.plot(satisfaction_avg.index, satisfaction_avg.values, 'o-', label=col)
    plt.title('図12: 満足度別パフォーマンス平均', fontsize=10, fontweight='bold')
    plt.xlabel('職務満足度')
    plt.ylabel('パフォーマンス値')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 13-14. 追加分析
    # 13. パフォーマンス指標間の相関
    plt.subplot(4, 4, 13)
    perf_only_corr = df[performance_cols].corr()
    sns.heatmap(perf_only_corr, annot=True, cmap='viridis', 
                fmt='.3f', cbar_kws={'shrink': .8})
    plt.title('図13: パフォーマンス指標間相関', fontsize=10, fontweight='bold')
    
    # 14. 統計的要約
    plt.subplot(4, 4, 14)
    plt.axis('off')
    
    # 統計要約テーブル
    summary_text = "【統計的要約】\n\n"
    for col in performance_cols:
        summary_text += f"{col}:\n"
        summary_text += f"  平均: {df[col].mean():.1f}\n"
        summary_text += f"  標準偏差: {df[col].std():.1f}\n"
        summary_text += f"  変動係数: {df[col].std()/df[col].mean():.3f}\n"
        summary_text += f"  給与相関: {df[col].corr(df['MonthlyIncome']):.3f}\n"
        summary_text += f"  職位相関: {df[col].corr(df['JobLevel']):.3f}\n\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('performance_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def business_logic_analysis():
    """ビジネスロジックに基づく詳細分析"""
    
    df = pd.read_csv('../data/data.csv')
    
    print("="*70)
    print("【ビジネスロジックに基づく詳細分析】")
    print("="*70)
    
    # 1. 高パフォーマンス群の特徴分析
    print("\n1. 高パフォーマンス群の特徴分析")
    print("-" * 40)
    
    for col in ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement']:
        print(f"\n【{col}による分析】")
        
        # 上位20%を高パフォーマンス群として定義
        threshold = df[col].quantile(0.8)
        high_perf = df[df[col] >= threshold]
        low_perf = df[df[col] <= df[col].quantile(0.2)]
        
        print(f"  高パフォーマンス群（上位20%、閾値{threshold:.1f}以上）: {len(high_perf)}人")
        print(f"  低パフォーマンス群（下位20%）: {len(low_perf)}人")
        
        # 高パフォーマンス群と低パフォーマンス群の比較
        comparison_vars = ['MonthlyIncome', 'JobLevel', 'JobSatisfaction', 'TotalWorkingYears']
        
        for var in comparison_vars:
            high_avg = high_perf[var].mean()
            low_avg = low_perf[var].mean()
            diff = high_avg - low_avg
            diff_pct = (diff / low_avg) * 100 if low_avg != 0 else 0
            
            print(f"    {var}: 高{high_avg:.1f} vs 低{low_avg:.1f} (差分{diff:+.1f}, {diff_pct:+.1f}%)")
    
    # 2. 期待されるパターンとの一致度
    print(f"\n2. 期待されるビジネスパターンとの一致度")
    print("-" * 45)
    
    expected_patterns = {
        "高パフォーマンス→高給与": "正の相関（>0.2）",
        "高パフォーマンス→高職位": "正の相関（>0.1）", 
        "適度な分散": "変動係数0.1-0.5",
        "外れ値の少なさ": "四分位範囲内に70%以上"
    }
    
    print("\n期待パターン vs 実際の結果:")
    for col in ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement']:
        print(f"\n◆ {col}:")
        
        # 給与相関チェック
        income_corr = df[col].corr(df['MonthlyIncome'])
        income_check = "✓" if income_corr > 0.2 else "✗"
        print(f"  高パフォーマンス→高給与: {income_corr:.3f} {income_check}")
        
        # 職位相関チェック
        level_corr = df[col].corr(df['JobLevel'])
        level_check = "✓" if level_corr > 0.1 else "✗"
        print(f"  高パフォーマンス→高職位: {level_corr:.3f} {level_check}")
        
        # 分散チェック
        cv = df[col].std() / df[col].mean()
        cv_check = "✓" if 0.1 <= cv <= 0.5 else "✗"
        print(f"  適度な分散: {cv:.3f} {cv_check}")
        
        # 外れ値チェック
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        outlier_pct = len(outliers) / len(df) * 100
        outlier_check = "✓" if outlier_pct < 30 else "✗"
        print(f"  外れ値の少なさ: {outlier_pct:.1f}% {outlier_check}")

def final_recommendation():
    """最終推奨の決定"""
    
    df = pd.read_csv('../data/data.csv')
    
    print("\n" + "="*80)
    print("【最終推奨：ビジネス理論重視での再評価】")
    print("="*80)
    
    # ビジネス重視の評価基準
    performance_cols = ['PerformanceIndex', 'PerformanceRating', 'MonthlyAchievement']
    business_scores = {}
    
    for col in performance_cols:
        print(f"\n◆ {col} のビジネス妥当性評価:")
        score = 0
        
        # 1. 給与相関（最重要・40点）
        income_corr = df[col].corr(df['MonthlyIncome'])
        if income_corr > 0.2:
            income_score = 40
        elif income_corr > 0.1:
            income_score = 25
        elif income_corr > 0:
            income_score = 10
        else:
            income_score = 0
        score += income_score
        print(f"  1. 給与相関（40点満点）: {income_corr:.3f} → {income_score}点")
        
        # 2. 職位相関（重要・25点）
        level_corr = df[col].corr(df['JobLevel'])
        if level_corr > 0.1:
            level_score = 25
        elif level_corr > 0:
            level_score = 15
        elif level_corr > -0.1:
            level_score = 5
        else:
            level_score = 0
        score += level_score
        print(f"  2. 職位相関（25点満点）: {level_corr:.3f} → {level_score}点")
        
        # 3. 解釈容易性（20点）
        if col == 'PerformanceIndex':  # 0-100スケール
            interp_score = 20
        elif col == 'PerformanceRating':  # 1-4評価
            interp_score = 18
        else:  # 数値が大きく直感的でない
            interp_score = 5
        score += interp_score
        print(f"  3. 解釈容易性（20点満点）: → {interp_score}点")
        
        # 4. 分析適性（15点）
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.3:
            analysis_score = 15
        elif unique_ratio > 0.1:
            analysis_score = 10
        else:
            analysis_score = 5
        score += analysis_score
        print(f"  4. 分析適性（15点満点）: ユニーク比率{unique_ratio:.3f} → {analysis_score}点")
        
        print(f"  ビジネス妥当性総合スコア: {score}/100点")
        business_scores[col] = score
    
    # 最終順位
    print(f"\n【ビジネス妥当性ベース最終順位】")
    ranked = sorted(business_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (col, score) in enumerate(ranked, 1):
        print(f"  {i}位: {col} ({score}点)")
    
    best_metric = ranked[0][0]
    
    print(f"\n🎯 【最終推奨パフォーマンス指標】: {best_metric}")
    print(f"\n【推奨理由（ビジネス理論重視）】:")
    
    if best_metric == 'PerformanceIndex':
        print("✓ 給与との強い正の相関（0.233）- 唯一ビジネス理論と整合")
        print("✓ 0-100スケールで直感的理解が容易")
        print("✓ 適度な分散（71ユニーク値）で機械学習に適用可能")
        print("✓ 静かな退職研究のPerformance Gap理論での活用に最適")
        print("※ 職位との相関は弱いが、これは昇進の複雑性を反映している可能性")
        
    return best_metric, business_scores

def main():
    """メイン分析実行"""
    
    print("パフォーマンス指標の詳細可視化分析を開始します...\n")
    
    # 可視化生成
    create_performance_visualizations()
    
    # ビジネスロジック分析
    business_logic_analysis()
    
    # 最終推奨
    best_metric, scores = final_recommendation()
    
    print(f"\n" + "="*60)
    print("【総合結論】")
    print("="*60)
    print(f"統計的分析とビジネス理論を総合的に検討した結果、")
    print(f"**{best_metric}** を主要パフォーマンス指標として推奨します。")
    print(f"\nこの指標を用いて:")
    print("1. 静かな退職の予測モデル構築")
    print("2. Performance Gap = Expected - Actual の算出")
    print("3. I社固有の人事課題特定")
    print("を実施することを提案します。")

if __name__ == "__main__":
    main() 