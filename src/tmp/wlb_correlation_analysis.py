import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
df = pd.read_csv('../data/data.csv')

print("=== ワークライフバランス相関分析 ===\n")
print(f"データサイズ: {df.shape}")

# 数値列のみを抽出（ワークライフバランス以外）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'WorkLifeBalance' in numeric_cols:
    numeric_cols.remove('WorkLifeBalance')

print(f"分析対象カラム数: {len(numeric_cols)}")

# ワークライフバランスとの相関計算
correlations = {}
p_values = {}

for col in numeric_cols:
    try:
        # 欠損値を除外して相関計算
        valid_data = df[[col, 'WorkLifeBalance']].dropna()
        if len(valid_data) > 0:
            corr_coef = np.corrcoef(valid_data[col], valid_data['WorkLifeBalance'])[0, 1]
            stat, p_val = stats.pearsonr(valid_data[col], valid_data['WorkLifeBalance'])
            correlations[col] = corr_coef
            p_values[col] = p_val
    except Exception as e:
        print(f"エラー - {col}: {e}")
        correlations[col] = np.nan
        p_values[col] = np.nan

# 結果をデータフレームに整理
corr_df = pd.DataFrame({
    'Column': list(correlations.keys()),
    'Correlation': list(correlations.values()),
    'P_Value': list(p_values.values()),
    'Abs_Correlation': [abs(x) if not np.isnan(x) else 0 for x in correlations.values()]
}).sort_values('Abs_Correlation', ascending=False)

# 統計的有意性のフラグ追加
corr_df['Significant'] = corr_df['P_Value'] < 0.05

print("=== 相関分析結果 (上位20位) ===")
print(corr_df.head(20).to_string(index=False, float_format='%.4f'))

print(f"\n=== 統計的に有意な相関 (p < 0.05) ===")
significant_corr = corr_df[corr_df['Significant'] == True].head(15)
print(significant_corr.to_string(index=False, float_format='%.4f'))

# 最高相関カラムの詳細分析
top_corr_col = corr_df.iloc[0]['Column']
top_corr_val = corr_df.iloc[0]['Correlation']
top_p_val = corr_df.iloc[0]['P_Value']

print(f"\n=== 最高相関カラム詳細分析 ===")
print(f"カラム: {top_corr_col}")
print(f"相関係数: {top_corr_val:.4f}")
print(f"p値: {top_p_val:.4f}")

# 基本統計
print(f"\n{top_corr_col}の基本統計:")
print(df[top_corr_col].describe())

print(f"\nWorkLifeBalanceの基本統計:")
print(df['WorkLifeBalance'].describe())

# 可視化
plt.figure(figsize=(20, 16))

# 1. 上位相関カラムのバープロット
plt.subplot(3, 4, 1)
top_15 = corr_df.head(15)
colors = ['red' if sig else 'lightblue' for sig in top_15['Significant']]
bars = plt.barh(range(len(top_15)), top_15['Correlation'], color=colors)
plt.yticks(range(len(top_15)), top_15['Column'])
plt.xlabel('相関係数')
plt.title('図1: ワークライフバランス相関ランキング（上位15位）\n赤：統計的有意、青：非有意')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
plt.gca().invert_yaxis()

# 相関値をバーに表示
for i, (bar, corr) in enumerate(zip(bars, top_15['Correlation'])):
    plt.text(corr + (0.01 if corr >= 0 else -0.01), i, f'{corr:.3f}', 
             va='center', ha='left' if corr >= 0 else 'right', fontsize=8)

# 2. 最高相関カラムとの散布図
plt.subplot(3, 4, 2)
plt.scatter(df[top_corr_col], df['WorkLifeBalance'], alpha=0.6)
plt.xlabel(f'{top_corr_col}')
plt.ylabel('ワークライフバランス')
plt.title(f'図2: {top_corr_col} vs ワークライフバランス\n相関係数: {top_corr_val:.4f}')
# トレンドライン
z = np.polyfit(df[top_corr_col], df['WorkLifeBalance'], 1)
p = np.poly1d(z)
plt.plot(df[top_corr_col], p(df[top_corr_col]), "r--", alpha=0.8)

# 3. ワークライフバランス別の最高相関カラム分布
plt.subplot(3, 4, 3)
if df[top_corr_col].dtype in ['int64', 'float64'] and df[top_corr_col].nunique() > 10:
    # 連続値の場合は箱ひげ図
    sns.boxplot(data=df, x='WorkLifeBalance', y=top_corr_col)
    plt.title(f'図3: ワークライフバランス別{top_corr_col}分布')
else:
    # カテゴリ値の場合はクロス集計
    cross_tab = pd.crosstab(df['WorkLifeBalance'], df[top_corr_col], normalize='index') * 100
    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='Blues')
    plt.title(f'図3: ワークライフバランス×{top_corr_col}クロス集計（%）')

# 4. 相関の強さ分布
plt.subplot(3, 4, 4)
plt.hist(corr_df['Abs_Correlation'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('絶対相関係数')
plt.ylabel('カラム数')
plt.title('図4: 相関の強さ分布')
plt.axvline(x=0.1, color='orange', linestyle='--', label='弱い相関(0.1)')
plt.axvline(x=0.3, color='red', linestyle='--', label='中程度相関(0.3)')
plt.legend()

# 5-8. 上位相関カラムとの詳細散布図（2-5位）
for i, idx in enumerate([1, 2, 3, 4], 5):
    if idx < len(corr_df):
        col_name = corr_df.iloc[idx]['Column']
        corr_val = corr_df.iloc[idx]['Correlation']
        
        plt.subplot(3, 4, i)
        plt.scatter(df[col_name], df['WorkLifeBalance'], alpha=0.6)
        plt.xlabel(f'{col_name}')
        plt.ylabel('ワークライフバランス')
        plt.title(f'図{i}: {col_name}\n相関: {corr_val:.4f}')
        
        # トレンドライン
        z = np.polyfit(df[col_name], df['WorkLifeBalance'], 1)
        p = np.poly1d(z)
        plt.plot(df[col_name], p(df[col_name]), "r--", alpha=0.8)

# 9. 正の相関 vs 負の相関
plt.subplot(3, 4, 9)
positive_corr = corr_df[corr_df['Correlation'] > 0]
negative_corr = corr_df[corr_df['Correlation'] < 0]
zero_corr = corr_df[abs(corr_df['Correlation']) < 0.01]

categories = ['正の相関', '負の相関', 'ほぼ無相関']
counts = [len(positive_corr), len(negative_corr), len(zero_corr)]
colors = ['lightgreen', 'lightcoral', 'lightgray']

plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
plt.title('図9: 相関の方向性分布')

# 10. 相関係数のヒートマップ（上位10カラム）
plt.subplot(3, 4, 10)
top_10_cols = corr_df.head(10)['Column'].tolist()
subset_df = df[top_10_cols + ['WorkLifeBalance']]
corr_matrix = subset_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
plt.title('図10: 上位10カラム相関マトリックス')

# 11. 統計的有意性の可視化
plt.subplot(3, 4, 11)
sig_counts = corr_df['Significant'].value_counts()
plt.pie(sig_counts.values, labels=['非有意(p≥0.05)', '有意(p<0.05)'], 
        colors=['lightblue', 'red'], autopct='%1.1f%%')
plt.title('図11: 統計的有意性分布')

# 12. 相関強度カテゴリ分布
plt.subplot(3, 4, 12)
def categorize_correlation(abs_corr):
    if abs_corr < 0.1:
        return '非常に弱い'
    elif abs_corr < 0.3:
        return '弱い'
    elif abs_corr < 0.5:
        return '中程度'
    elif abs_corr < 0.7:
        return '強い'
    else:
        return '非常に強い'

corr_df['Strength_Category'] = corr_df['Abs_Correlation'].apply(categorize_correlation)
strength_counts = corr_df['Strength_Category'].value_counts()
plt.pie(strength_counts.values, labels=strength_counts.index, autopct='%1.1f%%')
plt.title('図12: 相関強度カテゴリ分布')

plt.tight_layout()
plt.savefig('wlb_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# カテゴリ別詳細分析
print(f"\n=== カテゴリ別相関分析 ===")

# 強い相関（|r| >= 0.3）
strong_corr = corr_df[corr_df['Abs_Correlation'] >= 0.3]
if not strong_corr.empty:
    print(f"\n強い相関カラム（|r| ≥ 0.3）: {len(strong_corr)}個")
    print(strong_corr.to_string(index=False, float_format='%.4f'))
else:
    print("\n強い相関カラム（|r| ≥ 0.3）: なし")

# 中程度の相関（0.1 <= |r| < 0.3）
moderate_corr = corr_df[(corr_df['Abs_Correlation'] >= 0.1) & (corr_df['Abs_Correlation'] < 0.3)]
print(f"\n中程度相関カラム（0.1 ≤ |r| < 0.3）: {len(moderate_corr)}個")
if len(moderate_corr) > 0:
    print(moderate_corr.head(10).to_string(index=False, float_format='%.4f'))

# 弱い相関（|r| < 0.1）
weak_corr = corr_df[corr_df['Abs_Correlation'] < 0.1]
print(f"\n弱い相関カラム（|r| < 0.1）: {len(weak_corr)}個")

# 上位相関カラムのビジネス解釈
print(f"\n=== 上位相関カラムのビジネス解釈 ===")
top_5 = corr_df.head(5)
for idx, row in top_5.iterrows():
    col = row['Column']
    corr = row['Correlation']
    p_val = row['P_Value']
    significant = "統計的に有意" if p_val < 0.05 else "統計的に非有意"
    
    direction = "正の関係" if corr > 0 else "負の関係"
    strength = categorize_correlation(abs(corr))
    
    print(f"\n{idx+1}位: {col}")
    print(f"  - 相関係数: {corr:.4f} ({strength}{direction})")
    print(f"  - 統計的有意性: {significant} (p={p_val:.4f})")
    
    # ビジネス解釈の例
    if col in ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']:
        print(f"  - 解釈: {col}が高いほどワークライフバランスが{'良好' if corr > 0 else '悪化'}する傾向")
    elif col in ['StressRating', 'OverTime']:
        print(f"  - 解釈: {col}が高いほどワークライフバランスが{'良好' if corr > 0 else '悪化'}する傾向")
    elif col in ['Age', 'TotalWorkingYears']:
        print(f"  - 解釈: {col}が高いほどワークライフバランスが{'良好' if corr > 0 else '悪化'}する傾向")

print(f"\n=== 分析結論 ===")
print(f"ワークライフバランスと最も相関が高いカラム: {top_corr_col} (r={top_corr_val:.4f})")
if top_p_val < 0.05:
    print(f"この相関は統計的に有意です（p={top_p_val:.4f} < 0.05）")
else:
    print(f"この相関は統計的に有意ではありません（p={top_p_val:.4f} ≥ 0.05）")

strength = categorize_correlation(abs(top_corr_val))
print(f"相関の強度: {strength}") 