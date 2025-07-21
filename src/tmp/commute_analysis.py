import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
df = pd.read_csv('../data/data.csv')

print("=== 通勤時間特徴量エンジニアリング分析 ===\n")
print(f"データサイズ: {df.shape}")
print(f"基本統計（関連カラム）:")
print(df[['DistanceFromHome', 'OverTime', 'RemoteWork', 'WorkLifeBalance']].describe())

def convert_distance_to_commute_time(distance):
    """距離から往復通勤時間を計算する関数"""
    if distance <= 10:
        return 1.0  # 往復1時間
    elif distance <= 30:
        return 2.0  # 往復2時間
    else:
        return (distance / 40) * 2  # 往復時間（時速40km/hで計算）

def calculate_weekly_business_constraint(row):
    """週間業務拘束時間を計算する関数"""
    # 1. 距離から往復通勤時間を算出
    daily_commute_time = convert_distance_to_commute_time(row['DistanceFromHome'])
    
    # 2. リモートワーク頻度を考慮した週間通勤時間
    weekly_commute_time = (5 - row['RemoteWork']) * daily_commute_time
    
    # 3. 月残業時間を週平均に変換
    weekly_overtime = row['OverTime'] / 4
    
    # 4. 週間業務拘束時間の算出
    weekly_total_constraint = weekly_commute_time + 40 + weekly_overtime
    
    return {
        'daily_commute_time': daily_commute_time,
        'weekly_commute_time': weekly_commute_time, 
        'weekly_overtime': weekly_overtime,
        'weekly_business_constraint': weekly_total_constraint
    }

# 特徴量エンジニアリング実行
print("\n=== 特徴量エンジニアリング実行 ===")

# 各行に対して計算実行
results = df.apply(calculate_weekly_business_constraint, axis=1)

# 結果をデータフレームに追加
df['Daily_Commute_Time'] = [r['daily_commute_time'] for r in results]
df['Weekly_Commute_Time'] = [r['weekly_commute_time'] for r in results]  
df['Weekly_Overtime'] = [r['weekly_overtime'] for r in results]
df['Weekly_Business_Constraint'] = [r['weekly_business_constraint'] for r in results]

print("新しい特徴量の統計:")
new_features = ['Daily_Commute_Time', 'Weekly_Commute_Time', 'Weekly_Overtime', 'Weekly_Business_Constraint']
print(df[new_features].describe())

# 相関性分析
print("\n=== 相関性分析 ===")

# ワークライフバランス関連指標
wlb_features = ['WorkLifeBalance', 'StressRating', 'JobSatisfaction', 'EnvironmentSatisfaction']
business_features = ['Weekly_Business_Constraint', 'Weekly_Commute_Time', 'Weekly_Overtime', 'OverTime']

# 相関マトリックス計算
correlation_matrix = df[business_features + wlb_features].corr()

print("相関マトリックス:")
print(correlation_matrix.loc[business_features, wlb_features])

# 特に週間業務拘束時間とワークライフバランスの相関
constraint_series = df['Weekly_Business_Constraint'] 
wlb_series = df['WorkLifeBalance']
constraint_wlb_corr = np.corrcoef(constraint_series, wlb_series)[0, 1]
print(f"\n週間業務拘束時間 × ワークライフバランス相関係数: {constraint_wlb_corr:.4f}")

# 統計的有意性検定
stat, p_value = stats.pearsonr(constraint_series, wlb_series)
print(f"ピアソン相関係数: {stat:.4f}, p値: {p_value:.4f}")

# 可視化
plt.figure(figsize=(16, 12))

# 1. 相関ヒートマップ
plt.subplot(2, 3, 1)
sns.heatmap(correlation_matrix.loc[business_features, wlb_features], 
            annot=True, cmap='RdBu_r', center=0, fmt='.3f')
plt.title('図1: 業務拘束時間とワークライフバランス相関マトリックス')

# 2. 散布図：週間業務拘束時間 vs ワークライフバランス
plt.subplot(2, 3, 2)
plt.scatter(df['Weekly_Business_Constraint'], df['WorkLifeBalance'], alpha=0.6)
plt.xlabel('週間業務拘束時間（時間）')
plt.ylabel('ワークライフバランス評価（1-4）')
plt.title(f'図2: 週間業務拘束時間とワークライフバランス\n相関係数: {constraint_wlb_corr:.4f}')
# トレンドライン追加
z = np.polyfit(df['Weekly_Business_Constraint'], df['WorkLifeBalance'], 1)
p = np.poly1d(z)
plt.plot(df['Weekly_Business_Constraint'], p(df['Weekly_Business_Constraint']), "r--", alpha=0.8)

# 3. 箱ひげ図：ワークライフバランス別の週間業務拘束時間
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='WorkLifeBalance', y='Weekly_Business_Constraint')
plt.xlabel('ワークライフバランス評価')
plt.ylabel('週間業務拘束時間（時間）')
plt.title('図3: ワークライフバランス別業務拘束時間分布')

# 4. 週間通勤時間の分布
plt.subplot(2, 3, 4)
plt.hist(df['Weekly_Commute_Time'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('週間通勤時間（時間）')
plt.ylabel('従業員数')
plt.title('図4: 週間通勤時間分布')

# 5. リモートワーク頻度と週間通勤時間の関係
plt.subplot(2, 3, 5)
sns.boxplot(data=df, x='RemoteWork', y='Weekly_Commute_Time')
plt.xlabel('リモートワーク頻度（0-5）')
plt.ylabel('週間通勤時間（時間）')
plt.title('図5: リモートワーク頻度と週間通勤時間')

# 6. 業務拘束時間の成分内訳（通勤・標準労働・残業）
plt.subplot(2, 3, 6)
components = ['通勤時間', '標準労働時間', '残業時間']
means = [df['Weekly_Commute_Time'].mean(), 40, df['Weekly_Overtime'].mean()]
plt.bar(components, means, alpha=0.7)
plt.ylabel('週間時間（時間）')
plt.title('図6: 週間業務拘束時間の成分内訳')
for i, v in enumerate(means):
    plt.text(i, v + 0.5, f'{v:.1f}h', ha='center')

plt.tight_layout()
plt.savefig('commute_wlb_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 詳細分析結果
print("\n=== 詳細分析結果 ===")

# ワークライフバランス評価別の統計
print("\nワークライフバランス評価別の週間業務拘束時間:")
wlb_stats = df.groupby('WorkLifeBalance')['Weekly_Business_Constraint'].agg(['count', 'mean', 'std']).round(2)
print(wlb_stats)

# リモートワーク効果の定量化
print("\nリモートワーク効果の定量化:")
remote_effect = df.groupby('RemoteWork')[['Weekly_Commute_Time', 'Weekly_Business_Constraint', 'WorkLifeBalance']].mean().round(2)
print(remote_effect)

# 高拘束時間群の特定（上位25%）
high_constraint_threshold = df['Weekly_Business_Constraint'].quantile(0.75)
high_constraint_group = df[df['Weekly_Business_Constraint'] >= high_constraint_threshold]

print(f"\n高拘束時間群（上位25%、{high_constraint_threshold:.1f}時間以上）の特徴:")
print(f"- 人数: {len(high_constraint_group)}人 ({len(high_constraint_group)/len(df)*100:.1f}%)")
print(f"- 平均ワークライフバランス: {high_constraint_group['WorkLifeBalance'].mean():.2f}")
print(f"- 離職率: {(high_constraint_group['Attrition'] == 'Yes').mean()*100:.1f}%")

# 低拘束時間群との比較
low_constraint_threshold = df['Weekly_Business_Constraint'].quantile(0.25)
low_constraint_group = df[df['Weekly_Business_Constraint'] <= low_constraint_threshold]

print(f"\n低拘束時間群（下位25%、{low_constraint_threshold:.1f}時間以下）との比較:")
print(f"- 平均ワークライフバランス: {low_constraint_group['WorkLifeBalance'].mean():.2f}")
print(f"- 離職率: {(low_constraint_group['Attrition'] == 'Yes').mean()*100:.1f}%")

# 相関の解釈
print(f"\n=== 相関性の解釈 ===")
if abs(constraint_wlb_corr) < 0.1:
    strength = "非常に弱い"
elif abs(constraint_wlb_corr) < 0.3:
    strength = "弱い"
elif abs(constraint_wlb_corr) < 0.5:
    strength = "中程度"
elif abs(constraint_wlb_corr) < 0.7:
    strength = "強い"
else:
    strength = "非常に強い"

direction = "負の" if constraint_wlb_corr < 0 else "正の"
print(f"週間業務拘束時間とワークライフバランスの間には{strength}{direction}相関があります。")

if p_value < 0.05:
    print(f"この相関は統計的に有意です（p < 0.05）。")
else:
    print(f"この相関は統計的に有意ではありません（p ≥ 0.05）。")

print(f"\n業務拘束時間が1時間増加すると、ワークライフバランススコアは約{constraint_wlb_corr:.4f}ポイント変化する傾向があります。") 