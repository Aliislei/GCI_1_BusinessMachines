import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """データを読み込む"""
    return pd.read_csv('../data/data.csv')

def create_wlb_composite_index(df):
    """
    ワークライフバランス複合指標を作成
    
    4つのサブ指標を組み合わせ：
    1. 時間的負荷指標 (Time Load Score)
    2. ストレス・満足度指標 (Stress & Satisfaction Score)  
    3. 制度活用指標 (Work Flexibility Score)
    4. 総合ワークライフバランス指標 (Comprehensive WLB Score)
    """
    df = df.copy()
    
    # 1. 時間的負荷指標 (Time Load Score)
    # 負荷が低いほど高スコア（0-1に正規化）
    overtime_norm = 1 - (df['OverTime'] / df['OverTime'].max())
    distance_norm = 1 - (df['DistanceFromHome'] / df['DistanceFromHome'].max())
    
    # BusinessTravelを数値化（Non-Travel=0, Travel_Rarely=1, Travel_Frequently=2）
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    travel_numeric = df['BusinessTravel'].map(travel_map)
    travel_norm = 1 - (travel_numeric / travel_numeric.max())
    
    df['TimeLoadScore'] = (overtime_norm + distance_norm + travel_norm) / 3
    
    # 2. ストレス・満足度指標 (Stress & Satisfaction Score)
    # ストレスは逆転、満足度はそのまま（すべて0-1に正規化）
    stress_rating_norm = 1 - ((df['StressRating'] - 1) / (5 - 1))
    stress_self_norm = 1 - ((df['StressSelfReported'] - 1) / (5 - 1))
    job_sat_norm = (df['JobSatisfaction'] - 1) / (4 - 1)
    env_sat_norm = (df['EnvironmentSatisfaction'] - 1) / (4 - 1)
    rel_sat_norm = (df['RelationshipSatisfaction'] - 1) / (4 - 1)
    
    df['StressSatisfactionScore'] = (
        stress_rating_norm + stress_self_norm + job_sat_norm + env_sat_norm + rel_sat_norm
    ) / 5
    
    # 3. 制度活用指標 (Work Flexibility Score)
    # すべて高いほど良い（0-1に正規化）
    flexible_work_norm = df['FlexibleWork']  # 既に0-1
    remote_work_norm = df['RemoteWork'] / df['RemoteWork'].max()
    welfare_norm = (df['WelfareBenefits'] - 1) / (4 - 1)
    extended_leave_norm = df['ExtendedLeave']  # 既に0-1
    
    df['WorkFlexibilityScore'] = (
        flexible_work_norm + remote_work_norm + welfare_norm + extended_leave_norm
    ) / 4
    
    # 4. 総合ワークライフバランス指標 (Comprehensive WLB Score)
    # 既存のWorkLifeBalanceと組み合わせ（重み付け）
    wlb_existing_norm = (df['WorkLifeBalance'] - 1) / (4 - 1)
    
    # 重み設定（相関係数を考慮）
    weight_time = 0.20          # 時間的負荷
    weight_stress = 0.25        # ストレス・満足度
    weight_flexibility = 0.30   # 制度活用（最重要）
    weight_existing = 0.25      # 既存評価
    
    df['ComprehensiveWLBScore'] = (
        weight_time * df['TimeLoadScore'] +
        weight_stress * df['StressSatisfactionScore'] +
        weight_flexibility * df['WorkFlexibilityScore'] +
        weight_existing * wlb_existing_norm
    )
    
    # 100点満点に変換
    df['ComprehensiveWLBScore_100'] = df['ComprehensiveWLBScore'] * 100
    
    return df

def analyze_wlb_indicators(df):
    """ワークライフバランス指標の分析と可視化"""
    
    # 基本統計
    wlb_cols = ['TimeLoadScore', 'StressSatisfactionScore', 'WorkFlexibilityScore', 
                'ComprehensiveWLBScore', 'ComprehensiveWLBScore_100']
    
    print("=== ワークライフバランス複合指標の基本統計 ===")
    print(df[wlb_cols].describe())
    
    # 相関分析
    print("\n=== 指標間の相関関係 ===")
    correlation_matrix = df[wlb_cols + ['WorkLifeBalance']].corr()
    print(correlation_matrix)
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ワークライフバランス複合指標の分析', fontsize=16)
    
    # 各指標の分布
    axes[0,0].hist(df['TimeLoadScore'], bins=30, alpha=0.7, color='skyblue')
    axes[0,0].set_title('時間的負荷指標の分布')
    axes[0,0].set_xlabel('Time Load Score')
    
    axes[0,1].hist(df['StressSatisfactionScore'], bins=30, alpha=0.7, color='lightgreen')
    axes[0,1].set_title('ストレス・満足度指標の分布')
    axes[0,1].set_xlabel('Stress & Satisfaction Score')
    
    axes[0,2].hist(df['WorkFlexibilityScore'], bins=30, alpha=0.7, color='salmon')
    axes[0,2].set_title('制度活用指標の分布')
    axes[0,2].set_xlabel('Work Flexibility Score')
    
    axes[1,0].hist(df['ComprehensiveWLBScore_100'], bins=30, alpha=0.7, color='gold')
    axes[1,0].set_title('総合WLB指標の分布（100点満点）')
    axes[1,0].set_xlabel('Comprehensive WLB Score')
    
    # 相関ヒートマップ
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1,1], cbar_kws={'shrink': 0.8})
    axes[1,1].set_title('指標間相関ヒートマップ')
    
    # 既存WorkLifeBalanceとの比較散布図
    axes[1,2].scatter(df['WorkLifeBalance'], df['ComprehensiveWLBScore_100'], alpha=0.6)
    axes[1,2].set_xlabel('既存 Work Life Balance (1-4)')
    axes[1,2].set_ylabel('新規 Comprehensive WLB Score (0-100)')
    axes[1,2].set_title('既存指標との比較')
    
    plt.tight_layout()
    plt.savefig('wlb_composite_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def segment_analysis(df):
    """WLB指標によるセグメント分析"""
    
    # 総合WLB指標による4分位分析
    df['WLB_Quartile'] = pd.qcut(df['ComprehensiveWLBScore_100'], 
                                q=4, labels=['低WLB', '中低WLB', '中高WLB', '高WLB'])
    
    print("\n=== WLB四分位別の特徴分析 ===")
    
    # 各四分位の基本情報
    quartile_summary = df.groupby('WLB_Quartile').agg({
        'Age': 'mean',
        'OverTime': 'mean',
        'DistanceFromHome': 'mean',
        'StressRating': 'mean',
        'JobSatisfaction': 'mean',
        'Attrition': lambda x: (x == 'Yes').mean() * 100,  # 離職率
        'MonthlyIncome': 'mean'
    }).round(2)
    
    print(quartile_summary)
    
    # 離職率の可視化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    attrition_by_wlb = df.groupby('WLB_Quartile')['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100
    )
    bars = plt.bar(attrition_by_wlb.index, attrition_by_wlb.values, 
                   color=['red', 'orange', 'lightblue', 'green'])
    plt.title('WLB四分位別の離職率 (%)')
    plt.ylabel('離職率 (%)')
    for bar, value in zip(bars, attrition_by_wlb.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center')
    
    # 残業時間の分布
    plt.subplot(2, 2, 2)
    df.boxplot(column='OverTime', by='WLB_Quartile', ax=plt.gca())
    plt.title('WLB四分位別の残業時間分布')
    plt.suptitle('')  # デフォルトのタイトルを削除
    
    # ストレス評価の分布
    plt.subplot(2, 2, 3)
    df.boxplot(column='StressRating', by='WLB_Quartile', ax=plt.gca())
    plt.title('WLB四分位別のストレス評価分布')
    plt.suptitle('')
    
    # 月収の分布
    plt.subplot(2, 2, 4)
    df.boxplot(column='MonthlyIncome', by='WLB_Quartile', ax=plt.gca())
    plt.title('WLB四分位別の月収分布')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('wlb_segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return quartile_summary

if __name__ == "__main__":
    # データ読み込み
    df = load_data()
    
    # 複合指標作成
    df_with_wlb = create_wlb_composite_index(df)
    
    # 分析実行
    correlation_matrix = analyze_wlb_indicators(df_with_wlb)
    quartile_summary = segment_analysis(df_with_wlb)
    
    # 結果をCSVで保存
    df_with_wlb.to_csv('../data/data_with_wlb_composite.csv', index=False)
    print(f"\n複合指標付きデータを保存しました: ../data/data_with_wlb_composite.csv")
    
    # 主要な知見をまとめる
    print("\n=== ワークライフバランス複合指標 主要知見 ===")
    print("1. 使用したカラム:")
    print("   - 時間的負荷: OverTime, DistanceFromHome, BusinessTravel")
    print("   - ストレス・満足: StressRating, StressSelfReported, JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction")
    print("   - 制度活用: FlexibleWork, RemoteWork, WelfareBenefits, ExtendedLeave")
    print("   - 既存評価: WorkLifeBalance")
    
    print(f"\n2. 総合WLB指標の統計:")
    print(f"   - 平均: {df_with_wlb['ComprehensiveWLBScore_100'].mean():.1f}点")
    print(f"   - 標準偏差: {df_with_wlb['ComprehensiveWLBScore_100'].std():.1f}点")
    print(f"   - 範囲: {df_with_wlb['ComprehensiveWLBScore_100'].min():.1f}-{df_with_wlb['ComprehensiveWLBScore_100'].max():.1f}点") 