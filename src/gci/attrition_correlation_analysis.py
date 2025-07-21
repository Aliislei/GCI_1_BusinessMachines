#!/usr/bin/env python3
"""
離職率・パフォーマンスとの相関分析スクリプト
GCI最終課題 - 機械学習による事業提案
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import japanize_matplotlib

def load_and_prepare_data():
    """データを読み込み、前処理を行う"""
    df = pd.read_csv('data/data.csv')
    
    # Attritionを数値に変換（Yes=1, No=0）
    df['Attrition_numeric'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

def encode_categorical_data(df):
    """カテゴリカルデータを数値にエンコード"""
    df_encoded = df.copy()
    
    # カテゴリカルカラムを特定
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Attritionは既に処理済みなので除外
    if 'Attrition' in categorical_columns:
        categorical_columns.remove('Attrition')
    
    # Label Encodingを実行
    le = LabelEncoder()
    for col in categorical_columns:
        df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
    
    return df_encoded, categorical_columns

def calculate_correlations(df, target_column):
    """指定したターゲットカラムとの相関を計算"""
    # 数値カラムのみを選択（エンコード済みカラムも含む）
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ターゲットカラムを除外（自己相関を避けるため）
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    # 分散が0のカラムを除外（全て同じ値のカラム）
    valid_columns = []
    for col in numeric_columns:
        if df[col].var() > 0 and not df[col].isna().all():
            valid_columns.append(col)
    
    # 相関を計算
    correlations = {}
    for col in valid_columns:
        corr = df[target_column].corr(df[col])
        if not pd.isna(corr):  # nanでない場合のみ追加
            correlations[col] = corr
    
    # 絶対値で並び替え（相関の強さで評価）
    correlations_sorted = dict(sorted(correlations.items(), 
                                    key=lambda x: abs(x[1]), reverse=True))
    
    return correlations_sorted

def display_correlation_results(correlations, target_name, top_n=15):
    """相関分析結果を表示"""
    print(f"=== {target_name}との相関が高いカラム Top {top_n} ===")
    print("順位 | カラム名 | 相関係数")
    print("-" * 50)
    
    for i, (col, corr) in enumerate(list(correlations.items())[:top_n], 1):
        print(f"{i:2d}位 | {col:<25} | {corr:7.4f}")
    
    print(f"\n注：相関係数の範囲は -1.0 ～ 1.0")
    print(f"    正の値：{target_name}と正の相関")
    print(f"    負の値：{target_name}と負の相関")
    print(f"    絶対値が大きいほど相関が強い")

def create_correlation_visualization(correlations, target_name, filename, top_n=15):
    """相関分析結果を可視化"""
    top_cols = list(correlations.keys())[:top_n]
    top_corrs = [correlations[col] for col in top_cols]
    
    plt.figure(figsize=(14, 10))
    colors = ['red' if x > 0 else 'blue' for x in top_corrs]
    bars = plt.barh(range(len(top_cols)), top_corrs, color=colors, alpha=0.7)
    
    plt.yticks(range(len(top_cols)), 
               [col.replace('_encoded', '') for col in top_cols])
    plt.xlabel('相関係数')
    plt.title(f'{target_name}との相関が高いカラム Top {top_n}', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # 0の線を追加
    plt.axvline(x=0, color='black', linewidth=0.8, alpha=0.8)
    
    # 相関値をバーに表示
    for i, (bar, corr) in enumerate(zip(bars, top_corrs)):
        plt.text(corr + (0.01 if corr > 0 else -0.01), i, 
                f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def create_integrated_correlation_table(attrition_corr, performance_corr, stress_corr):
    """3つの指標の相関を統合した表を作成し、fleetingに保存"""
    import os
    from datetime import datetime
    
    # 全てのカラムを取得（重複なし）
    all_columns = set()
    all_columns.update(attrition_corr.keys())
    all_columns.update(performance_corr.keys()) 
    all_columns.update(stress_corr.keys())
    
    # データの整理
    integrated_data = []
    for col in sorted(all_columns):
        attrition_val = attrition_corr.get(col, 0.0)
        performance_val = performance_corr.get(col, 0.0)
        stress_val = stress_corr.get(col, 0.0)
        
        # 絶対値の最大値で並び替え用のキーを作成
        max_abs_corr = max(abs(attrition_val), abs(performance_val), abs(stress_val))
        
        integrated_data.append({
            'column': col,
            'attrition': attrition_val,
            'performance': performance_val,
            'stress': stress_val,
            'max_abs': max_abs_corr
        })
    
    # 最大絶対値でソート（降順）
    integrated_data.sort(key=lambda x: x['max_abs'], reverse=True)
    
    # Markdownファイルの作成
    os.makedirs('doc/fleeting', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'doc/fleeting/correlation_analysis_integrated_{timestamp}.md'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 統合相関分析表 - 離職率・パフォーマンス・ストレス評価\n\n")
        f.write(f"**作成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"**分析対象**: {len(integrated_data)}個の数値カラム\n\n")
        
        f.write("## 概要\n")
        f.write("I社人事データにおける全数値カラムと以下3指標の相関係数を一覧化：\n")
        f.write("- **離職率（Attrition）**: 離職=1, 在職=0\n") 
        f.write("- **パフォーマンス指数（PerformanceIndex）**: 30-100の範囲\n")
        f.write("- **ストレス評価（StressRating）**: ストレス度合い評価\n\n")
        
        f.write("## 完全相関表\n")
        f.write("| 順位 | カラム名 | 離職率 | パフォーマンス | ストレス評価 | 最大絶対値 |\n")
        f.write("|------|----------|:------:|:-------------:|:------------:|:----------:|\n")
        
        for i, data in enumerate(integrated_data, 1):
            # カラム名を適切な長さに調整
            column_name = data['column'].replace('_encoded', '').replace('_', '')[:20]
            f.write(f"| {i:2d} | {column_name:<20} | {data['attrition']:+6.3f} | {data['performance']:+6.3f} | {data['stress']:+6.3f} | {data['max_abs']:6.3f} |\n")
        
        f.write("\n## 相関強度分類\n")
        f.write("- **強**: |r| ≥ 0.15\n")
        f.write("- **中**: 0.10 ≤ |r| < 0.15\n") 
        f.write("- **弱**: 0.05 ≤ |r| < 0.10\n")
        f.write("- **微**: |r| < 0.05\n\n")
        
        # 各指標別の統計
        f.write("## 指標別統計\n\n")
        
        for target_name, corr_dict in [
            ("離職率", attrition_corr),
            ("パフォーマンス指数", performance_corr), 
            ("ストレス評価", stress_corr)
        ]:
            strong = sum(1 for v in corr_dict.values() if abs(v) >= 0.15)
            medium = sum(1 for v in corr_dict.values() if 0.10 <= abs(v) < 0.15)
            weak = sum(1 for v in corr_dict.values() if 0.05 <= abs(v) < 0.10)
            micro = sum(1 for v in corr_dict.values() if abs(v) < 0.05)
            positive = sum(1 for v in corr_dict.values() if v > 0)
            negative = sum(1 for v in corr_dict.values() if v < 0)
            
            f.write(f"### {target_name}\n")
            f.write(f"- 強い相関: {strong}個\n")
            f.write(f"- 中程度相関: {medium}個\n")
            f.write(f"- 弱い相関: {weak}個\n")
            f.write(f"- 微弱相関: {micro}個\n")
            f.write(f"- 正の相関: {positive}個, 負の相関: {negative}個\n\n")
        
        # 注目すべき知見
        f.write("## 注目すべき知見\n\n")
        f.write("### 最も強い相関を持つカラム Top 5\n")
        for i, data in enumerate(integrated_data[:5], 1):
            clean_name = data['column'].replace('_encoded', '').replace('_', ' ')
            f.write(f"{i}. **{clean_name}** (最大絶対値={data['max_abs']:.3f})\n")
            f.write(f"   - 離職率: {data['attrition']:+.3f}\n")
            f.write(f"   - パフォーマンス: {data['performance']:+.3f}\n") 
            f.write(f"   - ストレス: {data['stress']:+.3f}\n\n")
            
        # 多方面影響要因
        f.write("### 複数指標に強い影響を与える要因\n")
        multi_impact = []
        for data in integrated_data:
            strong_count = sum(1 for val in [data['attrition'], data['performance'], data['stress']] 
                             if abs(val) >= 0.10)
            if strong_count >= 2:
                multi_impact.append(data)
        
        if multi_impact:
            for data in multi_impact:
                clean_name = data['column'].replace('_encoded', '').replace('_', ' ')
                f.write(f"- **{clean_name}**: ")
                impacts = []
                if abs(data['attrition']) >= 0.10:
                    impacts.append(f"離職率({data['attrition']:+.3f})")
                if abs(data['performance']) >= 0.10:
                    impacts.append(f"パフォーマンス({data['performance']:+.3f})")
                if abs(data['stress']) >= 0.10:
                    impacts.append(f"ストレス({data['stress']:+.3f})")
                f.write(", ".join(impacts) + "\n")
        else:
            f.write("複数指標に強い影響を与える要因は検出されませんでした。\n")
            
        f.write(f"\n---\n*分析実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}*\n")
    
    print(f"✅ 統合相関分析表を作成しました: {filename}")
    print(f"   📊 分析対象: {len(integrated_data)}個のカラム")
    return filename

def main():
    """メイン処理"""
    print("=== 離職率・パフォーマンス相関分析 ===\n")
    
    # データ読み込み
    df = load_and_prepare_data()
    print(f"データ形状: {df.shape}")
    print(f"離職率: {df['Attrition_numeric'].mean():.3f}")
    print(f"パフォーマンス指数: 平均{df['PerformanceIndex'].mean():.1f} (範囲: {df['PerformanceIndex'].min()}-{df['PerformanceIndex'].max()})\n")
    
    # カテゴリカルデータのエンコード
    df_encoded, categorical_cols = encode_categorical_data(df)
    print(f"エンコードしたカテゴリカルカラム: {len(categorical_cols)}個")
    print(f"カラム名: {categorical_cols}\n")
    
    # 1. 離職率との相関分析
    print("=" * 60)
    print("📈 1. 離職率（Attrition）との相関分析")
    print("=" * 60)
    attrition_correlations = calculate_correlations(df_encoded, 'Attrition_numeric')
    
    # Top 15の詳細表示
    display_correlation_results(attrition_correlations, "離職率")
    
    # 全数値カラムの順位づけ表示
    print("\n" + "-" * 80)
    print("📊 離職率との相関 - 全39カラム完全ランキング")
    print("-" * 80)
    print("順位 | カラム名 | 相関係数 | 分類")
    print("-" * 80)
    
    for i, (col, corr) in enumerate(attrition_correlations.items(), 1):
        # 相関の強さを分類
        if abs(corr) >= 0.15:
            category = "🔴 強"
        elif abs(corr) >= 0.10:
            category = "🟡 中"
        elif abs(corr) >= 0.05:
            category = "🔵 弱"
        else:
            category = "⚪ 微"
            
        print(f"{i:2d}位 | {col:<25} | {corr:7.4f} | {category}")
    
    # 統計サマリーの計算
    strong_corr = sum(1 for _, corr in attrition_correlations.items() if abs(corr) >= 0.15)
    medium_corr = sum(1 for _, corr in attrition_correlations.items() if 0.10 <= abs(corr) < 0.15)
    weak_corr = sum(1 for _, corr in attrition_correlations.items() if 0.05 <= abs(corr) < 0.10)
    micro_corr = sum(1 for _, corr in attrition_correlations.items() if abs(corr) < 0.05)
    
    positive_corr = sum(1 for _, corr in attrition_correlations.items() if corr > 0)
    negative_corr = sum(1 for _, corr in attrition_correlations.items() if corr < 0)
    
    max_corr = max(attrition_correlations.values())
    min_corr = min(attrition_correlations.values())
    avg_corr = sum(attrition_correlations.values()) / len(attrition_correlations)
    
    print(f"\n📊 統計サマリー:")
    print(f"  • 総数値カラム数: {len(attrition_correlations)}個")
    print(f"  • 強い相関(|r|≥0.15): {strong_corr}個 ({strong_corr/len(attrition_correlations)*100:.1f}%)")
    print(f"  • 中程度相関(0.10≤|r|<0.15): {medium_corr}個 ({medium_corr/len(attrition_correlations)*100:.1f}%)")
    print(f"  • 弱い相関(0.05≤|r|<0.10): {weak_corr}個 ({weak_corr/len(attrition_correlations)*100:.1f}%)")
    print(f"  • 微弱相関(|r|<0.05): {micro_corr}個 ({micro_corr/len(attrition_correlations)*100:.1f}%)")
    print(f"  • 正の相関: {positive_corr}個, 負の相関: {negative_corr}個")
    print(f"  • 最大相関: {max_corr:.4f}, 最小相関: {min_corr:.4f}, 平均: {avg_corr:.4f}")
    
    print("\n分類基準: 強(|r|≥0.15), 中(0.10≤|r|<0.15), 弱(0.05≤|r|<0.10), 微(|r|<0.05)")
    
    # Top 15の可視化
    create_correlation_visualization(attrition_correlations, "離職率", 
                                   'src/attrition_correlation_top15.png')
    
    print("\n" + "=" * 60)
    print("🎯 2. パフォーマンス指数（PerformanceIndex）との相関分析")
    print("=" * 60)
    performance_correlations = calculate_correlations(df_encoded, 'PerformanceIndex')
    display_correlation_results(performance_correlations, "パフォーマンス指数")
    create_correlation_visualization(performance_correlations, "パフォーマンス指数", 
                                   'src/performance_correlation_top15.png')
    
    print("\n" + "=" * 60)
    print("😰 3. ストレス評価（StressRating）との相関分析")
    print("=" * 60)
    stress_correlations = calculate_correlations(df_encoded, 'StressRating')
    display_correlation_results(stress_correlations, "ストレス評価")
    create_correlation_visualization(stress_correlations, "ストレス評価", 
                                   'src/stress_correlation_top15.png')
    
    # 4. 統合相関分析表の作成・保存
    print("\n" + "=" * 60)
    print("📋 4. 統合相関分析表の作成・保存")
    print("=" * 60)
    create_integrated_correlation_table(attrition_correlations, performance_correlations, stress_correlations)
    
    # 5. 比較分析
    print("\n" + "=" * 60)
    print("🔍 5. 離職率 vs パフォーマンス - 重要要因比較")
    print("=" * 60)
    
    # 上位10要因を比較
    attrition_top10 = list(attrition_correlations.items())[:10]
    performance_top10 = list(performance_correlations.items())[:10]
    
    print("離職率に影響する要因 Top 10:")
    for i, (col, corr) in enumerate(attrition_top10, 1):
        print(f"  {i:2d}. {col:<25} ({corr:+.3f})")
    
    print("\nパフォーマンスに影響する要因 Top 10:")
    for i, (col, corr) in enumerate(performance_top10, 1):
        print(f"  {i:2d}. {col:<25} ({corr:+.3f})")
    
    # 共通要因の分析
    attrition_factors = set([col for col, _ in attrition_top10])
    performance_factors = set([col for col, _ in performance_top10])
    common_factors = attrition_factors.intersection(performance_factors)
    
    print(f"\n🤝 両方に影響する共通要因 ({len(common_factors)}個):")
    if common_factors:
        for factor in common_factors:
            attrition_corr = attrition_correlations[factor]
            performance_corr = performance_correlations[factor]
            print(f"  • {factor:<25} | 離職率: {attrition_corr:+.3f} | パフォーマンス: {performance_corr:+.3f}")
    else:
        print("  • 共通要因はありません（Top 10範囲内）")
    
    return {
        'attrition_correlations': attrition_correlations,
        'performance_correlations': performance_correlations,
        'stress_correlations': stress_correlations
    }

if __name__ == "__main__":
    results = main() 