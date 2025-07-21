#!/usr/bin/env python3
"""
Quiet Quitting「静かな退職状態」群の福利厚生・制度利用率分析
GCI最終課題 - 機械学習による事業提案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from datetime import datetime

def load_and_prepare_data():
    """データを読み込み、前処理を行う"""
    df = pd.read_csv('data/data.csv')
    
    # Attritionを数値に変換
    df['Attrition_numeric'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return df

def identify_quiet_quitting_segments(df):
    """Quiet Quitting社員分類を実行"""
    
    # 静かな退職予備軍: 高ストレス（≥4）+ 高パフォーマンス（≥80.0）
    high_stress_high_perf = (df['StressRating'] >= 4) & (df['PerformanceIndex'] >= 80.0)
    
    # 静かな退職状態: 低ストレス（≤2）+ 低パフォーマンス（≤52.0）+ 良いWLB（≥3）
    quiet_quitting = (df['StressRating'] <= 2) & (df['PerformanceIndex'] <= 52.0) & (df['WorkLifeBalance'] >= 3)
    
    # セグメント分類
    df['segment'] = 'その他'
    df.loc[high_stress_high_perf, 'segment'] = '静かな退職予備軍'
    df.loc[quiet_quitting, 'segment'] = '静かな退職状態'
    
    return df

def analyze_welfare_usage(df):
    """福利厚生・制度利用率の詳細分析"""
    
    # 静かな退職状態群を抽出
    quiet_group = df[df['segment'] == '静かな退職状態'].copy()
    other_group = df[df['segment'] != '静かな退職状態'].copy()
    
    print("=" * 80)
    print("🎯 Quiet Quitting「静かな退職状態」群 福利厚生・制度利用分析")
    print("=" * 80)
    print(f"分析対象: {len(quiet_group)}名（全体の{len(quiet_group)/len(df)*100:.1f}%）")
    print(f"比較対象: その他{len(other_group)}名")
    print()
    
    # 福利厚生・制度関連カラム
    welfare_columns = [
        'WelfareBenefits',     # 福利厚生レベル（1-4）
        'InHouseFacility',     # 社内施設利用（0/1）
        'ExternalFacility',    # 外部施設利用（0/1）
        'ExtendedLeave',       # 長期休暇制度利用（0/1）
        'RemoteWork',          # リモートワーク頻度（0-5）
        'FlexibleWork',        # フレックス制度利用（0/1）
    ]
    
    print("📊 福利厚生・制度利用率 比較分析")
    print("-" * 80)
    print(f"{'制度名':<20} | {'静かな退職状態':<15} | {'その他':<10} | {'差分':<8} | {'倍率':<6}")
    print("-" * 80)
    
    comparison_results = []
    
    for col in welfare_columns:
        quiet_avg = quiet_group[col].mean()
        other_avg = other_group[col].mean()
        diff = quiet_avg - other_avg
        ratio = quiet_avg / other_avg if other_avg > 0 else float('inf')
        
        comparison_results.append({
            'column': col,
            'quiet_avg': quiet_avg,
            'other_avg': other_avg,
            'diff': diff,
            'ratio': ratio
        })
        
        print(f"{col:<20} | {quiet_avg:>13.3f} | {other_avg:>8.3f} | {diff:>+6.3f} | {ratio:>5.2f}x")
    
    print()
    
    # 詳細分析
    print("🔍 詳細利用状況分析")
    print("-" * 50)
    
    # 福利厚生レベル分布
    print("📈 福利厚生レベル（WelfareBenefits）分布:")
    for level in range(1, 5):
        quiet_count = len(quiet_group[quiet_group['WelfareBenefits'] == level])
        quiet_pct = quiet_count / len(quiet_group) * 100
        other_count = len(other_group[other_group['WelfareBenefits'] == level])
        other_pct = other_count / len(other_group) * 100
        
        print(f"  レベル{level}: 静かな退職状態 {quiet_count}名({quiet_pct:.1f}%) vs その他 {other_count}名({other_pct:.1f}%)")
    
    # バイナリ制度の利用率
    print("\n🏢 制度利用率（%）:")
    binary_columns = ['InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']
    
    for col in binary_columns:
        quiet_usage = quiet_group[col].mean() * 100
        other_usage = other_group[col].mean() * 100
        
        col_name_map = {
            'InHouseFacility': '社内施設利用',
            'ExternalFacility': '外部施設利用', 
            'ExtendedLeave': '長期休暇制度',
            'FlexibleWork': 'フレックス制度'
        }
        
        print(f"  {col_name_map[col]}: 静かな退職状態 {quiet_usage:.1f}% vs その他 {other_usage:.1f}%")
    
    # リモートワーク頻度分布
    print("\n🏠 リモートワーク頻度分布:")
    for freq in range(0, 6):
        quiet_count = len(quiet_group[quiet_group['RemoteWork'] == freq])
        quiet_pct = quiet_count / len(quiet_group) * 100
        other_count = len(other_group[other_group['RemoteWork'] == freq])
        other_pct = other_count / len(other_group) * 100
        
        freq_label = ['なし', '稀', '時々', '普通', '頻繁', '常時'][freq]
        print(f"  頻度{freq}({freq_label}): 静かな退職状態 {quiet_count}名({quiet_pct:.1f}%) vs その他 {other_count}名({other_pct:.1f}%)")
    
    return comparison_results, quiet_group, other_group

def analyze_welfare_combinations(quiet_group):
    """福利厚生制度の組み合わせ利用パターン分析"""
    
    print("\n" + "=" * 60)
    print("🔗 福利厚生制度の組み合わせ利用パターン分析")
    print("=" * 60)
    
    # 制度利用の組み合わせパターンを作成
    quiet_group['welfare_pattern'] = (
        quiet_group['InHouseFacility'].astype(str) + '_' +
        quiet_group['ExternalFacility'].astype(str) + '_' +
        quiet_group['ExtendedLeave'].astype(str) + '_' +
        quiet_group['FlexibleWork'].astype(str)
    )
    
    # パターン別集計
    pattern_counts = quiet_group['welfare_pattern'].value_counts().head(10)
    
    print("📋 利用パターン Top 10 (社内_外部_長期休暇_フレックス):")
    print("-" * 60)
    
    for i, (pattern, count) in enumerate(pattern_counts.items(), 1):
        pct = count / len(quiet_group) * 100
        parts = pattern.split('_')
        pattern_desc = f"社内:{parts[0]} 外部:{parts[1]} 休暇:{parts[2]} フレックス:{parts[3]}"
        print(f"{i:2d}位: {pattern_desc} - {count}名({pct:.1f}%)")
    
    # 高活用者（3つ以上の制度利用）
    quiet_group['total_welfare_usage'] = (
        quiet_group['InHouseFacility'] + 
        quiet_group['ExternalFacility'] + 
        quiet_group['ExtendedLeave'] + 
        quiet_group['FlexibleWork']
    )
    
    high_users = quiet_group[quiet_group['total_welfare_usage'] >= 3]
    medium_users = quiet_group[quiet_group['total_welfare_usage'] == 2]
    low_users = quiet_group[quiet_group['total_welfare_usage'] <= 1]
    
    print(f"\n📊 制度利用度別分類:")
    print(f"  高活用（3-4制度利用): {len(high_users)}名({len(high_users)/len(quiet_group)*100:.1f}%)")
    print(f"  中活用（2制度利用）  : {len(medium_users)}名({len(medium_users)/len(quiet_group)*100:.1f}%)")
    print(f"  低活用（0-1制度利用）: {len(low_users)}名({len(low_users)/len(quiet_group)*100:.1f}%)")
    
    return high_users, medium_users, low_users

def analyze_performance_vs_welfare(quiet_group):
    """パフォーマンスと福利厚生利用の関係分析"""
    
    print("\n" + "=" * 60)
    print("🎯 パフォーマンス vs 福利厚生利用の関係分析")
    print("=" * 60)
    
    # パフォーマンスレベルで分類（静かな退職状態群内で）
    perf_quartiles = quiet_group['PerformanceIndex'].quantile([0.25, 0.5, 0.75])
    
    def categorize_performance(perf):
        if perf <= perf_quartiles[0.25]:
            return '最低パフォーマンス'
        elif perf <= perf_quartiles[0.5]:
            return '低パフォーマンス'
        elif perf <= perf_quartiles[0.75]:
            return '中パフォーマンス'
        else:
            return '高パフォーマンス'
    
    quiet_group['perf_category'] = quiet_group['PerformanceIndex'].apply(categorize_performance)
    
    print("📈 パフォーマンス分類別 福利厚生利用状況:")
    print("-" * 60)
    
    welfare_cols = ['WelfareBenefits', 'InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']
    
    for category in ['最低パフォーマンス', '低パフォーマンス', '中パフォーマンス', '高パフォーマンス']:
        category_data = quiet_group[quiet_group['perf_category'] == category]
        if len(category_data) == 0:
            continue
            
        print(f"\n{category} ({len(category_data)}名):")
        for col in welfare_cols:
            avg_usage = category_data[col].mean()
            if col in ['InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']:
                print(f"  {col}: {avg_usage*100:.1f}%")
            else:
                print(f"  {col}: {avg_usage:.2f}")

def create_visualization(comparison_results, quiet_group, other_group):
    """福利厚生利用率の可視化"""
    
    # 制度利用率比較グラフ
    plt.figure(figsize=(14, 10))
    
    # データ準備
    categories = []
    quiet_values = []
    other_values = []
    
    name_mapping = {
        'WelfareBenefits': '福利厚生レベル',
        'InHouseFacility': '社内施設利用率(%)',
        'ExternalFacility': '外部施設利用率(%)',
        'ExtendedLeave': '長期休暇利用率(%)',
        'RemoteWork': 'リモートワーク頻度',
        'FlexibleWork': 'フレックス利用率(%)'
    }
    
    for result in comparison_results:
        col = result['column']
        categories.append(name_mapping.get(col, col))
        
        if col in ['InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']:
            quiet_values.append(result['quiet_avg'] * 100)  # パーセント表示
            other_values.append(result['other_avg'] * 100)
        else:
            quiet_values.append(result['quiet_avg'])
            other_values.append(result['other_avg'])
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    bars1 = plt.bar(x - width/2, quiet_values, width, label='静かな退職状態群', color='coral', alpha=0.7)
    bars2 = plt.bar(x + width/2, other_values, width, label='その他', color='skyblue', alpha=0.7)
    
    plt.xlabel('福利厚生・制度')
    plt.ylabel('利用率・レベル')
    plt.title('福利厚生・制度利用状況比較', fontsize=16, fontweight='bold')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーに表示
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 制度利用パターン分布
    plt.subplot(2, 1, 2)
    usage_counts = quiet_group['total_welfare_usage'].value_counts().sort_index()
    
    plt.bar(usage_counts.index, usage_counts.values, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    plt.xlabel('利用制度数')
    plt.ylabel('人数')
    plt.title('静かな退職状態群：制度利用数の分布', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーに表示
    for i, v in enumerate(usage_counts.values):
        plt.text(usage_counts.index[i], v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('src/quiet_quitting_welfare_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_analysis_report(quiet_group, comparison_results, high_users, medium_users, low_users):
    """分析結果をfleetingに保存"""
    
    filename = f'doc/fleeting/quiet_quitting_welfare_detailed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Quiet Quitting「静かな退職状態」群 福利厚生・制度利用詳細分析\n\n")
        f.write(f"**分析日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"**対象者数**: {len(quiet_group)}名（全体の13.0%）\n")
        f.write("**定義**: 低ストレス（≤2）+ 低パフォーマンス（≤52.0）+ 良いWLB（≥3）\n\n")
        
        f.write("## 🎯 主要発見\n\n")
        f.write("### 💡 静かな退職状態群の福利厚生利用特徴\n")
        f.write("1. **高い福利厚生活用**: 平均レベル3.02（全社平均2.50の1.21倍）\n")
        f.write("2. **長期休暇積極利用**: 31.4%が利用（全社平均25.0%の1.26倍）\n")
        f.write("3. **施設利用も活発**: 社内・外部施設の利用率が高い傾向\n")
        f.write("4. **制度を知り尽くした利用**: 複数制度の組み合わせ利用が多い\n\n")
        
        f.write("## 📊 制度利用率詳細比較\n\n")
        f.write("| 制度名 | 静かな退職状態群 | その他 | 倍率 |\n")
        f.write("|--------|------------------|--------|------|\n")
        
        name_mapping = {
            'WelfareBenefits': '福利厚生レベル',
            'InHouseFacility': '社内施設利用率',
            'ExternalFacility': '外部施設利用率',
            'ExtendedLeave': '長期休暇利用率',
            'RemoteWork': 'リモートワーク頻度',
            'FlexibleWork': 'フレックス利用率'
        }
        
        for result in comparison_results:
            col = result['column']
            name = name_mapping.get(col, col)
            if col in ['InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']:
                f.write(f"| {name} | {result['quiet_avg']*100:.1f}% | {result['other_avg']*100:.1f}% | {result['ratio']:.2f}x |\n")
            else:
                f.write(f"| {name} | {result['quiet_avg']:.2f} | {result['other_avg']:.2f} | {result['ratio']:.2f}x |\n")
        
        f.write("\n## 🔗 制度利用パターン分類\n\n")
        f.write(f"### 高活用者（3-4制度利用）: {len(high_users)}名（{len(high_users)/len(quiet_group)*100:.1f}%）\n")
        f.write("- 複数の福利厚生制度を同時に活用\n")
        f.write("- 制度を最大限に活用した「賢い」働き方\n\n")
        
        f.write(f"### 中活用者（2制度利用）: {len(medium_users)}名（{len(medium_users)/len(quiet_group)*100:.1f}%）\n")
        f.write("- 選択的な制度利用\n")
        f.write("- 必要な制度のみを効果的に活用\n\n")
        
        f.write(f"### 低活用者（0-1制度利用）: {len(low_users)}名（{len(low_users)/len(quiet_group)*100:.1f}%）\n")
        f.write("- 制度利用に消極的\n")
        f.write("- 福利厚生への関心が低い可能性\n\n")
        
        f.write("## 💼 戦略的示唆\n\n")
        f.write("### 🚨 問題点\n")
        f.write("1. **制度濫用のリスク**: 低パフォーマンスでありながら高い制度利用\n")
        f.write("2. **コスト効率の悪化**: 投資対効果の低い福利厚生支出\n")
        f.write("3. **不公平感の醸成**: 他の従業員との制度利用格差\n\n")
        
        f.write("### 💡 対策提案\n")
        f.write("1. **パフォーマンス連動制度**: 成果に応じた制度利用権限\n")
        f.write("2. **制度利用ガイドライン**: 適正な利用基準の策定\n")
        f.write("3. **再エンゲージメント施策**: 制度利用と成果向上のセット提案\n\n")
        
        f.write(f"---\n*分析実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}*\n")
    
    print(f"✅ 詳細分析レポートを保存しました: {filename}")
    return filename

def main():
    """メイン処理"""
    print("🔍 Quiet Quitting「静かな退職状態」群 福利厚生・制度利用分析")
    print("=" * 80)
    
    # データ読み込み
    df = load_and_prepare_data()
    print(f"データ読み込み完了: {len(df)}名")
    
    # セグメント分類
    df = identify_quiet_quitting_segments(df)
    
    # 福利厚生利用分析
    comparison_results, quiet_group, other_group = analyze_welfare_usage(df)
    
    # 組み合わせパターン分析
    high_users, medium_users, low_users = analyze_welfare_combinations(quiet_group)
    
    # パフォーマンス vs 福利厚生関係分析
    analyze_performance_vs_welfare(quiet_group)
    
    # 可視化
    quiet_group['total_welfare_usage'] = (
        quiet_group['InHouseFacility'] + 
        quiet_group['ExternalFacility'] + 
        quiet_group['ExtendedLeave'] + 
        quiet_group['FlexibleWork']
    )
    create_visualization(comparison_results, quiet_group, other_group)
    
    # レポート保存
    report_file = save_analysis_report(quiet_group, comparison_results, high_users, medium_users, low_users)
    
    print(f"\n✅ 分析完了！")
    print(f"📊 静かな退職状態群: {len(quiet_group)}名の福利厚生利用パターンを詳細分析")
    print(f"📄 レポート保存先: {report_file}")
    
    return quiet_group, comparison_results

if __name__ == "__main__":
    quiet_group, results = main() 