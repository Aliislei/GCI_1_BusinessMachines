#!/usr/bin/env python3
"""
静かな退職群 vs その他 福利厚生制度利用率可視化
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
    
    # 静かな退職状態: 低ストレス（≤2）+ 低パフォーマンス（≤52.0）+ 良いWLB（≥3）
    quiet_quitting = (df['StressRating'] <= 2) & (df['PerformanceIndex'] <= 52.0) & (df['WorkLifeBalance'] >= 3)
    
    # セグメント分類
    df['segment'] = 'その他'
    df.loc[quiet_quitting, 'segment'] = '静かな退職状態'
    
    return df

def create_comprehensive_welfare_visualization(df):
    """包括的な福利厚生利用率可視化"""
    
    # セグメント別データ準備
    quiet_group = df[df['segment'] == '静かな退職状態'].copy()
    other_group = df[df['segment'] == 'その他'].copy()
    
    print(f"📊 可視化対象:")
    print(f"  静かな退職状態: {len(quiet_group)}名（{len(quiet_group)/len(df)*100:.1f}%）")
    print(f"  その他: {len(other_group)}名（{len(other_group)/len(df)*100:.1f}%）")
    
    # 大きなキャンバスを作成
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 制度利用率比較 (バイナリ制度)
    plt.subplot(2, 3, 1)
    binary_welfare = {
        '社内施設利用': ['InHouseFacility'],
        '外部施設利用': ['ExternalFacility'], 
        '長期休暇制度': ['ExtendedLeave'],
        'フレックス制度': ['FlexibleWork']
    }
    
    categories = []
    quiet_rates = []
    other_rates = []
    
    for name, cols in binary_welfare.items():
        col = cols[0]
        quiet_rate = quiet_group[col].mean() * 100
        other_rate = other_group[col].mean() * 100
        
        categories.append(name)
        quiet_rates.append(quiet_rate)
        other_rates.append(other_rate)
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, quiet_rates, width, label='静かな退職状態', color='coral', alpha=0.8)
    bars2 = plt.bar(x + width/2, other_rates, width, label='その他', color='skyblue', alpha=0.8)
    
    plt.xlabel('制度')
    plt.ylabel('利用率 (%)')
    plt.title('制度利用率比較', fontsize=14, fontweight='bold')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーに表示
    for bar, rate in zip(bars1, quiet_rates):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar, rate in zip(bars2, other_rates):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. 福利厚生レベル分布
    plt.subplot(2, 3, 2)
    welfare_levels = [1, 2, 3, 4]
    quiet_level_dist = [len(quiet_group[quiet_group['WelfareBenefits'] == level])/len(quiet_group)*100 
                       for level in welfare_levels]
    other_level_dist = [len(other_group[other_group['WelfareBenefits'] == level])/len(other_group)*100 
                       for level in welfare_levels]
    
    x = np.arange(len(welfare_levels))
    bars1 = plt.bar(x - width/2, quiet_level_dist, width, label='静かな退職状態', color='coral', alpha=0.8)
    bars2 = plt.bar(x + width/2, other_level_dist, width, label='その他', color='skyblue', alpha=0.8)
    
    plt.xlabel('福利厚生レベル')
    plt.ylabel('割合 (%)')
    plt.title('福利厚生レベル分布', fontsize=14, fontweight='bold')
    plt.xticks(x, [f'レベル{l}' for l in welfare_levels])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーに表示
    for bar, rate in zip(bars1, quiet_level_dist):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar, rate in zip(bars2, other_level_dist):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. リモートワーク頻度分布
    plt.subplot(2, 3, 3)
    remote_freq = [0, 1, 2, 3, 4, 5]
    remote_labels = ['なし', '稀', '時々', '普通', '頻繁', '常時']
    
    quiet_remote_dist = [len(quiet_group[quiet_group['RemoteWork'] == freq])/len(quiet_group)*100 
                        for freq in remote_freq]
    other_remote_dist = [len(other_group[other_group['RemoteWork'] == freq])/len(other_group)*100 
                        for freq in remote_freq]
    
    plt.plot(remote_labels, quiet_remote_dist, 'o-', label='静かな退職状態', 
             color='coral', linewidth=2, markersize=6)
    plt.plot(remote_labels, other_remote_dist, 's-', label='その他', 
             color='skyblue', linewidth=2, markersize=6)
    
    plt.xlabel('リモートワーク頻度')
    plt.ylabel('割合 (%)')
    plt.title('リモートワーク頻度分布', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 制度利用格差（倍率）
    plt.subplot(2, 3, 4)
    ratios = []
    ratio_labels = []
    
    for name, cols in binary_welfare.items():
        col = cols[0]
        quiet_rate = quiet_group[col].mean()
        other_rate = other_group[col].mean()
        ratio = quiet_rate / other_rate if other_rate > 0 else 0
        
        ratios.append(ratio)
        ratio_labels.append(name)
    
    # 福利厚生レベルも追加
    quiet_welfare_avg = quiet_group['WelfareBenefits'].mean()
    other_welfare_avg = other_group['WelfareBenefits'].mean()
    welfare_ratio = quiet_welfare_avg / other_welfare_avg
    
    ratios.append(welfare_ratio)
    ratio_labels.append('福利厚生レベル')
    
    colors = ['red' if r > 1.2 else 'orange' if r > 1.1 else 'green' for r in ratios]
    bars = plt.bar(ratio_labels, ratios, color=colors, alpha=0.7)
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='同等利用')
    plt.xlabel('制度')
    plt.ylabel('利用率倍率（静かな退職状態 / その他）')
    plt.title('制度利用格差（倍率）', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーに表示
    for bar, ratio in zip(bars, ratios):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{ratio:.2f}x', ha='center', va='bottom', fontsize=8)
    
    # 5. 制度組み合わせ利用パターン
    plt.subplot(2, 3, 5)
    
    # 利用制度数を計算
    quiet_group['total_usage'] = (quiet_group['InHouseFacility'] + 
                                 quiet_group['ExternalFacility'] + 
                                 quiet_group['ExtendedLeave'] + 
                                 quiet_group['FlexibleWork'])
    
    other_group['total_usage'] = (other_group['InHouseFacility'] + 
                                 other_group['ExternalFacility'] + 
                                 other_group['ExtendedLeave'] + 
                                 other_group['FlexibleWork'])
    
    usage_levels = [0, 1, 2, 3, 4]
    quiet_usage_dist = [len(quiet_group[quiet_group['total_usage'] == level])/len(quiet_group)*100 
                       for level in usage_levels]
    other_usage_dist = [len(other_group[other_group['total_usage'] == level])/len(other_group)*100 
                       for level in usage_levels]
    
    x = np.arange(len(usage_levels))
    bars1 = plt.bar(x - width/2, quiet_usage_dist, width, label='静かな退職状態', color='coral', alpha=0.8)
    bars2 = plt.bar(x + width/2, other_usage_dist, width, label='その他', color='skyblue', alpha=0.8)
    
    plt.xlabel('利用制度数')
    plt.ylabel('割合 (%)')
    plt.title('制度組み合わせ利用パターン', fontsize=14, fontweight='bold')
    plt.xticks(x, [f'{l}制度' for l in usage_levels])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 値をバーに表示
    for bar, rate in zip(bars1, quiet_usage_dist):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar, rate in zip(bars2, other_usage_dist):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 6. 統計サマリー表
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # サマリーデータ準備
    summary_data = []
    
    # 基本統計
    summary_data.append(['人数', f'{len(quiet_group)}名', f'{len(other_group)}名'])
    summary_data.append(['割合', f'{len(quiet_group)/len(df)*100:.1f}%', f'{len(other_group)/len(df)*100:.1f}%'])
    summary_data.append(['', '', ''])
    
    # 制度利用率
    for name, cols in binary_welfare.items():
        col = cols[0]
        quiet_rate = quiet_group[col].mean() * 100
        other_rate = other_group[col].mean() * 100
        summary_data.append([name, f'{quiet_rate:.1f}%', f'{other_rate:.1f}%'])
    
    summary_data.append(['', '', ''])
    summary_data.append(['福利厚生レベル(平均)', f'{quiet_welfare_avg:.2f}', f'{other_welfare_avg:.2f}'])
    summary_data.append(['高活用者(3-4制度)', 
                        f'{len(quiet_group[quiet_group["total_usage"] >= 3])/len(quiet_group)*100:.1f}%',
                        f'{len(other_group[other_group["total_usage"] >= 3])/len(other_group)*100:.1f}%'])
    
    # 表を作成
    table = plt.table(cellText=summary_data,
                     colLabels=['指標', '静かな退職状態', 'その他'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # ヘッダーの色付け
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 静かな退職状態列の色付け
    for i in range(1, len(summary_data) + 1):
        table[(i, 1)].set_facecolor('#FFE5E5')
    
    plt.title('統計サマリー', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('src/welfare_usage_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return quiet_group, other_group

def create_detailed_heatmap(quiet_group, other_group):
    """詳細なヒートマップ可視化"""
    
    plt.figure(figsize=(12, 8))
    
    # データ準備
    welfare_metrics = {
        '社内施設利用': 'InHouseFacility',
        '外部施設利用': 'ExternalFacility',
        '長期休暇制度': 'ExtendedLeave', 
        'フレックス制度': 'FlexibleWork',
        '福利厚生レベル': 'WelfareBenefits',
        'リモートワーク頻度': 'RemoteWork'
    }
    
    # 部署別・セグメント別の利用率計算
    departments = ['Research & Development', 'Sales', 'Human Resources']
    
    heatmap_data = []
    row_labels = []
    
    for dept in departments:
        for segment in ['静かな退職状態', 'その他']:
            if segment == '静かな退職状態':
                dept_data = quiet_group[quiet_group['Department'] == dept]
            else:
                dept_data = other_group[other_group['Department'] == dept]
            
            if len(dept_data) == 0:
                row_data = [0] * len(welfare_metrics)
            else:
                row_data = []
                for metric, col in welfare_metrics.items():
                    if col in ['InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']:
                        value = dept_data[col].mean() * 100  # パーセント
                    else:
                        value = dept_data[col].mean()  # 実数値
                    row_data.append(value)
            
            heatmap_data.append(row_data)
            row_labels.append(f'{dept}\n({segment})')
    
    # ヒートマップ作成
    heatmap_array = np.array(heatmap_data)
    
    sns.heatmap(heatmap_array, 
                xticklabels=list(welfare_metrics.keys()),
                yticklabels=row_labels,
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                center=50,
                square=True,
                linewidths=0.5)
    
    plt.title('部署別・セグメント別 福利厚生利用状況ヒートマップ', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('福利厚生制度')
    plt.ylabel('部署・セグメント')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('src/welfare_usage_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(quiet_group, other_group):
    """サマリー統計の出力"""
    
    print("\n" + "=" * 80)
    print("📊 福利厚生制度利用率 - 詳細統計サマリー")
    print("=" * 80)
    
    welfare_columns = {
        'WelfareBenefits': '福利厚生レベル',
        'InHouseFacility': '社内施設利用率',
        'ExternalFacility': '外部施設利用率',
        'ExtendedLeave': '長期休暇制度利用率',
        'RemoteWork': 'リモートワーク頻度',
        'FlexibleWork': 'フレックス制度利用率'
    }
    
    print(f"{'制度名':<20} | {'静かな退職状態':<15} | {'その他':<10} | {'格差':<8} | {'倍率':<6}")
    print("-" * 70)
    
    for col, name in welfare_columns.items():
        quiet_avg = quiet_group[col].mean()
        other_avg = other_group[col].mean()
        diff = quiet_avg - other_avg
        ratio = quiet_avg / other_avg if other_avg > 0 else 0
        
        if col in ['InHouseFacility', 'ExternalFacility', 'ExtendedLeave', 'FlexibleWork']:
            print(f"{name:<20} | {quiet_avg*100:>13.1f}% | {other_avg*100:>8.1f}% | {diff*100:>+6.1f}% | {ratio:>5.2f}x")
        else:
            print(f"{name:<20} | {quiet_avg:>13.2f}  | {other_avg:>8.2f}  | {diff:>+6.2f}  | {ratio:>5.2f}x")

def main():
    """メイン処理"""
    print("🎨 静かな退職群 vs その他 福利厚生制度利用率可視化")
    print("=" * 80)
    
    # データ読み込み
    df = load_and_prepare_data()
    df = identify_quiet_quitting_segments(df)
    
    # 包括的可視化
    quiet_group, other_group = create_comprehensive_welfare_visualization(df)
    
    # ヒートマップ可視化
    create_detailed_heatmap(quiet_group, other_group)
    
    # 統計サマリー出力
    print_summary_statistics(quiet_group, other_group)
    
    print(f"\n✅ 可視化完了!")
    print(f"📊 生成ファイル:")
    print(f"  - src/welfare_usage_comprehensive_comparison.png (包括的比較)")
    print(f"  - src/welfare_usage_heatmap.png (部署別ヒートマップ)")

if __name__ == "__main__":
    main() 