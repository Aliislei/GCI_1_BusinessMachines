# 通勤時間特徴量エンジニアリング案

## 概要
リモートワーク頻度を考慮した実際の通勤時間計算と、距離データの時間換算による新しい特徴量の作成案

## 1. リモートワーク考慮通勤時間

### 基本式
```
実際の通勤時間 = (5 - RemoteWork) × 通勤時間（片道）
```

### 計算ロジック
- RemoteWorkは0-5の値（0:なし → 5:頻繁）
- 週5日勤務を前提として、リモートワーク日数分を差し引く
- 通勤が発生するのは出社日のみのため、実際の通勤負荷を算出

### 期待効果
- より現実的な通勤負荷の測定
- ワークライフバランスや離職意向との関係性が明確化
- リモートワーク制度の効果測定が可能

## 2. DistanceFromHome時間換算

### 距離別時間換算ルール

#### 10km以内：近距離通勤
- **移動手段**: 徒歩・自転車・バス
- **片道時間**: 0.5時間
- **往復時間**: 1.0時間

#### 10-30km：中距離通勤  
- **移動手段**: 電車
- **片道時間**: 1.0時間
- **往復時間**: 2.0時間

#### 30km以上：長距離通勤
- **移動手段**: 自家用車
- **計算式**: 距離 ÷ 時速40km/h × 2（往復）
- **例**: 40km → 40÷40×2 = 2.0時間

### 実装案
```python
def convert_distance_to_commute_time(distance):
    if distance <= 10:
        return 1.0  # 往復1時間
    elif distance <= 30:
        return 2.0  # 往復2時間
    else:
        return (distance / 40) * 2  # 往復時間
```

## 3. 週間業務拘束時間の算出

### 前提条件
- **OverTime**: 月残業時間（時間単位）
- **StandardHours**: 40時間/週（基本労働時間）
- **週間通勤時間**: リモートワーク日数を考慮した実際の通勤時間

### 計算式
```
週間業務拘束時間 = 週間通勤時間 + 40h + (OverTime ÷ 4)
```

### 詳細計算ロジック
```python
# 1. 距離から往復通勤時間を算出
daily_commute_time = convert_distance_to_commute_time(distance_from_home)

# 2. リモートワーク頻度を考慮した週間通勤時間
weekly_commute_time = (5 - remote_work_freq) * daily_commute_time

# 3. 月残業時間を週平均に変換
weekly_overtime = overtime_monthly / 4

# 4. 週間業務拘束時間の算出
weekly_total_constraint = weekly_commute_time + 40 + weekly_overtime
```

### 成分内訳
1. **週間通勤時間**: 実際の出社日のみの通勤負荷
2. **40時間**: 標準労働時間（週5日×8時間）
3. **OverTime/4**: 月残業時間の週平均

### 最終特徴量
```python
# 実際の週間通勤時間
actual_weekly_commute = (5 - remote_work_freq) * daily_commute_time

# 週間業務拘束時間（総合指標）
weekly_business_constraint = actual_weekly_commute + 40 + (overtime_monthly / 4)
```

## 期待される分析効果

1. **より現実的な通勤負荷測定**
   - リモートワーク制度の恩恵を定量化
   - 真の通勤ストレスを把握

2. **離職予測精度向上**
   - 通勤負荷と離職意向の関係性を正確に捉える
   - ワークライフバランス要因の精緻化

3. **政策提案の根拠強化**
   - リモートワーク拡大による効果の定量評価
   - 通勤手当制度の見直し提案

4. **総合的な業務負荷評価**
   - 通勤・労働・残業を統合した拘束時間指標
   - ワークライフバランス改善施策の優先順位づけ

---
*記録日: 2025年7月*  
*分類: 特徴量エンジニアリング* 