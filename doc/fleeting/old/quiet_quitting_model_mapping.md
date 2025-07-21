# Quiet Quitting Model - I社データセット指標マッピング

**作成日**: 2025年7月  
**参照**: The Quiet Quitting Model – Causal Loop Diagram  
**データ**: I社人事データセット（1,470名、44カラム）

## 理論モデル要素と実データ指標の対応関係

### 📊 **コア指標マッピング**

| 理論モデル要素 | I社データ指標 | カラム名 | 値の範囲 | 備考 |
|----------------|---------------|----------|----------|------|
| **Disengagement** | ストレス評価 | `StressRating` | 1-5 (1:低→5:高) | 高値=高ディスエンゲージメント |
| **Actual Performance** | 実パフォーマンス指数 | `PerformanceIndex` | 30-100 | 客観的成果指標 |
| **Work-Life Balance** | ワークライフバランス | `WorkLifeBalance` | 1-4 (1:悪→4:良) | 直接対応指標 |
| **Dissatisfaction** | 職場環境満足度 | `EnvironmentSatisfaction` | 1-4 (1:低→4:高) | 逆スケール：低値=高不満 |
| **Dissatisfaction** | 職務満足度 | `JobSatisfaction` | 1-4 (1:低→4:高) | 逆スケール：低値=高不満 |

### 🔍 **補完指標マッピング**

| 理論モデル要素 | I社データ指標 | カラム名 | 値の範囲 | 備考 |
|----------------|---------------|----------|----------|------|
| **Performance Gap** | 期待パフォーマンスとの乖離 | `PerformanceRating`（逆スケール） | 1-5 (1:高Gap→5:低Gap) | 低評価=高Performance Gap |
| **Job Involvement** | 職務関与度 | `JobInvolvement` | 1-4 (1:低→4:高) | ※理論と実態が乖離 |
| **Stress (Self-Reported)** | 自己申告ストレス | `StressSelfReported` | 1-4 (1:低→4:高) | Disengagementの補助指標 |
| **Relationship Satisfaction** | 人間関係満足度 | `RelationshipSatisfaction` | 1-4 (1:低→4:高) | Dissatisfactionの補助指標 |

### 🎯 **モデル未対応要素（データなし）**

| 理論モデル要素 | 説明 | I社データでの代替可能性 |
|----------------|------|-------------------------|
| **Job Creep** | 職務範囲の拡大 | データなし（質的調査が必要） |
| **Job Norms and Expectations** | 職務規範・期待値 | データなし（組織文化調査が必要） |
| **Expected Performance** | 期待パフォーマンス | Performance Gap経由で間接測定可能 |
| **Citizenship Fatigue** | 組織市民行動の疲労 | データなし（行動観察が必要） |
| **Citizenship Crafting** | 組織市民行動の調整 | データなし（行動分析が必要） |
| **Sense of Belonging** | 帰属意識 | `RelationshipSatisfaction`で一部代替 |
| **Quiet Quitting** | 静かな退職行動 | `Attrition`で一部測定（結果のみ） |

## 分析上の重要な注意点

### ⚠️ **指標解釈の注意事項**

1. **逆スケール指標**:
   - 満足度系指標（低値=高不満=モデルの負の状態）
   - ワークライフバランス（低値=悪い状態）

2. **スケール正規化**:
   - PerformanceIndex（30-100）vs 他指標（1-4/1-5）
   - 比較時は正規化が必要

3. **理論との乖離**:
   - JobInvolvementは理論上のEngagementと乖離
   - 実際は「過労リスク指標」として機能

4. **Performance Gap**:
   - 企業期待パフォーマンスとの乖離を表す
   - PerformanceRatingの低さが高Performance Gapを示唆
   - 実パフォーマンス(PerformanceIndex)との差ではない点に注意

### 📈 **複合指標の構築可能性**

| 複合指標名 | 構成要素 | 計算式案 |
|------------|----------|----------|
| **総合不満度** | EnvironmentSatisfaction + JobSatisfaction | (8 - ES - JS) / 6 |
| **ディスエンゲージメント度** | StressRating + StressSelfReported | (SR + SSR - 2) / 7 |
| **パフォーマンスギャップ** | PerformanceRating（期待未達度） | (6 - PR) / 5 （逆スケール正規化） |
| **ワークライフバランス逆転** | WorkLifeBalance | (5 - WLB) / 4 |

## 実証分析への適用

### 🔄 **因果関係検証のための指標設定**

1. **Job Creep → Performance Gap**: データ制約により検証困難
2. **Performance Gap → Dissatisfaction**: PerformanceRating低下 vs 満足度指標低下
3. **Dissatisfaction → Disengagement**: 満足度指標 vs StressRating
4. **Disengagement → Work-Life Balance**: StressRating vs WorkLifeBalance ✓
5. **Work-Life Balance → Quiet Quitting**: WorkLifeBalance vs Attrition ✓

### ✅ **検証可能な仮説**

- **H1**: StressRating ↔ WorkLifeBalance（負の相関）
- **H2**: 満足度低下 → StressRating上昇
- **H3**: WorkLifeBalance悪化 → Attrition増加
- **H4**: PerformanceRating低下（Performance Gap増大）と満足度低下の関係

---
*このマッピングにより、Quiet Quittingモデルの一部要素について定量的検証が可能。ただし、Job CreepやCitizenship系要素は質的調査が必要。* 