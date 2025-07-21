# 静かな退職防止・組織パフォーマンス最大化施策の事業提案アイデア

## プロジェクト概要
- **提案テーマ**: 静かな退職を防ぎ組織のパフォーマンスを最大化するための施策
- **参考資料**: 静かな退職の因果ループ図（システム思考による構造分析）
- **データセット**: IBM HR Analytics Employee Attrition & Performance（1,471名の従業員データ）

## 因果ループ図から見える重要な要因
1. **Job Creep（業務拡大）** → Job Norms and Expectations（業務規範と期待）
2. **Performance Gap（パフォーマンスギャップ）** = Expected Performance - Actual Performance
3. **Work-Life Balance（ワークライフバランス）** ← Performance Gap
4. **Dissatisfaction（不満）** ← Performance Gap + Work-Life Balance
5. **Disengagement（非従事）** ← Dissatisfaction
6. **Quiet Quitting（静かな退職）** ← Disengagement
7. **Citizenship Fatigue（市民性疲労）** → Citizenship Crafting → Sense of Belonging

## データセット分析対象カラム（計44カラム）

### 【核心指標】離職・退職関連
- **Attrition**: 離職の有無（Yes/No）→ **主要な目的変数候補**

### 【パフォーマンス関連】（因果ループの中核）
- **PerformanceIndex**: パフォーマンス指数（数値）
- **PerformanceRating**: パフォーマンス評価（1-4段階）
- **MonthlyAchievement**: 月次成果（数値）

### 【満足度・エンゲージメント関連】（Dissatisfaction/Disengagement測定）
- **JobSatisfaction**: 職務満足度（1-4段階）
- **EnvironmentSatisfaction**: 環境満足度（1-4段階）
- **JobInvolvement**: 職務関与度（1-4段階）
- **RelationshipSatisfaction**: 人間関係満足度（1-4段階）

### 【ワークライフバランス関連】（因果ループの重要な中間変数）
- **WorkLifeBalance**: ワークライフバランス（1-4段階）
- **OverTime**: 残業時間（数値）
- **StressRating**: ストレス評価（1-5段階）
- **StressSelfReported**: 自己申告ストレス（1-5段階）

### 【職務・役割関連】（Job Creep/Job Norms関連）
- **JobRole**: 職務役割（カテゴリ）
- **JobLevel**: 職務レベル（1-5段階）
- **BusinessTravel**: 出張頻度（カテゴリ）

### 【組織サポート・福利厚生関連】（Sense of Belonging影響要因）
- **WelfareBenefits**: 福利厚生（数値）
- **RemoteWork**: リモートワーク（0-5段階）
- **FlexibleWork**: 柔軟な働き方（0/1）
- **InHouseFacility**: 社内施設（0/1）
- **ExternalFacility**: 外部施設（0/1）
- **ExtendedLeave**: 長期休暇（0/1）

### 【キャリア・経験関連】（期待とパフォーマンスギャップの背景）
- **TotalWorkingYears**: 総勤務年数
- **YearsAtCompany**: 在籍年数
- **YearsInCurrentRole**: 現職務での年数
- **YearsSinceLastPromotion**: 昇進からの年数
- **YearsWithCurrManager**: 現マネージャーとの期間

### 【基本属性】（セグメント分析用）
- **Age**: 年齢
- **Gender**: 性別
- **Education**: 教育レベル（1-5段階）
- **MaritalStatus**: 婚姻状況
- **MonthlyIncome**: 月収

## 分析仮説・アプローチ案

### 1. 因果ループ理論に基づく静かな退職予測モデル
**理論的基盤**: Zieba (2023) R1強化ループ・B1/B2バランシングループ
- **目的変数**: Attrition（離職）+ Job Involvement（静かな退職代理指標）
- **R1ループ変数**: Performance Gap → Dissatisfaction → Disengagement → Actual Performance
- **B1ループ変数**: Job Creep → Expected Performance ⇄ Performance Gap
- **B2ループ変数**: Disengagement → Work-Life Balance → Dissatisfaction
- **手法**: 構造方程式モデリング（SEM）、Random Forest、XGBoost等で因果関係検証

### 2. Performance Gap構造分析（B1ループ検証）
- **Performance Gap指標**: (Expected Performance - Actual Performance) 
  - Expected Performance代理指標: JobLevel × YearsSinceLastPromotion
  - Actual Performance: PerformanceIndex, PerformanceRating
- **Job Creep測定**: OverTime × JobInvolvement × YearsInCurrentRole相互作用
- **期待値調整効果**: マネージャー関係年数（YearsWithCurrManager）の調整効果

### 3. 帰属意識・エンゲージメント因果分析（R1ループ検証）
- **Sense of Belonging構成要素**: RelationshipSatisfaction, EnvironmentSatisfaction
- **Dissatisfaction複合指標**: 各満足度指標の主成分分析
- **Disengagement測定**: JobInvolvement逆転 + 低パフォーマンス組み合わせ
- **調整変数**: RemoteWork, FlexibleWorkの孤立感緩和効果

### 4. Citizenship Fatigue・組織サポート効果（B2ループ検証）  
- **Citizenship活動負荷**: BusinessTravel + OverTime + 福利厚生未活用度
- **個人動機セグメント**: クラスター分析による動機タイプ分類
- **Work-Life Balance改善効果**: StressRating減少 × WelfareBenefits活用度
- **組織信頼指標**: 上司関係満足度 × 意思決定参加度（JobLevel）

## 事業提案の方向性

### 施策案1: パフォーマンスギャップ・業務過重防止システム
**学術的根拠**: Zieba (2023) B1ループ理論 - Job Creep対策とExpected Performance調整
- **機械学習による離職リスク予測**: 満足度低下の早期発見
- **Job Creep検知システム**: 業務拡大の自動監視・警告機能
- **期待値動的調整メカニズム**: 実績パフォーマンスとのギャップ分析による期待値再調整
- **持続可能業務負荷算定**: 個人能力と業務責任の最適マッチング

### 施策案2: 帰属意識向上・個別最適化プログラム  
**学術的根拠**: Sense of Belonging理論 - 多様性を考慮した個別アプローチ
- **従業員セグメント別サポート**: 属性・職務・満足度パターン別の個別化支援
- **安全な経験共有プラットフォーム**: 問題・懸念の安心な共有環境構築
- **全方位投資戦略**: キー人材偏重を避けた全従業員への包括的投資
- **リモートワーク孤立感対策**: デジタル帰属意識向上プログラム

### 施策案3: Citizenship Crafting（市民性設計）プログラム
**学術的根拠**: Citizenship Fatigue軽減理論 - 個人動機に基づく活動設計  
- **動機別活動マッチングシステム**: 
  - 公的認知志向者 → 組織内可視性の高い活動
  - 他者支援志向者 → 社内外支援・メンタリング活動
- **エネルギー付与型タスク設計**: 疲労ではなくエンゲージメント向上を目的
- **参加型意思決定プロセス**: 職務関連決定への従業員参画促進

### 施策案4: 組織信頼・キャリア開発統合システム
**学術的根拠**: 組織信頼向上とキャリアコミットメント理論
- **透明性向上メカニズム**: 意思決定プロセスの可視化・説明責任強化
- **個別キャリア開発計画**: データ駆動型キャリアパス設計・進捗管理
- **共感・思いやり指標**: メンタルヘルス・福祉・保護の定量評価システム

## 学術的根拠・参考文献

### 主要理論文献
- **Zieba, K. (2023)**. "Great Resignation and Quiet Quitting as Post-Pandemic Dangers to Knowledge Management". *Proceedings of the 24th European Conference on Knowledge Management, ECKM 2023*, pp.1516-1522.
  - **因果ループ図**: R1強化ループ（Performance Gap → Dissatisfaction → Disengagement → Actual Performance）
  - **B1バランシングループ**: Job Creep ⇄ Expected Performance ⇄ Performance Gap  
  - **B2バランシングループ**: Disengagement → Work-Life Balance → Dissatisfaction
  - **対策理論**: Citizenship Crafting, Sense of Belonging向上, Job Creep管理

### 追加参考文献（今後調査予定）
- Harter, J. (2022). "Is quiet quitting real?" *Gallup*
- Klotz, A., & Bolino, M. (2022). "When quiet quitting is worse than the real thing." *Harvard Business Review*
- Sull, D., Sull, C. and Zweig, B. (2022). "Toxic culture is driving the Great Resignation." *MIT Sloan Management Review*

## 次のステップ
1. **EDA（探索的データ分析）の実施**
   - 因果ループ変数の基本統計・分布確認
   - Performance Gap指標の構築・可視化
2. **市場分析（quiet quitting関連の最新研究・事例調査）**
   - 日本企業における静かな退職事例調査
   - ITコンサル業界固有の課題分析
3. **因果ループ理論検証のための機械学習モデル構築**
   - 構造方程式モデリング（SEM）による因果関係検証
   - Random Forest・XGBoostによる予測精度比較
4. **事業効果の定量評価（ROI計算）**
   - 静かな退職による知識損失コスト算定
   - 提案施策の投資対効果分析
5. **学術的根拠に基づくプレゼンテーション資料作成** 