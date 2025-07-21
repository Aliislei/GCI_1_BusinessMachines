# I社人事データ - データディクショナリ

## データセット概要
- **データ名**: IBM HR Analytics Employee Attrition & Performance (I社版)
- **行数**: 1,470名の従業員データ
- **列数**: 44カラム
- **期間**: 2023-2024年
- **通貨単位**: USD（外資系国際企業）

## カラム詳細説明

### 1. 基本属性情報
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| Age | 年齢 | 数値 | 18-60歳 |
| Gender | 性別 | カテゴリ | Female, Male |
| MaritalStatus | 婚姻状況 | カテゴリ | Single, Married, Divorced |
| Over18 | 18歳以上フラグ | 固定値 | Y（全員18歳以上） |

### 2. 教育・経歴情報
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| Education | 教育レベル | 数値 | 1-5（1:低 → 5:高） |
| EducationField | 教育専門分野 | カテゴリ | Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other |
| NumCompaniesWorked | 転職回数 | 数値 | 過去の勤務企業数 |
| TotalWorkingYears | 総勤務年数 | 数値 | 全キャリアでの勤務年数 |

### 3. 組織・職務情報
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| Department | 所属部署 | カテゴリ | Research & Development, Sales, Human Resources |
| JobRole | 職種 | カテゴリ | Research Scientist, Laboratory Technician, Sales Executive, Sales Representative, Manager, Manufacturing Director, Research Director, Healthcare Representative, Human Resources |
| JobLevel | 職位レベル | 数値 | 1-5（1:低 → 5:高） |
| JobInvolvement | 職務関与度 | 数値 | 1-4（1:低 → 4:高） |
| YearsAtCompany | 在社年数 | 数値 | 現在の会社での勤務年数 |
| YearsInCurrentRole | 現職種年数 | 数値 | 現在の職種での勤務年数 |
| YearsSinceLastPromotion | 昇進からの年数 | 数値 | 最後の昇進からの経過年数 |
| YearsWithCurrManager | 現上司との年数 | 数値 | 現在の上司の下での勤務年数 |

### 4. 勤務条件・環境
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| BusinessTravel | 出張頻度 | カテゴリ | Non-Travel, Travel_Rarely, Travel_Frequently |
| DistanceFromHome | 自宅からの距離 | 数値 | 通勤距離（単位不明） |
| OverTime | 残業時間 | 数値 | 0-61時間/月 |
| StandardHours | 標準労働時間 | 固定値 | 40時間/週 |
| RemoteWork | リモートワーク頻度 | 数値 | 0-5（0:なし → 5:頻繁） |
| FlexibleWork | フレックス制度利用 | バイナリ | 0:利用なし, 1:利用あり |

### 5. 採用・雇用形態
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| HowToEmploy | 採用経路 | カテゴリ | intern, agent_A, agent_B, agent_C, direct_recruiting, New_graduate_recruitment |
| EmployeeCount | 従業員カウント | 固定値 | 1（全員） |
| EmployeeNumber | 従業員番号 | 数値 | ユニークID |

### 6. 満足度・評価指標
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| EnvironmentSatisfaction | 職場環境満足度 | 数値 | 1-4（1:低 → 4:高） |
| JobSatisfaction | 職務満足度 | 数値 | 1-4（1:低 → 4:高） |
| RelationshipSatisfaction | 人間関係満足度 | 数値 | 1-4（1:低 → 4:高） |
| WorkLifeBalance | ワークライフバランス | 数値 | 1-4（1:悪い → 4:良い） |

### 7. パフォーマンス指標
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| PerformanceIndex | 実パフォーマンス指数 | 数値 | 30-100（客観的な成果指標） |
| PerformanceRating | パフォーマンス評価 | 数値 | 1-4（1:低 → 4:高、マネージャー評価） |
| MonthlyAchievement | 月次成果達成度 | 数値 | USD（業績評価を金額で表現した指標） |

### 8. ストレス・健康指標
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| StressRating | ストレス評価 | 数値 | 1-5（1:低 → 5:高） |
| StressSelfReported | 自己申告ストレス | 数値 | 1-4（1:低 → 4:高） |

### 9. 報酬・待遇
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| MonthlyIncome | 月収 | 数値 | USD（ドル建て） |
| Incentive | インセンティブ | 数値 | USD（追加報酬） |
| StockOptionLevel | ストックオプション | 数値 | 0-3（0:なし → 3:高レベル） |

### 10. 福利厚生・制度
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| WelfareBenefits | 福利厚生レベル | 数値 | 1-4（1:基本 → 4:充実） |
| InHouseFacility | 社内施設利用 | バイナリ | 0:利用なし, 1:利用あり |
| ExternalFacility | 外部施設利用 | バイナリ | 0:利用なし, 1:利用あり |
| ExtendedLeave | 長期休暇制度利用 | バイナリ | 0:利用なし, 1:利用あり |

### 11. 成長・開発
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| TrainingTimesLastYear | 昨年研修回数 | 数値 | 年間の研修参加回数 |

### 12. 目的変数
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| Attrition | 離職フラグ | カテゴリ | No:在籍, Yes:離職 |

### 13. 時間情報
| カラム名 | 意味 | データ型 | 値の範囲 |
|----------|------|----------|----------|
| Year | データ年 | 数値 | 2023, 2024 |

## 重要な注意事項

### パフォーマンス指標の違い
- **PerformanceIndex**: 客観的な成果指標（30-100）
- **PerformanceRating**: マネージャーによる主観評価（1-4の5段階評価）
- **MonthlyAchievement**: 月次業績を金額で表現した評価指標（2,094-26,999 USD）

### データの特性
- 外資系国際企業のデータのため、日本特有の文化的要因は反映されていない可能性
- 通貨はUSD建てのため、日本円換算時は為替レートを考慮
- 一部カラムの詳細説明は実務上の制約により未開示

### 分析における活用ポイント
1. **離職予測**: Attritionを目的変数とした予測モデル構築
2. **満足度分析**: 各種満足度指標と離職の関係性分析
3. **パフォーマンス分析**: 客観指標と主観評価の相関分析
4. **ワークライフバランス**: リモートワーク、残業時間との関係性
5. **報酬分析**: 給与体系と従業員エンゲージメントの関係

---
*作成日: 2025年7月*  
*データソース: IBM HR Analytics Employee Attrition & Performance (I社版)* 