# [ベースライン] GCI最終課題：I社人事データによる離職予測・ストレス予測分析

# モジュールのインポート
import numpy as np  # 数値計算や配列操作を行うためのライブラリ
import pandas as pd  # 表形式のデータを扱うためのライブラリ
import matplotlib.pyplot as plt  # データ可視化のための基本的なグラフ描画ライブラリ
import seaborn as sns  # 高機能な統計グラフを描画するライブラリ
import japanize_matplotlib  # 日本語フォント対応
from sklearn.preprocessing import LabelEncoder  # カテゴリ変数を数値に変換するエンコーダ
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレストによる分類器
from sklearn.model_selection import StratifiedKFold, train_test_split  # 層化K分割交差検証とデータ分割を行うクラス
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score  # ROC AUCスコアと精度を計算する評価指標
from sklearn.base import clone  # モデルをクローンするための関数
import xgboost as xgb  # XGBoostライブラリ
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif  # 特徴量選択
from sklearn.ensemble import VotingClassifier  # アンサンブル用
from sklearn.linear_model import LogisticRegression  # ロジスティック回帰
from catboost import CatBoostClassifier  # CatBoost追加
import lightgbm as lgb  # LightGBM追加
from sklearn.neighbors import NearestNeighbors  # k近傍法
from imblearn.over_sampling import SMOTE  # SMOTEライブラリ


## 1. データ読み込み

# 読み込むデータが格納されたディレクトリのパス
PATH = 'data/'

dataset = pd.read_csv(PATH + 'data.csv')  # データセットの読み込み

print('Dataset:', dataset.shape)
print("\n=== データセット情報 ===")
dataset.info()

## 2. 前処理

print("\n=== 前処理開始 ===")

# EmployeeNumber列は従業員の識別番号であり、予測に直接関係がないため削除
dataset = dataset.drop(columns=["EmployeeNumber"])

# 数値列の欠損値を平均で補完
numeric_cols_with_missing = dataset.select_dtypes(include=['number']).columns
for col in numeric_cols_with_missing:
    if dataset[col].isnull().sum() > 0:
        mean_value = dataset[col].mean()
        dataset[col] = dataset[col].fillna(mean_value)
        print(f"{col}の欠損値を平均値 {mean_value:.2f} で補完")

# 欠損値確認
print(f"前処理後の欠損値: {dataset.isnull().sum().sum()}個")


# Attrition/Genderを数値に変換。同時にカラムもint型にする
dataset['Attrition'] = dataset['Attrition'].map({'Yes': 1, 'No': 0}).astype(int)
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0}).astype(int)

# StressRatingを0から始まるクラスラベルに変換（XGBoost対応）
print("StressRatingを0ベースに変換: 1→0, 2→1, 3→2, 4→3, 5→4")
dataset['StressRating'] = dataset['StressRating'] - 1

# HowToEmploy・BusinessTravel・MaritalStatusはワンホットエンコーディング
dataset = pd.get_dummies(dataset, columns=['HowToEmploy', 'BusinessTravel', 'MaritalStatus'])

# 残ったカテゴリカルデータをラベルエンコーディング
label_encoders = {}
categorical_cols_to_encode = dataset.select_dtypes(include=['object']).columns
for c in categorical_cols_to_encode:
    if c not in ['Attrition']:  # 目的変数Attritionは後で個別に処理
        label_encoders[c] = LabelEncoder()
        dataset[c] = label_encoders[c].fit_transform(dataset[c].astype(str))
        print(f"{c}をラベルエンコーディング")

# 週あたり勤務拘束時間を算出
# OverTimeは月間残業時間と仮定
# 週あたり通勤時間を算出する。まず片道通勤時間を２倍して1日あたり通勤時間を算出し、(5-Remote Work)を掛けると週あたり通勤時間になる。
# 片道通勤時間：通勤距離2km未満は0.5h。30km未満は1h。30km以上は1.5hとする。
commute_time = dataset['DistanceFromHome']
commute_time = commute_time.apply(lambda x: 0.5 if x < 2 else 1 if x < 30 else 1.5)
commute_time = commute_time * 2
commute_time = commute_time * (5 - dataset['RemoteWork'])
dataset['WeeklyHours'] = (dataset['StandardHours']) + (dataset['OverTime'] / 4) + commute_time

# 評定ランクを算出
# 評定ランクは、PerformanceRatingとJobLevelの組み合わせで算出する。
# PerformanceRating1は降格級、4は昇格級と考える。
dataset['PerformanceRank'] = dataset['PerformanceRating'] + (dataset['JobLevel'] - 1) * 2

# 年収を算出
# 年収は、MonthlyIncomeを12倍し、Incentiveを加算する。他カラムとオーダーが若干異なるためstandardscalerで標準化する
scaler = StandardScaler()
dataset['AnnualIncome'] = dataset['MonthlyIncome'] * 12 + dataset['Incentive']
dataset['AnnualIncome'] = scaler.fit_transform(dataset['AnnualIncome'].values.reshape(-1, 1)).flatten()

# 厚生制度利用率を算出
#　社内施設・社外施設・長期休暇・フレックスのうち利用しているものの数を数える。
#　その数を5で割ったものを厚生制度利用率とする。
dataset['HealthcareUtilization'] = (dataset['InHouseFacility'] + dataset['ExternalFacility'] + dataset['ExtendedLeave'] + dataset['FlexibleWork']) / 4

#　満足度評価指標平均値を算出
#　満足度評価指標は、JobSatisfaction・EnvironmentSatisfaction・RelationshipSatisfaction・WorkLifeBalanceの平均値とする。
dataset['SatisfactionScore'] = (dataset['JobSatisfaction'] + dataset['EnvironmentSatisfaction'] + dataset['RelationshipSatisfaction'] + dataset['WorkLifeBalance']) / 4

## 3. 特徴量選択関数

def select_top_correlated_features(dataset, target_column, n_features=20, exclude_features=None):
    """
    SelectKBestを使用して目的変数との関連が高い上位n個の特徴量を選択する
    
    Args:
        dataset (pd.DataFrame): データセット
        target_column (str): 目的変数のカラム名
        n_features (int): 選択する特徴量の数
        exclude_features (list): 除外する特徴量のリスト
    
    Returns:
        list: 選択された特徴量のリスト
    """
    print(f"\n=== {target_column} とのSelectKBestによる特徴量選択 ===")
    
    # 数値データのみを対象とする
    numeric_features = dataset.select_dtypes(include=['number']).columns.tolist()
    
    # 目的変数を除外
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    # 除外指定された特徴量を除外
    if exclude_features:
        for feature in exclude_features:
            if feature in numeric_features:
                numeric_features.remove(feature)
                print(f"除外指定: {feature}")
    
    print(f"対象特徴量数: {len(numeric_features)}個")
    
    # 定数列を除外
    valid_features = []
    for feature in numeric_features:
        if dataset[feature].std() == 0:
            print(f"定数列のためスキップ: {feature}")
            continue
        valid_features.append(feature)
    
    if len(valid_features) < n_features:
        print(f"警告: 有効な特徴量数({len(valid_features)})が指定数({n_features})より少ないため、全て使用します")
        n_features = len(valid_features)
    
    # SelectKBestを使用して特徴量選択
    X = dataset[valid_features]
    y = dataset[target_column]
    
    selector = SelectKBest(score_func=f_classif, k=n_features)
    selector.fit(X, y)
    
    # 選択された特徴量を取得
    selected_mask = selector.get_support()
    selected_features = [valid_features[i] for i in range(len(valid_features)) if selected_mask[i]]
    
    # スコアを取得して表示
    feature_scores = selector.scores_
    feature_score_pairs = [(valid_features[i], feature_scores[i]) for i in range(len(valid_features)) if selected_mask[i]]
    feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"上位{n_features}個の特徴量を選択（F統計量順）:")
    for i, (feature, score) in enumerate(feature_score_pairs, 1):
        print(f"{i:2d}. {feature:<25} F統計量: {score:.4f}")
    
    return selected_features

## 4. ベースラインモデル関数

def train_cross_validation_models(dataset, features, target_column, model, n_splits=10):
    """
    クロスバリデーションでモデルを訓練し、精度確認後にデータセット全体で学習したモデルを返す
    
    Args:
        dataset (pd.DataFrame): データセット
        features (list): 学習に使用する特徴量のカラムリスト
        target_column (str): 目的変数のカラム名
        model: 使用する機械学習モデル（sklearn互換）
        n_splits (int): クロスバリデーションの分割数
    
    Returns:
        model: データセット全体で学習したモデル
    """
    print(f"\n=== {target_column} 予測モデル訓練開始 ===")
    print(f"使用モデル: {type(model).__name__}")
    
    # データセットの準備
    X = dataset[features]
    y = dataset[target_column]
    
    print(f"特徴量数: {len(features)}")
    print(f"データセット: {X.shape[0]}件")
    print(f"目的変数の分布: {dict(y.value_counts().sort_index())}")
    
    # ホールドアウトデータを確保
    X_train_cv, X_holdout, y_train_cv, y_holdout = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=511
    )
    print(f"CV用データ: {X_train_cv.shape[0]}件")
    print(f"ホールドアウトデータ: {X_holdout.shape[0]}件")
    
    # CVの設定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=511)
    cv_scores = []
    f1_scores = []  # F1スコアを記録するリスト
    precision_scores = []  # Precisionスコアを記録するリスト
    recall_scores = []  # Recallスコアを記録するリスト
    
    # 二値分類か多クラス分類かを判定
    is_binary = len(y.unique()) == 2
    
    # Stratified K- Fold による学習と評価
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_cv, y_train_cv), 1):
        print(f"Fold {fold}")
        
        # 訓練データと検証データに分割
        X_train, X_valid = X_train_cv.iloc[train_idx], X_train_cv.iloc[valid_idx]
        y_train, y_valid = y_train_cv.iloc[train_idx], y_train_cv.iloc[valid_idx]
        
        # モデルをクローンして学習
        model_fold = clone(model)
        model_fold.fit(X_train, y_train)
        
        if is_binary:
            # 二値分類（Attrition）
            y_valid_pred_proba = model_fold.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, y_valid_pred_proba)
            cv_scores.append(auc)
            print(f"CV  AUC: {round(auc, 4)}")
        else:
            # 多クラス分類（StressRating）
            y_valid_pred = model_fold.predict(X_valid)
            y_valid_pred_proba = model_fold.predict_proba(X_valid)
            
            # F1スコア（マクロ平均）を計算
            f1_macro = f1_score(y_valid, y_valid_pred, average='macro')
            f1_scores.append(f1_macro)
            print(f"CV  F1-Macro: {round(f1_macro, 4)}")
            
            # Precisionスコア（マクロ平均）を計算
            precision_macro = precision_score(y_valid, y_valid_pred, average='macro')
            precision_scores.append(precision_macro)
            print(f"CV  Precision-Macro: {round(precision_macro, 4)}")
            
            # Recallスコア（マクロ平均）を計算
            recall_macro = recall_score(y_valid, y_valid_pred, average='macro')
            recall_scores.append(recall_macro)
            print(f"CV  Recall-Macro: {round(recall_macro, 4)}")
                
    # データセット全体でモデルを学習
    print(f"\n=== データセット全体での最終モデル学習 ===")
    final_model = clone(model)
    final_model.fit(X_train_cv, y_train_cv)

    print(f"\n=== 予測精度値 ===")
    if is_binary:
        # CV平均AUCを表示
        mean_score = np.mean(cv_scores)
        print(f"Average Validation AUC: {round(mean_score, 4)}")
        
        # ホールドアウトデータでの評価
        y_holdout_pred_proba = final_model.predict_proba(X_holdout)[:, 1]
        holdout_auc = roc_auc_score(y_holdout, y_holdout_pred_proba)
        print(f"Holdout AUC: {round(holdout_auc, 4)}")
    else:
        # VV F1スコアの平均も表示（多クラス分類の場合）
        mean_f1 = np.mean(f1_scores)
        print(f"Average Validation F1-Macro: {round(mean_f1, 4)}")
        
        # CV Precisionスコアの平均も表示
        mean_precision = np.mean(precision_scores)
        print(f"Average Validation Precision-Macro: {round(mean_precision, 4)}")
        
        # CV Recallスコアの平均も表示
        mean_recall = np.mean(recall_scores)
        print(f"Average Validation Recall-Macro: {round(mean_recall, 4)}")
        
        # ホールドアウトデータでの評価
        y_holdout_pred = final_model.predict(X_holdout)
        holdout_f1 = f1_score(y_holdout, y_holdout_pred, average='macro')
        holdout_precision = precision_score(y_holdout, y_holdout_pred, average='macro')
        holdout_recall = recall_score(y_holdout, y_holdout_pred, average='macro')
        print(f"Holdout F1-Macro: {round(holdout_f1, 4)}")
        print(f"Holdout Precision-Macro: {round(holdout_precision, 4)}")
        print(f"Holdout Recall-Macro: {round(holdout_recall, 4)}")
    
    
    return final_model

## 5. 離職予測モデル

print("\n" + "="*50)
print("離職予測分析")
print("="*50)

# 離職予測用の特徴量を指定
target = "Attrition"

# kbestによる特徴量選択
attrition_features = select_top_correlated_features(dataset, target, n_features=25)

# 離職予測用モデルの設定
attrition_rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=511)
attrition_cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=511,
    verbose=0,
    loss_function='Logloss',
    eval_metric='Logloss'
)
attrition_lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=511,
    objective='binary',
    class_weight=None,
    verbose=-1  # ログ抑制
)
attrition_xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=511,
    eval_metric='logloss'
)
attrition_model = VotingClassifier(
    estimators=[
        ('xgb', attrition_xgb_model),
        ('rf', attrition_rf_model),
        ('cat', attrition_cat_model),
        ('lgb', attrition_lgb_model)
    ],
    voting='soft',
    n_jobs=-1
)

# 離職予測モデルを訓練
attrition_models = train_cross_validation_models(dataset, attrition_features, target, attrition_model)

# 離職予測の特徴量重要度を可視化
# VotingClassifierはfeature_importances_を持たないため、XGBoostの重要度を表示
print("\n=== 離職予測の特徴量重要度 (XGBoostのみ) ===")
attrition_xgb_fitted = attrition_models.named_estimators_["xgb"] if hasattr(attrition_models, "named_estimators_") else attrition_models.estimators_[0]
attrition_feature_importances = pd.DataFrame({
    'Feature': attrition_features,
    'Importance': attrition_xgb_fitted.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(attrition_feature_importances.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(data=attrition_feature_importances.head(15), x='Importance', y='Feature')
plt.title('離職予測 - 特徴量重要度 (Top 15, XGBoost)', fontsize=16)
plt.xlabel('重要度')
plt.ylabel('特徴量')
plt.tight_layout()
plt.savefig('src/attrition_feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

## 6. ストレス予測モデル

print("\n" + "="*50)
print("ストレス予測分析")
print("="*50)

# ストレス予測用の特徴量を指定
target = "StressRating"

# SMOTEによるリサンプリング
print(f"\n=== ストレス予測用データにSMOTE適用 ===")
X_for_stress = dataset.drop(columns=[target, 'Attrition'])
y_for_stress = dataset[target]
print(f"SMOTE適用前のクラス分布: {dict(y_for_stress.value_counts())}")

smote = SMOTE(random_state=511)
X_stress_resampled, y_stress_resampled = smote.fit_resample(X_for_stress, y_for_stress)
print(f"SMOTE適用後のクラス分布: {dict(pd.Series(y_stress_resampled).value_counts())}")

# リサンプリングされたデータでデータセットを再構築
dataset_stress_resampled = pd.DataFrame(X_stress_resampled, columns=X_for_stress.columns)
dataset_stress_resampled[target] = y_stress_resampled

# kbestによる特徴量選択（リサンプリングされたデータセットを使用）
stress_features = select_top_correlated_features(dataset_stress_resampled, target, n_features=15, exclude_features=['Attrition'])

# ストレス予測用モデルの設定
#　TODO:離職モデルとは構成を変えた方が良い？そもそも多クラス分類と回帰どちらが良い？
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=511)
cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=511,
    verbose=0,
    loss_function='MultiClass',
    eval_metric='MultiClass'
)
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=511,
    objective='multiclass',
    class_weight=None,
    verbose=-1  # ログ抑制
)
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=511,
    eval_metric='mlogloss',  # 多クラス分類用の評価指標
    objective='multi:softprob'  # 多クラス分類用の目的関数
)
stress_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('cat', cat_model),
        ('lgb', lgb_model)
    ],
    voting='soft',
    n_jobs=-1
)

# ストレス予測モデルを訓練
stress_models = train_cross_validation_models(dataset_stress_resampled, stress_features, target, stress_model)

# ストレス予測の特徴量重要度を可視化
# VotingClassifierはfeature_importances_を持たないため、XGBoostの重要度を表示
print("\n=== ストレス予測の特徴量重要度 (XGBoostのみ) ===")
xgb_fitted = stress_models.named_estimators_["xgb"] if hasattr(stress_models, "named_estimators_") else stress_models.estimators_[0]
stress_feature_importances = pd.DataFrame({
    'Feature': stress_features,
    'Importance': xgb_fitted.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(stress_feature_importances.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(data=stress_feature_importances.head(15), x='Importance', y='Feature')
plt.title('ストレス予測 - 特徴量重要度 (Top 15, XGBoost)', fontsize=16)
plt.xlabel('重要度')
plt.ylabel('特徴量')
plt.tight_layout()
plt.savefig('src/stress_feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

## 7. 打ち手の分析
### 7-1 福利厚生推進によるストレスレベル低下確認
print("\n" + "="*50)
print("福利厚生推進施策の検証用データセット作成")
print("="*50)

# 1. 一定パフォーマンス（PerformanceIndex:80）以上の社員への長期休暇支給制度（必ず利用）
# 2. 一定パフォーマンス以上かつストレスレベル４・５の社員への福利厚生設備およびフレックス制度の適用

# 元データセットをコピー
df_welfare_promotion = dataset.copy()

# 施策適用前にストレス値/離職率を予測モデルベースに変更
df_welfare_promotion['StressRating'] = stress_models.predict(df_welfare_promotion[stress_features])
df_welfare_promotion['Attrition'] = attrition_models.predict(df_welfare_promotion[attrition_features])

# 施策適用前の状況確認
print("\n=== 施策適用前の状況 ===")
print(f"全体離職率: {df_welfare_promotion['Attrition'].mean():.3f} ({df_welfare_promotion['Attrition'].sum()}名)")
print(f"PerformanceIndex >= 80の社員数: {(df_welfare_promotion['PerformanceIndex'] >= 80).sum()}名")
print(f"PerformanceIndex >= 80の社員の離職率: {df_welfare_promotion[df_welfare_promotion['PerformanceIndex'] >= 80]['Attrition'].mean():.3f} ({df_welfare_promotion[df_welfare_promotion['PerformanceIndex'] >= 80]['Attrition'].sum()}名)")
print(f"PerformanceIndex >= 80 かつ StressRating >= 3の社員数: {((df_welfare_promotion['PerformanceIndex'] >= 80) & (df_welfare_promotion['StressRating'] >= 3)).sum()}名")

#　静かな退職群のパフォーマンス平均値確認
#　静かな退職条件：退職していない・ストレスレベルが０or1・パフォーマンスが下位25%・ワークライフバランスが3以上
quiet_retirement_mask = (df_welfare_promotion['Attrition'] == 0) & (df_welfare_promotion['StressRating'] <= 1) & (df_welfare_promotion['PerformanceIndex'] <= df_welfare_promotion['PerformanceIndex'].quantile(0.25)) & (df_welfare_promotion['WorkLifeBalance'] >= 3)
quiet_retirement_performance_mean = df_welfare_promotion[quiet_retirement_mask]['PerformanceIndex'].mean()
print(f"静かな退職群のパフォーマンス平均値: {quiet_retirement_performance_mean:.2f}")

total_performance_sum = df_welfare_promotion[df_welfare_promotion['Attrition'] == 0]['PerformanceIndex'].sum()

#　施策実施前のパフォーマンス低下予測
df_performance_calc = df_welfare_promotion.copy()
# 高ストレス社員はいずれ静かな退職状態に移行すると仮定する
#　退職していない高ストレス社員のパフォーマンスは静かな退職群のパフォーマンス平均値となる
df_performance_calc.loc[(df_performance_calc['StressRating'] >= 3)&(df_performance_calc['Attrition'] == 0)&(df_performance_calc['PerformanceIndex'] >= quiet_retirement_performance_mean), 'PerformanceIndex'] = quiet_retirement_performance_mean
# 実際に退職した社員のパフォーマンスは失われる
df_performance_calc.loc[df_performance_calc['Attrition'] == 1, 'PerformanceIndex'] = 0
performance_index_reduce_before = total_performance_sum - df_performance_calc['PerformanceIndex'].sum()
print(f"施策実施前のパフォーマンス損失: {performance_index_reduce_before}")



# ストレスレベル別の社員数
print("\nストレスレベル別の社員数:")
for stress_level in range(5):  # 0-4の5レベル
    count = (df_welfare_promotion['StressRating'] == stress_level).sum()
    percentage = (count / len(df_welfare_promotion)) * 100
    print(f"  StressRating {stress_level}: {count}名 ({percentage:.1f}%)")

# 施策1: 一定パフォーマンス（PerformanceIndex:80）以上の社員への長期休暇支給制度（必ず利用）
high_performance_mask = df_welfare_promotion['PerformanceIndex'] >= 80
df_welfare_promotion.loc[high_performance_mask, 'ExtendedLeave'] = 1

print(f"\n施策1適用: PerformanceIndex >= 80の社員{high_performance_mask.sum()}名に長期休暇制度を強制適用")

# 施策2: 一定パフォーマンス以上かつストレスレベル４・５の社員への福利厚生設備およびフレックス制度の適用（必ず利用）
# StressRatingは0ベースに変換されているため、元の4・5は3・4に相当
high_performance_high_stress_mask = (df_welfare_promotion['PerformanceIndex'] >= 80) & (df_welfare_promotion['StressRating'] >= 3)
#　InHouseは利用しやすい。
df_welfare_promotion.loc[high_performance_high_stress_mask, 'InHouseFacility'] = 1
#　外部施設は利用率向上の難易度が高い
# 非利用者の3割を利用に転換できれば上々と考える。元々1ならそのまま、0なら30%の確率で1に変更
np.random.seed(511)
for index in df_welfare_promotion.index:
    if high_performance_high_stress_mask[index] and df_welfare_promotion.loc[index, 'ExternalFacility'] == 0:
        if np.random.random() < 0.3:
            df_welfare_promotion.loc[index, 'ExternalFacility'] = 1
#　フレックスは活用推奨しやすい。全員活用で。
df_welfare_promotion.loc[high_performance_high_stress_mask, 'FlexibleWork'] = 1

print(f"施策2適用: PerformanceIndex >= 80 かつ StressRating >= 3の社員{high_performance_high_stress_mask.sum()}名に福利厚生設備・フレックス制度を強制適用")

# 施策適用後の状況確認
print("\n=== 施策適用後の状況 ===")
print(f"長期休暇制度利用率: {df_welfare_promotion['ExtendedLeave'].mean():.3f} ({df_welfare_promotion['ExtendedLeave'].sum()}名)")
print(f"社内施設利用率: {df_welfare_promotion['InHouseFacility'].mean():.3f} ({df_welfare_promotion['InHouseFacility'].sum()}名)")
print(f"外部施設利用率: {df_welfare_promotion['ExternalFacility'].mean():.3f} ({df_welfare_promotion['ExternalFacility'].sum()}名)")
print(f"フレックス制度利用率: {df_welfare_promotion['FlexibleWork'].mean():.3f} ({df_welfare_promotion['FlexibleWork'].sum()}名)")

# 厚生制度利用率を再計算
df_welfare_promotion['HealthcareUtilization'] = (df_welfare_promotion['InHouseFacility'] + 
                                                 df_welfare_promotion['ExternalFacility'] + 
                                                 df_welfare_promotion['ExtendedLeave'] + 
                                                 df_welfare_promotion['FlexibleWork']) / 4

print(f"厚生制度利用率（平均）: {df_welfare_promotion['HealthcareUtilization'].mean():.3f}")

# 施策適用対象者の詳細確認
print("\n=== 施策適用対象者の詳細 ===")
target_employees = df_welfare_promotion[high_performance_high_stress_mask]
print(f"施策2適用対象者数: {len(target_employees)}名")
if len(target_employees) > 0:
    print(f"対象者の平均PerformanceIndex: {target_employees['PerformanceIndex'].mean():.2f}")
    print(f"対象者の平均StressRating: {target_employees['StressRating'].mean():.2f}")
    print(f"対象者の部署分布:")
    print(target_employees['Department'].value_counts())

print("\n検証用データセット作成完了！")
print(f"作成されたデータセット: {df_welfare_promotion.shape}")

###  ストレス予測モデルによるストレス値再算定
print("\n" + "="*50)
print("ストレス予測モデルによるストレス値再算定")
print("="*50)

# 検証用データセットからストレス予測に必要な特徴量を準備
# ストレス予測モデルで使用した特徴量のみを抽出
X_welfare_for_stress = df_welfare_promotion[stress_features]

# ストレス予測モデルで新しいストレス値を予測
print("\n=== ストレス予測モデルによる再予測 ===")
stress_predictions = stress_models.predict(X_welfare_for_stress)

# 予測結果をデータセットに追加
df_welfare_promotion['StressRating_Original'] = df_welfare_promotion['StressRating'].copy()
df_welfare_promotion['StressRating'] = stress_predictions

print(f"ストレス予測完了: {len(stress_predictions)}件")

# 再算定後のストレスレベル別の社員数
print("\n=== 再算定後のストレスレベル別の社員数 ===")
for stress_level in range(5):  # 0-4の5レベル
    count = (df_welfare_promotion['StressRating'] == stress_level).sum()
    percentage = (count / len(df_welfare_promotion)) * 100
    print(f"  StressRating {stress_level}: {count}名 ({percentage:.1f}%)")

# 施策適用対象者のストレス変化
print("\n=== 施策適用対象者のストレス変化 ===")
target_employees_after = df_welfare_promotion[high_performance_high_stress_mask]
if len(target_employees_after) > 0:
    target_original_mean = target_employees_after['StressRating_Original'].mean()
    target_predicted_mean = target_employees_after['StressRating'].mean()
    print(f"施策2適用対象者の平均ストレスレベル: {target_original_mean:.3f} → {target_predicted_mean:.3f} (変化: {target_predicted_mean - target_original_mean:+.3f})")


print("\nストレス値再算定完了！")

### 7-4 施策対象者のストレスレベル分布変化の可視化
print("\n" + "="*50)
print("施策対象者のストレスレベル分布変化の可視化")
print("="*50)

# 施策対象者のストレスレベル分布を取得
target_employees_1 = df_welfare_promotion[high_performance_high_stress_mask]

# 施策前後のストレスレベル分布を集計
stress_before = target_employees_1['StressRating_Original'].value_counts().sort_index()
stress_after = target_employees_1['StressRating'].value_counts().sort_index()

# 全てのストレスレベル（0-4）を含むように調整
all_levels = range(5)
stress_before_full = pd.Series(0, index=all_levels)
stress_after_full = pd.Series(0, index=all_levels)

for level in all_levels:
    if level in stress_before.index:
        stress_before_full[level] = stress_before[level]
    if level in stress_after.index:
        stress_after_full[level] = stress_after[level]

# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 施策前の分布
bars1 = ax1.bar(stress_before_full.index, stress_before_full.values, color='lightcoral', alpha=0.7)
ax1.set_title('施策前のストレスレベル分布', fontsize=14, fontweight='bold')
ax1.set_xlabel('ストレスレベル')
ax1.set_ylabel('社員数')
ax1.set_xticks(range(5))
ax1.set_ylim(0, max(stress_before_full.max(), stress_after_full.max()) + 2)

# バーの上に数値を表示
for bar, value in zip(bars1, stress_before_full.values):
    if value > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(int(value)), ha='center', va='bottom', fontweight='bold')

# 施策後の分布
bars2 = ax2.bar(stress_after_full.index, stress_after_full.values, color='lightgreen', alpha=0.7)
ax2.set_title('施策後のストレスレベル分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('ストレスレベル')
ax2.set_ylabel('社員数')
ax2.set_xticks(range(5))
ax2.set_ylim(0, max(stress_before_full.max(), stress_after_full.max()) + 2)

# バーの上に数値を表示
for bar, value in zip(bars2, stress_after_full.values):
    if value > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(int(value)), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('src/policy_target_stress_distribution_change.png', dpi=300, bbox_inches='tight')
plt.show()

#　7−2　福利厚生推進の結果、離職率はどう変動するか
df_welfare_attrition = df_welfare_promotion.copy()
df_welfare_attrition['Attrition'] = attrition_models.predict(df_welfare_attrition[attrition_features])
print(f"施策後 離職率: {df_welfare_attrition['Attrition'].mean():.3f} ({df_welfare_attrition['Attrition'].sum()}名)")
print(f"施策後 PerformanceIndex >= 80の社員の離職率: {df_welfare_attrition[df_welfare_promotion['PerformanceIndex'] >= 80]['Attrition'].mean():.3f} ({df_welfare_attrition[df_welfare_attrition['PerformanceIndex'] >= 80]['Attrition'].sum()}名)")

#　インセンティブ施策は有意な効果が得られなかった。却下する
"""
#　７−3　インセンティブ制度変更
#　df_welfare_attritionをベースに、インセンティブを再配分したデータセットを作成する
#　現状のインセンティブ総和を確認
incentive_total = df_welfare_attrition['Incentive'].sum()
print(f"現状のインセンティブ総和: {incentive_total}")

#　インセンティブをPerformanceIndex80以上の社員にPerformanceIndex比例で再配分
#　PerformanceIndexが80以上の社員のPerformanceIndexの合計を計算
performance_index_total = df_welfare_attrition[df_welfare_attrition['PerformanceIndex'] >= 80]['PerformanceIndex'].sum()
print(f"PerformanceIndexが80以上の社員のPerformanceIndexの合計: {performance_index_total}")
#　インセンティブをPerformanceIndex比例で再配分（80以上の社員のみ）
df_welfare_attrition['Incentive_Original'] = df_welfare_attrition['Incentive'].copy()
df_welfare_attrition['Incentive'] = 0
df_welfare_attrition.loc[df_welfare_attrition['PerformanceIndex'] >= 80, 'Incentive'] = performance_index_total * df_welfare_attrition['PerformanceIndex'] / performance_index_total

#　インセンティブ総和を確認
incentive_total_after = df_welfare_attrition['Incentive'].sum()
print(f"施策後のインセンティブ総和: {incentive_total_after}")

#　ストレスおよび退職率率の変化を確認
df_welfare_attrition['StressRating_Original'] = df_welfare_attrition['StressRating'].copy()
df_welfare_attrition['StressRating'] = stress_models.predict(df_welfare_attrition[stress_features])
df_welfare_attrition['Attrition_Original'] = df_welfare_attrition['Attrition'].copy()
df_welfare_attrition['Attrition'] = attrition_models.predict(df_welfare_attrition[attrition_features])

print(f"インセンティブ施策後 離職率: {df_welfare_attrition['Attrition'].mean():.3f} ({df_welfare_attrition['Attrition'].sum()}名)")
print(f"インセンティブ施策後 PerformanceIndex >= 80の社員の離職率: {df_welfare_attrition[df_welfare_attrition['PerformanceIndex'] >= 80]['Attrition'].mean():.3f} ({df_welfare_attrition[df_welfare_attrition['PerformanceIndex'] >= 80]['Attrition'].sum()}名)")
"""

#　8　総合効果確認
# 高ストレス社員はいずれ静かな退職状態に移行すると仮定する
#　退職していない高ストレス社員のパフォーマンスは静かな退職群のパフォーマンス平均値となる
df_welfare_promotion.loc[(df_welfare_promotion['StressRating'] >= 3)&(df_welfare_promotion['Attrition'] == 0)&(df_welfare_promotion['PerformanceIndex'] >= quiet_retirement_performance_mean), 'PerformanceIndex'] = quiet_retirement_performance_mean
# 実際に退職した社員のパフォーマンスは失われる
df_welfare_promotion.loc[df_welfare_promotion['Attrition'] == 1, 'PerformanceIndex'] = 0
# 施策実施後のパフォーマンス総和をとる
performance_index_reduce_after = total_performance_sum - df_welfare_promotion['PerformanceIndex'].sum()
print(f"施策実施後のパフォーマンス損失: {performance_index_reduce_after}")

# 施策実施後のパフォーマンス総和の変化率をとる
performance_index_reduce_change_rate = (performance_index_reduce_after - performance_index_reduce_before) / performance_index_reduce_before
print(f"施策実施後のパフォーマンス損失の変化率: {performance_index_reduce_change_rate:.2f}")

#　全社パフォーマンスの変化率をとる
performance_index_total_change_rate = (performance_index_reduce_after - performance_index_reduce_before) / total_performance_sum
print(f"施策実施後の全社パフォーマンスの変化率: {performance_index_total_change_rate:.2f}")
# 日本IBM：2023/01-12月期決算の売上高７３０９億円
print(f"施策実施後の全社売上高上昇: {round(-1*performance_index_total_change_rate*7309, 4):.2f}億円")


########################
#　資料ではこの先で&DおよびSalesへの社内募集施策提案を行う。
#　・福利厚生施策の結果若干の工数不足が見込まれる。
#　・そもそも若干の業務過多であったと考えられる
#　・理論上業務量調整もし退職＆静かな退職の防止につながる
#　・理論上自発的業務参画が退職および静かな退職の防止につながる
########################


print("\n分析完了！")




