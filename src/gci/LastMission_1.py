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
from sklearn.metrics import roc_auc_score, accuracy_score  # ROC AUCスコアと精度を計算する評価指標
from sklearn.base import clone  # モデルをクローンするための関数
import xgboost as xgb  # XGBoostライブラリ
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif  # 特徴量選択

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

def train_cross_validation_models(dataset, features, target_column, model, n_splits=5):
    """
    クロスバリデーションでモデルを訓練し、訓練済みモデルのリストを返す
    
    Args:
        dataset (pd.DataFrame): データセット
        features (list): 学習に使用する特徴量のカラムリスト
        target_column (str): 目的変数のカラム名
        model: 使用する機械学習モデル（sklearn互換）
        n_splits (int): クロスバリデーションの分割数
    
    Returns:
        list: 訓練済みモデルのリスト
    """
    print(f"\n=== {target_column} 予測モデル訓練開始 ===")
    print(f"使用モデル: {type(model).__name__}")
    
    # データセットを訓練用とテスト用に分割
    X = dataset[features]
    y = dataset[target_column]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=511, stratify=y)
    
    print(f"特徴量数: {len(features)}")
    print(f"訓練データ: {train_X.shape[0]}件, テストデータ: {test_X.shape[0]}件")
    print(f"目的変数の分布: {dict(y.value_counts().sort_index())}")
    
    # クロスバリデーション用のデータ（訓練データ全体を使用）
    X = train_X
    y = train_y
    
    # CVの設定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=511)
    
    # スコア格納用
    cv_scores = []
    test_pred_list = []
    models = []
    
    # 二値分類か多クラス分類かを判定
    is_binary = len(y.unique()) == 2
    
    # Stratified K-Fold による学習と評価
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # モデル学習（引数で渡されたモデルをクローンして使用）
        model_fold = clone(model)
        model_fold.fit(X_train, y_train)
        models.append(model_fold)
        
        if is_binary:
            # 二値分類（Attrition）
            y_valid_pred_proba = model_fold.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, y_valid_pred_proba)
            cv_scores.append(auc)
            print(f"CV  AUC: {round(auc, 4)}")
            
            # テストデータ予測
            test_pred_proba = model_fold.predict_proba(test_X)[:, 1]
            test_pred_list.append(test_pred_proba)
            test_auc = roc_auc_score(test_y, test_pred_proba)
            print(f"Real AUC: {round(test_auc, 4)}")
        else:
            # 多クラス分類（StressRating）
            y_valid_pred = model_fold.predict(X_valid)
            y_valid_pred_proba = model_fold.predict_proba(X_valid)
            
            # 精度を計算
            accuracy = accuracy_score(y_valid, y_valid_pred)
            cv_scores.append(accuracy)
            print(f"CV  Accuracy: {round(accuracy, 4)}")
            
            # AUCも計算（多クラス分類用）
            try:
                auc = roc_auc_score(y_valid, y_valid_pred_proba, multi_class='ovr')
                print(f"CV  AUC: {round(auc, 4)}")
            except:
                print(f"CV  AUC: N/A")
            
            # テストデータ予測
            test_pred = model_fold.predict(test_X)
            test_pred_proba = model_fold.predict_proba(test_X)
            test_pred_list.append(test_pred)
            
            # テストデータでの精度
            test_accuracy = accuracy_score(test_y, test_pred)
            print(f"Real Accuracy: {round(test_accuracy, 4)}")
            
            # テストデータでのAUC
            try:
                test_auc = roc_auc_score(test_y, test_pred_proba, multi_class='ovr')
                print(f"Real AUC: {round(test_auc, 4)}")
            except:
                print(f"Real AUC: N/A")
    
    if is_binary:
        # 平均AUCを表示
        mean_score = np.mean(cv_scores)
        print(f"\nAverage Validation AUC: {round(mean_score, 4)}")
        
        # テスト予測の平均を計算
        test_pred_mean = np.mean(test_pred_list, axis=0)
        final_auc = roc_auc_score(test_y, test_pred_mean)
        print(f"Average Real AUC: {round(final_auc, 4)}")
    else:
        # 平均精度を表示
        mean_score = np.mean(cv_scores)
        print(f"\nAverage Validation Accuracy: {round(mean_score, 4)}")
        
        # テスト予測のアンサンブル（多数決）
        test_pred_array = np.array(test_pred_list)
        test_pred_ensemble = []
        for i in range(test_pred_array.shape[1]):
            values, counts = np.unique(test_pred_array[:, i], return_counts=True)
            test_pred_ensemble.append(values[np.argmax(counts)])
        
        ensemble_accuracy = accuracy_score(test_y, test_pred_ensemble)
        print(f"Ensemble Test Accuracy: {round(ensemble_accuracy, 4)}")
    
    return models

## 5. 離職予測モデル

print("\n" + "="*50)
print("離職予測分析")
print("="*50)

# 離職予測用の特徴量を指定
target = "Attrition"

# kbestによる特徴量選択
features = select_top_correlated_features(dataset, target, n_features=25)

# 離職予測用モデルの設定
attrition_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=511,
    eval_metric='logloss'
)

# 離職予測モデルを訓練
attrition_models = train_cross_validation_models(dataset, features, target, attrition_model, n_splits=5)

# 離職予測の特徴量重要度を可視化
print("\n=== 離職予測の特徴量重要度 ===")
attrition_feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': attrition_models[-1].feature_importances_
}).sort_values(by='Importance', ascending=False)

print(attrition_feature_importances.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(data=attrition_feature_importances.head(15), x='Importance', y='Feature')
plt.title('離職予測 - 特徴量重要度 (Top 15)', fontsize=16)
plt.xlabel('重要度')
plt.ylabel('特徴量')
plt.tight_layout()
plt.show()

## 6. ストレス予測モデル

print("\n" + "="*50)
print("ストレス予測分析")
print("="*50)

# ストレス予測用の特徴量を指定
target = "StressRating"

# kbestによる特徴量選択（Attritionを除外）
features = select_top_correlated_features(dataset, target, n_features=20, exclude_features=['Attrition'])

# ストレス予測用モデルの設定
stress_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=511,
    eval_metric='mlogloss',  # 多クラス分類用の評価指標
    objective='multi:softprob'  # 多クラス分類用の目的関数
)

# ストレス予測モデルを訓練
stress_models = train_cross_validation_models(dataset, features, target, stress_model, n_splits=5)

# ストレス予測の特徴量重要度を可視化
print("\n=== ストレス予測の特徴量重要度 ===")
stress_feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': stress_models[-1].feature_importances_
}).sort_values(by='Importance', ascending=False)

print(stress_feature_importances.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(data=stress_feature_importances.head(15), x='Importance', y='Feature')
plt.title('ストレス予測 - 特徴量重要度 (Top 15)', fontsize=16)
plt.xlabel('重要度')
plt.ylabel('特徴量')
plt.tight_layout()
plt.show()

## 7. 結果比較


print("\n分析完了！")



