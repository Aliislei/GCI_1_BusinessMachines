# [ベースライン] GCI最終課題：I社人事データによるストレス評価予測分析

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

## 2. データ読み込み

# 読み込むデータが格納されたディレクトリのパス（※必要に応じて変更の必要があります）
PATH = 'data/'

dataset = pd.read_csv(PATH + 'data.csv')  # データセットの読み込み

# まず初めに、データのサイズを確認してみましょう。

print('Dataset:', dataset.shape)

# データセットの概要を確認しました。後でこのデータを訓練用とテスト用に分割します。  

# 次に、データセットの初めの5データを見てみましょう。

dataset.head()

# pandasのDataFrameは.info()を使用することで、詳細な情報を確認できます。

dataset.info()

## 3. データの分析・EDA

dataset.isnull().sum()

# 欠損値があることが分かりました。これらは後で対処することとします。  

# 次にストレス評価の分布を見てみましょう。

stress_counts = dataset['StressRating'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
plt.bar(stress_counts.index.astype(str), stress_counts.values)
plt.title('Distribution of StressRating', fontsize=16)
plt.xlabel('StressRating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# ストレス評価の割合を見てみましょう。

stress_percentage = dataset['StressRating'].value_counts(normalize=True) * 100

for rating in sorted(stress_percentage.index):
    print(f"StressRating {rating}: {stress_percentage[rating]:.2f}%")


# 数値列だけを取り出す
numeric_cols = dataset.select_dtypes(include=['number']).columns
numeric_cols = numeric_cols.drop(['EmployeeNumber'])

# プロット
num_cols = len(numeric_cols)
cols = 3
rows = (num_cols + cols - 1) // cols

plt.figure(figsize=(5 * cols, 4 * rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, cols, i)
    plt.hist(dataset[col].dropna(), bins=30, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 数値列だけを取り出す
numeric_cols = dataset.select_dtypes(include=['number']).drop(['EmployeeNumber'], axis=1)

# 相関行列を計算
corr_matrix = numeric_cols.corr()

# ヒートマップをプロット
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5
)

plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.show()

# 次は、カテゴリデータについて可視化してみましょう。
# カテゴリデータを抽出
categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns

# 各列の水準数を取得
levels_count = {col: dataset[col].nunique() for col in categorical_cols}

for col, count in levels_count.items():
    print(f"{col}: {count} levels")

# カテゴリデータ（object型またはcategory型）を抽出
categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns

# グラフ描画準備
num_cols = len(categorical_cols)
rows = 1
cols = num_cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5))

if cols == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# 各カテゴリ変数でカウントプロット
for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=dataset, order=dataset[col].value_counts().index, ax=axes[i])
    axes[i].set_title(f'Count Plot of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Count', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# カテゴリ変数（object型またはcategory型）を抽出
categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns

# グラフ描画準備
num_cols = len(categorical_cols)
rows = 1
cols = num_cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5))

if cols == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# 各カテゴリ変数ごとに ストレス評価平均を棒グラフで描画
for i, col in enumerate(categorical_cols):
    mean_values = dataset.groupby(col)['StressRating'].mean().sort_values(ascending=False)
    sns.barplot(x=mean_values.index, y=mean_values.values, ax=axes[i])
    axes[i].set_title(f'Average StressRating by {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Average StressRating', fontsize=12)
    axes[i].set_ylim(1, 5)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


## 4. 前処理

# 欠損補完、エンコーディングを行います。  
# EmployeeNumber列は従業員の識別番号であり、離職予測に直接関係がないと予想されるため、削除します。

# 使わない列の削除
dataset = dataset.drop(columns=["EmployeeNumber"])

# 欠損値がある列について、数値データは平均値で補完します。

# 数値列の欠損値を平均で補完
numeric_cols_with_missing = dataset.select_dtypes(include=['number']).columns
for col in numeric_cols_with_missing:
    if dataset[col].isnull().sum() > 0:
        mean_value = dataset[col].mean()
        dataset[col] = dataset[col].fillna(mean_value)
        print(f"{col}の欠損値を平均値 {mean_value:.2f} で補完")

# 欠損値があるかを確認してみましょう。

dataset.isnull().sum()

# カテゴリデータをラベルエンコーディング
label_encoders = {}
categorical_cols_to_encode = dataset.select_dtypes(include=['object']).columns
for c in categorical_cols_to_encode:
    # 目的変数StressRatingは数値なので変換不要
    label_encoders[c] = LabelEncoder()
    dataset[c] = label_encoders[c].fit_transform(dataset[c].astype(str))
    print(f"{c}をラベルエンコーディング")

## 5. ベースラインモデル

def train_cross_validation_models(dataset, features, target_column, n_splits=5):
    """
    クロスバリデーションでモデルを訓練し、訓練済みモデルのリストを返す
    
    Args:
        dataset (pd.DataFrame): データセット
        features (list): 学習に使用する特徴量のカラムリスト
        target_column (str): 目的変数のカラム名
        n_splits (int): クロスバリデーションの分割数
    
    Returns:
        list: 訓練済みモデルのリスト
    """
    # データセットを訓練用とテスト用に分割
    X = dataset[features]
    y = dataset[target_column]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # クロスバリデーション用のデータ（訓練データ全体を使用）
    X = train_X
    y = train_y
    
    # CVの設定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # スコア格納用
    auc_scores = []
    test_pred_proba_list = []
    models = []
    
    # Stratified K-Fold による学習と評価
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # モデル学習
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=2025
        )
        model.fit(X_train, y_train)
        models.append(model)
        
        # 予測とスコアリング（多クラス分類）
        y_valid_pred = model.predict(X_valid)
        y_valid_pred_proba = model.predict_proba(X_valid)
        
        # 精度を計算
        accuracy = accuracy_score(y_valid, y_valid_pred)
        auc_scores.append(accuracy)  # 精度を保存
        print(f"CV  Accuracy: {round(accuracy, 4)}")
        
        # AUCも計算（多クラス分類用）
        try:
            auc = roc_auc_score(y_valid, y_valid_pred_proba, multi_class='ovr')
            print(f"CV  AUC: {round(auc, 4)}")
        except:
            print(f"CV  AUC: N/A")
        
        # テストデータ予測を保存
        test_pred = model.predict(test_X)
        test_pred_proba = model.predict_proba(test_X)
        test_pred_proba_list.append(test_pred)  # 予測ラベルを保存
        
        # テストデータでの精度
        test_accuracy = accuracy_score(test_y, test_pred)
        print(f"Real Accuracy: {round(test_accuracy, 4)}")
        
        # テストデータでのAUC
        try:
            test_auc = roc_auc_score(test_y, test_pred_proba, multi_class='ovr')
            print(f"Real AUC: {round(test_auc, 4)}")
        except:
            print(f"Real AUC: N/A")
    
    # 平均精度を表示
    mean_accuracy = np.mean(auc_scores)  # auc_scoresには精度が保存されている
    print(f"\nAverage Validation Accuracy: {round(mean_accuracy, 4)}")
    
    # テスト予測の多数決を計算（最も多い予測値を選択）
    test_pred_array = np.array(test_pred_proba_list)
    test_pred_ensemble = []
    for i in range(test_pred_array.shape[1]):
        # 各サンプルの予測値の中で最も多いものを選択
        values, counts = np.unique(test_pred_array[:, i], return_counts=True)
        test_pred_ensemble.append(values[np.argmax(counts)])
    
    ensemble_accuracy = accuracy_score(test_y, test_pred_ensemble)
    print(f"\nEnsemble Test Accuracy: {round(ensemble_accuracy, 4)}")
    
    return models

# 学習に使用する特徴量を指定
target_column = "StressRating"
features = [col for col in dataset.columns if col != target_column]

# クロスバリデーションでモデルを訓練
models = train_cross_validation_models(dataset, features, target_column, n_splits=5)

# 一部の機械学習モデルでは、入力した特徴量の重要度を計算することができます。特徴量重要度を可視化してみましょう。
# 特徴量とその重要度をDataFrameにまとめる（最後に訓練されたモデルを使用）
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': models[-1].feature_importances_
}).sort_values(by='Importance', ascending=False)

# 可視化
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature')
plt.title('Feature Importances from Random Forest', fontsize=16)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



