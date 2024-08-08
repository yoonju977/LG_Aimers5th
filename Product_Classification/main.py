import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. 파일 경로 설정
file_paths = {
    "auto_clave": "/Users/song-yoonju/LG_Aimers/Product_Classification/Auto clave.csv",
    "dam_dispensing": "/Users/song-yoonju/LG_Aimers/Product_Classification/Dam dispensing.csv",
    "fill1_dispensing": "/Users/song-yoonju/LG_Aimers/Product_Classification/Fill1 dispensing.csv",
    "fill2_dispensing": "/Users/song-yoonju/LG_Aimers/Product_Classification/Fill2 dispensing.csv",
    "train_y": "/Users/song-yoonju/LG_Aimers/Product_Classification/train_y.csv",
    "submission": "/Users/song-yoonju/LG_Aimers/Product_Classification/submission.csv",
}

# 2. 데이터 불러오기
X_Dam = pd.read_csv(file_paths["dam_dispensing"], low_memory=False)
X_AutoClave = pd.read_csv(file_paths["auto_clave"], low_memory=False)
X_Fill1 = pd.read_csv(file_paths["fill1_dispensing"], low_memory=False)
X_Fill2 = pd.read_csv(file_paths["fill2_dispensing"], low_memory=False)
y = pd.read_csv(file_paths["train_y"])

# 3. 데이터 병합 및 전처리
# 칼럼명 수정
X_Dam.columns = [i + " - Dam" for i in X_Dam.columns]
X_AutoClave.columns = [i + " - AutoClave" for i in X_AutoClave.columns]
X_Fill1.columns = [i + " - Fill1" for i in X_Fill1.columns]
X_Fill2.columns = [i + " - Fill2" for i in X_Fill2.columns]

# 'Set ID'로 통일
X_Dam = X_Dam.rename(columns={"Set ID - Dam": "Set ID"})
X_AutoClave = X_AutoClave.rename(columns={"Set ID - AutoClave": "Set ID"})
X_Fill1 = X_Fill1.rename(columns={"Set ID - Fill1": "Set ID"})
X_Fill2 = X_Fill2.rename(columns={"Set ID - Fill2": "Set ID"})

# 데이터 병합
X = pd.merge(X_Dam, X_AutoClave, on="Set ID")
X = pd.merge(X, X_Fill1, on="Set ID")
X = pd.merge(X, X_Fill2, on="Set ID")

# 'train_y'와 병합
df_merged = pd.merge(X, y, "inner", on="Set ID")

# 결측치가 절반 이상인 칼럼 제거
drop_cols = [
    column
    for column in df_merged.columns
    if df_merged[column].isnull().sum() > (len(df_merged) / 2)
]
df_merged = df_merged.drop(columns=drop_cols)

# 'LOT ID - Dam' 칼럼 제거
df_merged = df_merged.drop(columns=["LOT ID - Dam"])

# 4. 데이터 분할
df_train, df_val = train_test_split(
    df_merged, test_size=0.3, stratify=df_merged["target"], random_state=110
)

# 5. 모델 학습
model = RandomForestClassifier(random_state=110)

# 숫자형 변환 가능한 특징 선택
features = [
    col for col in df_train.columns if df_train[col].dtype in ["int64", "float64"]
]

train_x = df_train[features]
train_y = df_train["target"]

model.fit(train_x, train_y)

# 6. 검증 데이터로 성능 평가
val_x = df_val[features]
val_y = df_val["target"]

val_pred = model.predict(val_x)

f1 = f1_score(val_y, val_pred, pos_label="AbNormal")
precision = precision_score(val_y, val_pred, pos_label="AbNormal")
recall = recall_score(val_y, val_pred, pos_label="AbNormal")

print(f"F1 Score: {f1}, Precision: {precision}, Recall: {recall}")

# 7. 테스트 데이터 예측 및 결과 저장
df_test_y = pd.read_csv(file_paths["submission"])
df_test = pd.merge(X, df_test_y, "outer", on="Set ID")

# 병합 후 길이 확인
print(
    f"Length of df_test_y: {len(df_test_y)}, Length of df_test after merge: {len(df_test)}"
)

# 중복된 Set ID 확인
duplicates = df_test.duplicated(subset=["Set ID"], keep=False)
duplicate_count = duplicates.sum()
print(f"Number of duplicated Set ID in df_test: {duplicate_count}")

if duplicate_count > 0:
    df_test = df_test.drop_duplicates(subset=["Set ID"])

# 병합 후 데이터에서 실제 테스트 데이터만 남기도록 필터링
df_test = df_test[df_test["Set ID"].isin(df_test_y["Set ID"])]

df_test_x = df_test[features]

# 예측 수행
test_pred = model.predict(df_test_x)

# 예측 값의 길이와 df_test_y 길이 비교
if len(test_pred) == len(df_test_y):
    df_test_y["target"] = test_pred
    df_test_y.to_csv("submission.csv", index=False)
    print("Submission file created successfully")
else:
    print("Length mismatch: Cannot assign predictions to submission file")
