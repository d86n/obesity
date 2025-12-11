import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# 1. ĐỌC & XỬ LÝ DỮ LIỆU
# ==========================

df = pd.read_csv("Du_lieu_Chau_A.csv", encoding="utf-8")
df.columns = df.columns.str.strip()

# Xử lý cột số
df["Chiều cao (m)"] = pd.to_numeric(df["Chiều cao (m)"], errors="coerce")
df["Cân nặng (kg)"] = pd.to_numeric(df["Cân nặng (kg)"], errors="coerce")
df["Tuổi"] = pd.to_numeric(df["Tuổi"], errors="coerce")

# Bỏ các dòng thiếu dữ liệu quan trọng
df = df.dropna(subset=["Tuổi", "Chiều cao (m)", "Cân nặng (kg)"])

target_col = "Mức độ béo phì"
X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = ["Tuổi", "Chiều cao (m)", "Cân nặng (kg)"]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ==========================
# 2. CHIA TRAIN / TEST
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# 3. RANDOM FOREST VỚI CẤU HÌNH (BẠN CÓ THỂ CHỈNH SAU)
# ==========================

rf_clf = RandomForestClassifier(
    n_estimators=300,     # nếu bạn có file test RF Châu Á thì chỉnh theo kết quả đẹp nhất
    max_depth=None,
    min_samples_split=20,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("rf", rf_clf),
])

model.fit(X_train, y_train)

# ==========================
# 4. ĐÁNH GIÁ
# ==========================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("=== RANDOM FOREST - Du_lieu_Chau_A.csv ===")
print("Train accuracy:", accuracy_score(y_train, y_train_pred))
print("Test accuracy :", accuracy_score(y_test, y_test_pred))

print("\nClassification report (TEST):")
print(classification_report(y_test, y_test_pred))

print("Confusion matrix (TEST):")
print(confusion_matrix(y_test, y_test_pred))
