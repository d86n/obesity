import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ==========================
# 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ==========================
df = pd.read_csv("Du_lieu_Chau_A.csv", encoding="utf-8")
df.columns = df.columns.str.strip()  # bỏ khoảng trắng / xuống dòng

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
# 3. THỬ NHIỀU ĐỘ SÂU max_depth
# ==========================
depth_list = [None, 3, 5, 7, 9, 11, 13, 15]

print("Thử các giá trị max_depth khác nhau (Châu Á):")
print("{:<10} {:<15} {:<15}".format("max_depth", "train_accuracy", "test_accuracy"))

for d in depth_list:
    clf = DecisionTreeClassifier(
        max_depth=d,
        min_samples_split=20,
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("{:<10} {:<15.4f} {:<15.4f}".format(str(d), train_acc, test_acc))
