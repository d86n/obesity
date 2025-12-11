import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# 1. ĐỌC DỮ LIỆU ĐÃ MÃ HÓA
# ==========================

# Đọc X đã mã hóa (Châu Á)
X = pd.read_csv("X_ChauA_encoded.csv", encoding="utf-8-sig")

# Đọc y đã mã hóa (Châu Á)
y_df = pd.read_csv("y_ChauA_encoded.csv", encoding="utf-8-sig")

# Nếu file y_ChauA_encoded có 2 cột: label_text, label_encoded
if "label_encoded" in y_df.columns:
    y = y_df["label_encoded"]
else:
    # Trường hợp bạn chỉ lưu mỗi cột y là số
    y = y_df.iloc[:, 0]

print("Kích thước X:", X.shape)
print("Kích thước y:", y.shape)

# ==========================
# 2. CHIA TRAIN / TEST
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% dữ liệu để test
    random_state=42,
    stratify=y           # giữ tỉ lệ các lớp
)

# ==========================
# 3. TẠO & TRAIN DECISION TREE
# ==========================

clf = DecisionTreeClassifier(
    criterion="gini",    # hoặc "entropy"
    max_depth=11,        # sau này bạn test depth rồi sửa lại
    min_samples_split=20,
    random_state=42
)

clf.fit(X_train, y_train)

# ==========================
# 4. ĐÁNH GIÁ MÔ HÌNH
# ==========================

y_pred = clf.predict(X_test)

print("=== KẾT QUẢ DECISION TREE (Châu Á) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================
# 5. IN CẤU TRÚC CÂY RA FILE
# ==========================

feature_names = list(X.columns)

tree_rules = export_text(clf, feature_names=feature_names)
with open("decision_tree_rules_ChauA.txt", "w", encoding="utf-8") as f:
    f.write(tree_rules)

print("\nĐã lưu luật của cây vào file decision_tree_rules_ChauA.txt")
