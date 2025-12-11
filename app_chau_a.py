import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# ==========================
# 1. HÀM TRAIN MODEL 1 LẦN
# ==========================
@st.cache_resource
def train_model():
    # Đọc dữ liệu gốc Châu Á
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

    # Các cột số & cột phân loại
    numeric_features = ["Tuổi", "Chiều cao (m)", "Cân nặng (kg)"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = DecisionTreeClassifier(
        max_depth=11,        # theo kết quả test độ sâu bạn đã chạy
        min_samples_split=20,
        random_state=42,
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    # Train trên toàn bộ dữ liệu (demo sản phẩm)
    model.fit(X, y)

    return model, df, numeric_features, categorical_features


model, df, numeric_features, categorical_features = train_model()

# ==========================
# 2. GIAO DIỆN NHẬP THÔNG TIN
# ==========================

st.title("Demo dự đoán mức độ béo phì (Decision Tree - Dữ liệu Châu Á)")
st.write("Nhập thông tin theo các câu hỏi bên dưới rồi bấm **Dự đoán**.")

# --- 3 cột số nhập trực tiếp ---
col1, col2, col3 = st.columns(3)

with col1:
    # Label giống code mẫu của bạn, nhưng map vào cột 'Tuổi'
    tuoi = st.number_input("Tuổi tác", min_value=5, max_value=100, value=25, step=1)

with col2:
    chieu_cao = st.number_input(
        "Chiều cao (m)",
        min_value=1.20,
        max_value=2.20,
        value=1.70,
        step=0.01,
        format="%.2f",
    )

with col3:
    can_nang = st.number_input(
        "Cân nặng (kg)",
        min_value=20,
        max_value=200,
        value=60,
        step=1,
    )

# Tạo dict dữ liệu người mới
# LƯU Ý: key trong dict phải trùng tên cột trong file CSV
nguoi_moi = {
    "Tuổi": tuoi,
    "Chiều cao (m)": chieu_cao,
    "Cân nặng (kg)": can_nang,
}

st.markdown("### Các câu hỏi thói quen sinh hoạt")

# --- Các cột phân loại: lấy trực tiếp từ dữ liệu để tránh sai chính tả ---
for col in categorical_features:
    options = sorted(df[col].dropna().unique())
    value = st.selectbox(col, options, index=0)
    nguoi_moi[col] = value

# ==========================
# 3. DỰ ĐOÁN
# ==========================

if st.button("Dự đoán"):
    input_df = pd.DataFrame([nguoi_moi])
    du_doan = model.predict(input_df)[0]
    st.success(f"Kết quả dự đoán cho người này: **{du_doan}**")
