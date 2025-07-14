import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# 页面配置
st.set_page_config(page_title="PB Predictor", layout="centered")
st.title("A prediction model for MPP-plastic bronchitis in children")
st.markdown("Enter 5 clinical variables to predict the risk of plastic bronchitis (PB) and understand the key factors driving the model's decision.")


# 加载模型与背景数据
@st.cache_resource
def load_model():
    return joblib.load("logistic_model.pkl")

@st.cache_resource
def load_background():
    return joblib.load("shap_background_data.pkl")


model = load_model()
background_data = load_background()
explainer = shap.Explainer(model, background_data)

# 用户输入
age = st.number_input("Age", value=5.0, step=0.1)
wbc_nlr = st.selectbox("NLR > 2", [0, 1])
ddi = st.number_input("DDI", value=0.5, step=0.01)
crp_ldh = st.selectbox("CRP > 30 and LDH > 300", [0, 1])
stenosis = st.selectbox("Tracheal stenosis", [0, 1])

input_df = pd.DataFrame([{
    "Age": age,
    "NLR>2": wbc_nlr,
    "DDI": ddi,
    "CRP>30 & LDH>300": crp_ldh,
    "Tracheal stenosis": stenosis,
}])

# 预测与解释
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Based on feature values, predicted possibility of PB is：**{prob * 100:.2f}%**")

    # st.subheader("SHAP 力图解释")
    shap_values = explainer(input_df)

    # 修正后的SHAP力图生成
    fig= shap.plots.force(
        explainer.expected_value[1],  # 使用模型的基础期望值（正类）
        shap_values.values[0, :, 1],  # 获取正类的SHAP值
        input_df.iloc[0],  # 特征值
        feature_names=input_df.columns.tolist(),
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
