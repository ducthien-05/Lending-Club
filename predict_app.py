import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load mô hình và các đối tượng
model = joblib.load("loan_default_model.pkl")
ordinal = joblib.load("ordinal_encoder.pkl")
ohe = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
nominal_feature = joblib.load("nominal_features.pkl")
ordinal_feature = joblib.load("ordinal_features.pkl")
num_feature = joblib.load("numerical_features.pkl")
feature_names = joblib.load("feature_names.pkl")


def predict_default(
    annual_inc, emp_length, verification_status, term, installment, sub_grade, purpose,
    fico_range_high, dti, avg_cur_bal, bc_open_to_buy, open_acc, revol_util, total_acc,
    total_rev_hi_lim, acc_open_past_24mths, mort_acc, tot_hi_cred_lim, credit_years
):
    # Tạo DataFrame từ input
    input_dict = {
        'annual_inc': [annual_inc],
        'emp_length': [emp_length],
        'verification_status': [verification_status],
        'term': [term],
        'installment': [installment],
        'sub_grade': [sub_grade],
        'purpose': ['Other' if purpose in ['major_purchase', 'medical', 'car', 'vacation',
                                           'small_business', 'house', 'moving', 'renewable_energy', 'wedding']
                    else purpose],
        'fico_range_high': [fico_range_high],
        'dti': [dti],
        'avg_cur_bal': [avg_cur_bal],
        'bc_open_to_buy': [bc_open_to_buy],
        'open_acc': [open_acc],
        'revol_util': [revol_util],
        'total_acc': [total_acc],
        'total_rev_hi_lim': [total_rev_hi_lim],
        'acc_open_past_24mths': [acc_open_past_24mths],
        'mort_acc': [mort_acc],
        'tot_hi_cred_lim': [tot_hi_cred_lim],
        'credit_years': [credit_years]
    }

    df = pd.DataFrame(input_dict)

    # Encode ordinal
    df[ordinal_feature] = ordinal.transform(df[ordinal_feature])

    # One-hot encode nominal
    df_ohe = pd.DataFrame(ohe.transform(df[nominal_feature]),
                          columns=ohe.get_feature_names_out(nominal_feature))
    df = df.drop(nominal_feature, axis=1)
    df = pd.concat([df, df_ohe], axis=1)

    # Đảm bảo đúng số lượng cột đầu vào cho model
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Dự đoán xác suất
    probability = model.predict_proba(df_scaled)[0][1]
    prediction = model.predict(df_scaled)[0]

    result_text = f"❌ Người này **CÓ NGUY CƠ VỠ NỢ** với xác suất: **{probability * 100:.2f}%**" if prediction == 1 else \
                  f"✅ Người này **KHÔNG CÓ NGUY CƠ VỠ NỢ**, xác suất vỡ nợ chỉ là: **{probability * 100:.2f}%**"
    return result_text


# Giao diện người dùng với Gradio
interface = gr.Interface(
    fn=predict_default,
    inputs=[
        gr.Number(label="Thu nhập hàng năm (annual_inc)"),
        gr.Dropdown(choices=['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                             '6 years', '7 years', '8 years', '9 years', '10+ years'], label="Kinh nghiệm làm việc (emp_length)"),
        gr.Dropdown(choices=['Not Verified', 'Source Verified', 'Verified'], label="Trạng thái xác minh"),
        gr.Dropdown(choices=[36, 60], label="Thời hạn khoản vay (tháng)"),
        gr.Number(label="Khoản trả hàng tháng (installment)"),
        gr.Dropdown(choices=['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
                             'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
                             'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
                             'G1', 'G2', 'G3', 'G4', 'G5'], label="Sub-grade"),
        gr.Dropdown(choices=['credit_card', 'debt_consolidation', 'home_improvement', 'educational',
                             'major_purchase', 'medical', 'car', 'vacation', 'small_business',
                             'house', 'moving', 'renewable_energy', 'wedding', 'Other'], label="Mục đích vay (purpose)"),
        gr.Number(label="FICO Score Cao Nhất"),
        gr.Number(label="DTI (Tỷ lệ nợ trên thu nhập)"),
        gr.Number(label="Số dư trung bình hiện tại"),
        gr.Number(label="Hạn mức tín dụng còn lại (bc_open_to_buy)"),
        gr.Number(label="Số tài khoản đang mở (open_acc)"),
        gr.Number(label="Tỷ lệ sử dụng tín dụng quay vòng (revol_util)"),
        gr.Number(label="Tổng số tài khoản (total_acc)"),
        gr.Number(label="Hạn mức tín dụng quay vòng cao nhất"),
        gr.Number(label="Số tài khoản mở trong 24 tháng qua"),
        gr.Number(label="Số tài khoản thế chấp"),
        gr.Number(label="Tổng hạn mức tín dụng cao nhất từng có"),
        gr.Number(label="Số năm có tín dụng (credit_years)")
    ],
    outputs=gr.Textbox(label="Kết quả dự đoán"),
    title="Dự đoán khả năng vỡ nợ",
    description="Nhập thông tin của người vay để dự đoán khả năng vỡ nợ bằng mô hình Machine Learning."
)

# Chạy ứng dụng
interface.launch(share=True)
