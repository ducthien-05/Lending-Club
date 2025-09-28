import gradio as gr
import joblib
import pandas as pd

# Load mô hình và các đối tượng
model = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/loan_default_model.pkl")
ordinal = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/ordinal_encoder.pkl")
ohe = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/onehot_encoder.pkl")
scaler = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/scaler.pkl")
nominal_feature = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/nominal_features.pkl")
ordinal_feature = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/ordinal_features.pkl")
num_feature = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/numerical_features.pkl")
feature_names = joblib.load("D:/TIN/K3N2/KPDL/Final Test/Topic 1/models/feature_names.pkl")


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

    result_text = (
        f"❌ Người này **CÓ NGUY CƠ VỠ NỢ** với xác suất: **{probability * 100:.2f}%**"
        if prediction == 1 else
        f"✅ Người này **KHÔNG CÓ NGUY CƠ VỠ NỢ**, xác suất vỡ nợ chỉ là: **{probability * 100:.2f}%**"
    )
    return result_text


# ------------------ GIAO DIỆN ------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Loan Default Prediction") as demo:
    gr.Markdown(
        """
        # 💳 Loan Default Prediction App  
        Nhập thông tin của người vay để dự đoán **khả năng vỡ nợ** bằng mô hình Machine Learning.  
        ---
        """
    )

    with gr.Row():
        with gr.Column():
            annual_inc = gr.Number(label="Thu nhập hàng năm")
            emp_length = gr.Dropdown(
                choices=['< 1 year', '1 year', '2 years', '3 years', '4 years',
                         '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'],
                label="Kinh nghiệm làm việc"
            )
            verification_status = gr.Dropdown(
                choices=['Not Verified', 'Source Verified', 'Verified'],
                label="Trạng thái xác minh"
            )
            term = gr.Dropdown(choices=[36, 60], label="Thời hạn khoản vay (tháng)")
            installment = gr.Number(label="Khoản trả hàng tháng")
            sub_grade = gr.Dropdown(
                choices=['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5',
                         'C1','C2','C3','C4','C5','D1','D2','D3','D4','D5',
                         'E1','E2','E3','E4','E5','F1','F2','F3','F4','F5',
                         'G1','G2','G3','G4','G5'],
                label="Sub-grade"
            )
            purpose = gr.Dropdown(
                choices=['credit_card','debt_consolidation','home_improvement','educational',
                         'major_purchase','medical','car','vacation','small_business',
                         'house','moving','renewable_energy','wedding','Other'],
                label="Mục đích vay"
            )

        with gr.Column():
            fico_range_high = gr.Number(label="FICO Score Cao Nhất")
            dti = gr.Number(label="DTI (Tỷ lệ nợ/thu nhập)")
            avg_cur_bal = gr.Number(label="Số dư trung bình hiện tại")
            bc_open_to_buy = gr.Number(label="Hạn mức tín dụng còn lại")
            open_acc = gr.Number(label="Số tài khoản đang mở")
            revol_util = gr.Number(label="Tỷ lệ sử dụng tín dụng quay vòng")

        with gr.Column():
            acc_open_past_24mths = gr.Number(label="Số TK mở trong 24 tháng")
            mort_acc = gr.Number(label="Số tài khoản thế chấp")
            tot_hi_cred_lim = gr.Number(label="Tổng hạn mức tín dụng cao nhất")
            credit_years = gr.Number(label="Số năm có tín dụng")
            total_acc = gr.Number(label="Tổng số tài khoản")
            total_rev_hi_lim = gr.Number(label="Hạn mức tín dụng quay vòng cao nhất")

    with gr.Row():
        with gr.Column(scale=1):  # Cột trống bên trái
            pass
        with gr.Column(scale=2, elem_id="center-col"):  # Cột giữa để căn giữa
            submit_btn = gr.Button("🚀 Dự đoán", elem_id="predict-btn")
            output = gr.Textbox(label="Kết quả dự đoán", elem_id="result-box", lines=3)
        with gr.Column(scale=1):  # Cột trống bên phải
            pass

    # Liên kết nút bấm với hàm
    submit_btn.click(
        fn=predict_default,
        inputs=[annual_inc, emp_length, verification_status, term, installment,
                sub_grade, purpose, fico_range_high, dti, avg_cur_bal, bc_open_to_buy,
                open_acc, revol_util, total_acc, total_rev_hi_lim,
                acc_open_past_24mths, mort_acc, tot_hi_cred_lim, credit_years],
        outputs=output
    )
demo.css = """
#center-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
#predict-btn {
    font-size: 18px;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 10px;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    margin-bottom: 15px;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4);
}
#predict-btn:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
}
#result-box {
    width: 100%;
    max-width: 500px;
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    padding: 15px;
    font-size: 16px;
    background: #f9fafb;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
}
"""
demo.launch()
