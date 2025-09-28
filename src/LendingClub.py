import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib


def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath)

    loan_status_mapping = {
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    }
    df['loan_status'] = df['loan_status'].map(loan_status_mapping)

    df = df[['annual_inc', 'emp_length', 'verification_status', 'term', 'installment', 'sub_grade', 'purpose',
             'fico_range_high', 'dti', 'avg_cur_bal', 'bc_open_to_buy', 'open_acc', 'revol_util', 'total_acc',
             'earliest_cr_line', 'total_rev_hi_lim', 'acc_open_past_24mths', 'mort_acc', 'tot_hi_cred_lim',
             'loan_status']]

    columns_to_fill = ['dti', 'revol_util', 'bc_open_to_buy', 'avg_cur_bal']
    for col in columns_to_fill:
        df[col] = df[col].fillna(df[col].median())
    df['emp_length'] = df['emp_length'].fillna(df['emp_length'].mode()[0])

    df['credit_years'] = (pd.Timestamp.today() - df['earliest_cr_line']).dt.days / 365.25
    df = df.drop('earliest_cr_line', axis=1)

    df.loc[df['purpose'].isin(['major_purchase', 'medical', 'car', 'vacation',
                               'small_business', 'house', 'moving', 'renewable_energy', 'wedding']), 'purpose'] = 'Other'

    term_values = {' 36 months': 36, ' 60 months': 60}
    df['term'] = df.term.map(term_values)
    return df


def handle_outliers(df, num_feature):
    Q1 = df[num_feature].quantile(0.25)
    Q3 = df[num_feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for col in num_feature:
        df[col] = np.where(df[col] < lower_bound[col], Q1[col], df[col])
        df[col] = np.where(df[col] > upper_bound[col], Q3[col], df[col])

    return df


def train_model():
    df = load_and_preprocess_data("D:/TIN/K3N2/KPDL/Final Test/Topic 1/data/train.xlsx")

    num_feature = ['annual_inc', 'installment', 'fico_range_high', 'dti', 'avg_cur_bal', 'bc_open_to_buy',
                   'open_acc', 'revol_util', 'total_acc', 'total_rev_hi_lim', 'acc_open_past_24mths',
                   'mort_acc', 'tot_hi_cred_lim', 'credit_years']
    ordinal_feature = ['sub_grade', 'emp_length']
    nominal_feature = ['verification_status', 'purpose', 'term']

    df = handle_outliers(df, num_feature)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)

    ordinal = OrdinalEncoder(categories=[
        ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
         'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
         'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
         'G1', 'G2', 'G3', 'G4', 'G5'],
        ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
         '6 years', '7 years', '8 years', '9 years', '10+ years']
    ])
    X_train[ordinal_feature] = ordinal.fit_transform(X_train[ordinal_feature])
    X_test[ordinal_feature] = ordinal.transform(X_test[ordinal_feature])

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    ohe.fit(X_train[nominal_feature])

    X_train_ohe = pd.DataFrame(ohe.transform(X_train[nominal_feature]),
                               columns=ohe.get_feature_names_out(nominal_feature),
                               index=X_train.index)
    X_test_ohe = pd.DataFrame(ohe.transform(X_test[nominal_feature]),
                              columns=ohe.get_feature_names_out(nominal_feature),
                              index=X_test.index)

    X_train = pd.concat([X_train.drop(nominal_feature, axis=1), X_train_ohe], axis=1)
    X_test = pd.concat([X_test.drop(nominal_feature, axis=1), X_test_ohe], axis=1)

    feature_names = X_train.columns.tolist()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sm = RandomOverSampler()
    X_train_new, y_train_new = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, max_features='log2',
                                   criterion='entropy', random_state=42)
    model.fit(X_train_new, y_train_new)

    # Lưu model và các đối tượng đã huấn luyện
    joblib.dump(feature_names, "feature_names.pkl")
    joblib.dump(model, "loan_default_model.pkl")
    joblib.dump(ordinal, "ordinal_encoder.pkl")
    joblib.dump(ohe, "onehot_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(nominal_feature, "nominal_features.pkl")
    joblib.dump(ordinal_feature, "ordinal_features.pkl")
    joblib.dump(num_feature, "numerical_features.pkl")

if __name__ == "__main__":
    train_model()
