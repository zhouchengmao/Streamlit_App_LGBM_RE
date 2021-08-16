import pandas as pd
import streamlit as st

from MLUtils import *

st.title('LightGBM and RE')  # 算法名称 and XXX

age = None
gender = None
stage = None
T = None
M = None
N = None
risk = None

uploader = None

gbm = None


# 配置侧边栏（添加生成新数据并预测的功能）
def setup_sidebar():
    global age, gender, stage, T, M, N, risk
    st.sidebar.title("Dr. Z.C.M.")

    if uploader is not None:
        age = st.sidebar.number_input("Please input age", value=0, format="%d", key="age")
        gender = st.sidebar.number_input("Please input gender", value=0, format="%d", key="gender")
        stage = st.sidebar.number_input("Please input stage", value=0, format="%d", key="stage")
        T = st.sidebar.number_input("Please input T", value=0, format="%d", key="T")
        M = st.sidebar.number_input("Please input M", value=0, format="%d", key="M")
        N = st.sidebar.number_input("Please input N", value=0, format="%d", key="N")
        risk = st.sidebar.selectbox("Please select risk", options=['low', 'high'], key="risk")

        btn_predict = st.sidebar.button("Do Predict")
        if btn_predict:
            do_predict()


# 配置上传条
def setup_uploader():
    global uploader
    uploader = st.file_uploader('Upload data file here', type=['csv', 'txt'], help='Upload dataset for training',
                                key="uploader")
    if uploader is not None:
        if st.checkbox('Show uploaded file information', value=True, key="cb_file_info"):
            # 显示文件详细信息
            file_details = {
                "FileName": uploader.name, "FileType": uploader.type, "FileSize": uploader.size
            }
            st.write(file_details)

        do_processing()


# 对上传的文件进行处理和展示
def do_processing():
    global uploader
    global gbm
    pocd = read_csv(uploader)

    st.text("Dataset Description")
    st.write(pocd.describe())
    if st.checkbox('Show detail of this dataset'):
        st.write(pocd)

    # 分割数据
    X_train, X_test, y_train, y_test = do_split_data(pocd)
    X_train, X_test, y_train, y_test = do_xy_preprocessing(X_train, X_test, y_train, y_test)

    # 模型训练、显示结果
    gbm = lgb.LGBMClassifier(learning_rate=0.01, n_estimators=200, lambda_l1=0.01, lambda_l2=10, max_depth=10,
                             bagging_fraction=0.8, feature_fraction=0.8)  # lgb
    gbm_result = model_fit_score(gbm, X_train, y_train)
    st.text("Training Result")
    msg = model_print(gbm_result, "LGBMClassifier - Train")
    st.write(msg)
    # 训练画图
    fig_train = plt_roc_auc([
        (gbm_result, 'gbm',),
    ], 'Train ROC')
    st.pyplot(fig_train)
    # 模型测试、显示结果
    gbm_test_result = model_score(gbm, X_test, y_test)
    st.text("Testing Result")
    msg = model_print(gbm_test_result, "LGBMClassifier - Test")
    st.write(msg)
    # 测试画图
    fig_test = plt_roc_auc([
        (gbm_test_result, 'gbm',),
    ], 'Validate ROC')
    st.pyplot(fig_test)


# 对生成的预测数据进行处理
def do_predict():
    global age, gender, stage, T, M, N, risk
    global gbm

    # 处理生成的预测数据的输入
    pocd_predict = pd.DataFrame(data=[
        [age, gender, stage, T, M, N, risk]
    ], columns=COL_INPUT)
    pocd_predict = do_base_preprocessing(pocd_predict, with_y=False)
    st.text("Preview for detail of this predict data")
    st.write(pocd_predict)
    pocd_predict = do_predict_preprocessing(pocd_predict)

    # 进行预测并输出
    pr = gbm.predict(pocd_predict)
    pr = pr.astype(np.int)
    st.markdown(r"$\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{" + str(COL_Y[0]) + r"}$ $\color{red}{is}$ $\color{red}{" + str(pr[0]) + "}$")
    # st.text(f"Predict result {COL_Y[0]} is {pr[0]}")


if __name__ == "__main__":
    setup_uploader()
    setup_sidebar()
