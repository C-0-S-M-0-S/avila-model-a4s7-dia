import pandas as pd
from PIL import Image
import streamlit as st
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(layout='wide')

st.title("💡 Inference")
col_md, col_img = st.columns(2)
col_md.markdown("""<br><br><br><br><br>
                The best model found during our research is a Gradient Boosting Classifier.  
                A grid search was performed to find the best hyperparameters, which are the following :  
                - learning rate = 0.2, contribution of each node to the final result, higher values means faster learning but higher overfitting risk.  
                - max depth = 4, maximum depth of the tree, interaction bewteen features, higher values means higher overfitting risk.  
                - number of estimators = 200, number of trees, lower values means more risks of biases.""",
            unsafe_allow_html=True)
col_img.image(Image.open("./streamlit_ui/AIServer.jpg"), width=400, caption="Illustration made with DALL-E 3")

hyperpp_expander = st.expander("Hyperparameters tuning", expanded=True)
hyperpp_expander.write("Default values are the best ones.")

col1, col2, col3 = hyperpp_expander.columns(3, gap="large")
rate_input = col1.slider("learning rate", value=0.2, min_value=0.0, max_value=5.0)
depth_input = col2.slider("max depth", value=4,  min_value=1, max_value=10)
nbestimators_input = col3.slider("number of estimators", value=200, min_value=25, max_value=600)

if st.button("Run model"):
    with st.spinner("Running model..."):
    
        result_expander = st.expander("Results", expanded=True)

        df_train = pd.read_csv('avila/avila-tr.txt', header=None)
        target_train = df_train[10]
        target_train.replace({
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 
            'G': 7, 'H': 8, 'I': 9, 'W': 10, 'X': 11, 'Y': 12}, inplace=True)
        df_train.drop(columns=10, inplace=True)

        df_test = pd.read_csv('avila/avila-ts.txt', header=None)
        target_test = df_test[10]
        target_test.replace({
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 
            'G': 7, 'H': 8, 'I': 9, 'W': 10, 'X': 11, 'Y': 12}, inplace=True)
        df_test.drop(columns=10, inplace=True)

        x_train, y_train, x_test, y_test = df_train, target_train, df_test, target_test  

        if [rate_input,depth_input,nbestimators_input] == [0.2,4,200]:
            mdl = load(filename = 'mdl_0.2_4_200.joblib')

        else:
            mdl = GradientBoostingClassifier(learning_rate=rate_input, max_depth=depth_input, n_estimators=nbestimators_input)
            mdl.fit(x_train, y_train)
            dump(value=mdl, filename=f'mdl_{rate_input}_{depth_input}_{nbestimators_input}.joblib')

        y_pred = mdl.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        col_acc, col_mse = result_expander.columns(2)
        col_acc.metric(label="Accuracy", value=f"{accuracy*100:.2f}%")
        col_mse.metric(label="MSE", value=f"{mse:.2f}")

        fig, ax = plt.subplots()

        display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=mdl.classes_)
        display.plot(ax=ax)
        st.pyplot(fig)

