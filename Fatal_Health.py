import streamlit as st
st. set_page_config(layout="wide", page_icon=":hospital:")
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('default')


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA

########################################################################################################################
start_time=time.time()
#Title for the webpage
tit1,tit2 = st.beta_columns((4, 1))
tit1.markdown("<h1 style='text-align: center;'><u>Fatal Health Conditions Main</u> </h1>",unsafe_allow_html=True)
st.sidebar.title("Dataset and ML Classifier")

dataset_select=st.sidebar.selectbox("Select Dataset: ",('Heart Attack',"Fatal Health Conditions"))
classifier_select = st.sidebar.selectbox("Select ML Classifier: ", ("Logistic Regression","KNN","SVM","Decision Trees",
                                                              "Random Forest","Gradient Boosting","XGBoost"))

LE = LabelEncoder()
def get_dataset(dataset_select):
    if dataset_select == "Heart Attack":
        data=pd.read_csv("https://raw.githubusercontent.com/ajinkyalahade/Streamlit_ML_WebApp/main/Data/heart.csv")
        st.header("Heart Attack UCI Data Based Prediction")
        return data

    else:
        data = pd.read_csv("https://raw.githubusercontent.com/ajinkyalahade/Streamlit_ML_WebApp/main/Data/fetal_health.csv")
        st.header("Fatal Health Conditions")
        return data

data = get_dataset(dataset_select)

def selected_dataset(dataset_select):
    if dataset_select == "Heart Attack":
        X = data.drop(["output"],axis=1)
        Y = data.output
        return X,Y
    elif dataset_select == "Fatal Health Conditions":
        X = data.drop(["fetal_health"],axis=1)
        Y = data.fetal_health
        return X,Y

X,Y = selected_dataset(dataset_select)

#Charts
def plot_op(dataset_select):
    col1, col2 = st.beta_columns((1, 5))
    plt.figure(figsize=(12, 3))
    plt.title("Classes in 'Y'")
    if dataset_select == "Heart Attack":
        col1.write(Y)
        sns.countplot(Y, palette='colorblind')
        col2.pyplot()

    elif dataset_select == "Fatal Health Conditions":
        col1.write(Y)
        sns.countplot(Y, palette='colorblind')
        col2.pyplot()
########################################################################################################################

st.write(data)
st.write("Shape of dataset: ",data.shape)
st.write("Number of classes: ",Y.nunique())
plot_op(dataset_select)

########################################################################################################################

def add_parameter_ui(clf_name):
    params={}
    st.sidebar.write("Select Parameters: ")

    if clf_name == "Logistic Regression":
        R = st.sidebar.slider("Regularization",0.1,10.0,step=0.1)
        MI = st.sidebar.slider("max_iter",50,400,step=50)
        params["R"] = R
        params["MI"] = MI

    elif clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors",1,20)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("Regularization",0.01,10.0,step=0.01)
        kernel = st.sidebar.selectbox("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params["C"] = C
        params["kernel"] = kernel

    elif clf_name == "Decision Trees":
        M = st.sidebar.slider("max_depth", 2, 20)
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        SS = st.sidebar.slider("min_samples_split",1,10)
        params["M"] = M
        params["C"] = C
        params["SS"] = SS

    elif clf_name == "Random Forest":
        N = st.sidebar.slider("n_estimators",50,500,step=50,value=100)
        M = st.sidebar.slider("max_depth",2,20)
        C = st.sidebar.selectbox("Criterion",("gini","entropy"))
        params["N"] = N
        params["M"] = M
        params["C"] = C

    elif clf_name == "Gradient Boosting":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50,value=100)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5)
        L = st.sidebar.selectbox("Loss", ('deviance', 'exponential'))
        M = st.sidebar.slider("max_depth",2,20)
        params["N"] = N
        params["LR"] = LR
        params["L"] = L
        params["M"] = M

    elif clf_name == "XGBoost":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5,value=0.1)
        O = st.sidebar.selectbox("Objective", ('binary:logistic','reg:logistic','reg:squarederror',"reg:gamma"))
        M = st.sidebar.slider("max_depth", 1, 20,value=6)
        G = st.sidebar.slider("Gamma",0,10,value=5)
        L = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
        A = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
        CS = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)
        params["N"] = N
        params["LR"] = LR
        params["O"] = O
        params["M"] = M
        params["G"] = G
        params["L"] = L
        params["A"] = A
        params["CS"] = CS

    RS=st.sidebar.slider("Random State",0,100)
    params["RS"] = RS
    return params

params = add_parameter_ui(classifier_select)








