import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt


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

dataset_select=st.sidebar.selectbox("Select Dataset: ",("Fatal Health Data","Some Else"))
classifier_select = st.sidebar.selectbox("Select ML Classifier: ", ("Logistic Regression","KNN","SVM","Decision Trees",
                                                              "Random Forest","Gradient Boosting","XGBoost"))

LE = LabelEncoder()

def get_dataset(dataset_select):
    if dataset_select == "Fatal Health Data":
        data=pd.read_csv("")