#!/usr/bin/env python
# coding: utf-8

# ## Data Cleaning  

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:/Users/HP/Downloads/Telecom-Churn.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# # Null Value 

# ### 1) Mean Median & Mode

# ### 2) KNNImputer

# ### 3) Create one Machine Learning Model and Predict It 

# ### 4) Drop the null Values

# In[11]:


import seaborn as sns


# In[12]:


sns.heatmap(df.isnull(),cmap="YlOrBr")


# In[13]:


df[df["TotalCharges"].isnull()]


# In[14]:


df["Churn"].value_counts().plot(kind="pie",autopct="%0.3f")


# In[15]:


idx = df[df["TotalCharges"].isnull()].index


# In[16]:


df = df.drop(idx,axis=0)


# In[17]:


df = df.reset_index()


# In[18]:


df = df.drop("index",axis=1)


# In[19]:


df = df.drop("customerID",axis=1)


# In[20]:


df.duplicated().sum()


# In[21]:


df[df.duplicated()]


# In[22]:


idx = df[df.duplicated()].index


# In[23]:


df = df.drop(idx,axis=0).reset_index()


# In[24]:


df


# In[25]:


df = df.drop("index",axis=1)
df


# In[26]:


df.to_csv("clean.csv")
print("Model Saving")


# # Data Analysis & Processing 

# In[27]:


df = pd.read_csv("clean.csv")


# In[28]:


df


# In[29]:


df.Churn.value_counts().plot(kind="pie",autopct="%0.2f")


# In[30]:


df = df.drop("Unnamed: 0",axis=1)


# In[31]:


obj_col = df.loc[:, df.dtypes == object].columns
obj_col


# In[32]:


for i in obj_col:
    data = df[i].unique()
    print("Column:{} || {}".format(i,data))


# In[33]:


for i in obj_col:
    data = df[i].value_counts()
    print("Column: {} ||\n{}\n\n".format(i,data))


# # Feature Selection

# In[34]:


df


# In[35]:


from sklearn.preprocessing import LabelEncoder


# In[36]:


obj_col


# In[37]:


for i in obj_col:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])


# In[38]:


df.corr()


# In[39]:


df.corr()["Churn"]


# In[40]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[41]:


df.corr().sort_values("Churn")["Churn"]


# In[42]:


X = df.drop("Churn",axis=1)
y = df["Churn"]


# ## RFE 

# In[43]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

rfe = RFE(model, n_features_to_select=11) #Decison Tree
fit = rfe.fit(X,y)

print(fit.n_features_,fit.support_,fit.ranking_)


# In[44]:


df.columns


# In[45]:


f_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']


# In[46]:


data = pd.DataFrame({"Column":f_col,"Values":fit.ranking_}) 


# In[47]:


data.sort_values("Values")


# # Decision Tree Features Importance

# In[48]:


from sklearn.tree import DecisionTreeClassifier


# In[49]:


model = DecisionTreeClassifier()
model.fit(X,y)


# In[50]:


model.feature_importances_


# In[51]:


data = pd.DataFrame({"Column":f_col,"Values":model.feature_importances_})


# In[52]:


data.sort_values("Values")


# # Decision Tree:

# PhoneService 0.003198 OnlineBackup 0.009649 StreamingTV 0.010209 StreamingMovies 0.013739 Partner 0.015636 PaperlessBilling 0.015825

# # Correlation

# Partner -0.148670 InternetService -0.047169 StreamingMovies -0.036802 StreamingTV -0.034312 gender -0.008694 PhoneService 0.011072 MultipleLines 0.040181 PaymentMethod 0.107032

# # #RFE
# 
# PaperlessBilling	4
# Partner	5
# MultipleLines	6
# StreamingMovies	7
# StreamingTV	8
# PhoneService	9

# # Data Balancing : Oversampling And Underampling

# In[53]:


df = pd.read_csv('Clean.csv')


# In[54]:


df


# In[55]:


df = df.drop(["Unnamed: 0","PhoneService","StreamingMovies","StreamingTV","Partner","MultipleLines"],axis=1)


# In[56]:


df["Churn"].value_counts().plot(kind="pie",autopct="%0.2f")
plt.show()


# In[57]:


df["Churn"].value_counts().plot(kind="bar")


# In[58]:


df


# In[59]:


pip install imblearn


# In[60]:


from imblearn.over_sampling import RandomOverSampler


# In[61]:


X = df.drop("Churn",axis=1)
y=df['Churn']


# In[62]:


OverS = RandomOverSampler()
X_Over,Y_Over = OverS.fit_resample(X,y)


# In[63]:


Y_Over.shape


# In[64]:


Y_Over.value_counts().plot(kind="pie",autopct="%0.2f")
plt.show()


# In[65]:


X_Over["Churn"] = Y_Over


# In[66]:


X_Over.to_csv("final_data.csv")
print("Final dataset")


# # Model Evalution

# In[67]:


df = pd.read_csv("final_data.csv")
df.head()


# In[68]:


df = df.drop("Unnamed: 0",axis=1)
df.head()


# In[69]:


obj_col = df.loc[:,df.dtypes == object].columns
obj_col


# In[70]:


from sklearn.preprocessing import LabelEncoder
for i in obj_col:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])


# In[71]:


df.head()


# In[72]:


X = df.drop(["Churn"],axis=1)
y = df["Churn"]


# In[73]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(X)


# #  Logistic Regression

# In[74]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=0)


# In[75]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[76]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[77]:


y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)


# In[78]:


cm


# In[79]:


accuracy_score(y_test,y_pred)


# In[80]:


print(classification_report(y_test,y_pred))


# In[81]:


from sklearn.model_selection import KFold,cross_val_score


# In[82]:


fold = KFold(n_splits=10)
model = LogisticRegression()
results = cross_val_score(model,x,y,cv=fold)


# In[83]:


results.min(),results.max()


# In[84]:


results.mean()


# In[85]:


results.std()


# # Model Comparission: SVM,Random Forst,XGBoost,KNN,Logistic Regression

# In[86]:


pip install xgboost


# In[87]:


from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[88]:


models = []
models.append(('LR', LogisticRegression(max_iter=400)))
models.append(('KNN',KNeighborsClassifier()))
models.append(('RF',RandomForestClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
models.append(('XGboost',XGBClassifier()))


# In[89]:


models


# In[90]:


results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits = 10)
    cv_results = cross_val_score(model,x,y, cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)


# In[91]:


# Lest take final model random Forest With 90%a accuracy


# In[92]:


fold = KFold(n_splits=10)
model = RandomForestClassifier()
results = cross_val_score(model,x,y,cv=fold)


# In[93]:


results.min(),results.max()


# # Pipe Line 

# In[94]:


df = pd.read_csv("final_data.csv")


# In[95]:


df


# In[96]:


X = df.drop(["Churn","Unnamed: 0"],axis=1)
y =df["Churn"]


# In[97]:


X.info()


# In[98]:


pip install sklearn_pandas


# In[99]:


from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper


# In[100]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[101]:


obj_col = X.loc[:, X.dtypes == object].columns
obj_col


# In[102]:


m = DataFrameMapper([(["gender",LabelEncoder()]),(["Dependents",LabelEncoder()]),
                     (["InternetService",LabelEncoder()]),(["OnlineSecurity",LabelEncoder()]),
                     (["OnlineBackup",LabelEncoder()]),
                     (["DeviceProtection",LabelEncoder()]),(["TechSupport",LabelEncoder()]),
                     (["Contract",LabelEncoder()]),(["PaperlessBilling",LabelEncoder()]),
                     (["PaymentMethod",LabelEncoder()])
                ])


# In[103]:


classifier = []
classifier.append(("mapper",m))
classifier.append(('standardize', StandardScaler()))
classifier.append(("model",RandomForestClassifier(criterion= 'gini', n_estimators= 50)))


# In[104]:


classifier


# In[105]:


model = Pipeline(classifier)


# In[106]:


model


# In[107]:


model.fit(X,y)


# ## User Testing

# In[108]:


dict(df.iloc[3,:])


# In[109]:


data = {'gender': 'Male',
 'SeniorCitizen': 0,
 'Dependents': 'No',
 'tenure': 45,
 'InternetService': 'DSL',
 'OnlineSecurity': 'Yes',
 'OnlineBackup': 'No',
 'DeviceProtection': 'Yes',
 'TechSupport': 'Yes',
 'Contract': 'One year',
 'PaperlessBilling': 'No',
 'PaymentMethod': 'Bank transfer (automatic)',
 'MonthlyCharges': 42.3,
 'TotalCharges': 1840.75}


# In[110]:


new_user = pd.DataFrame(data,index=[0])
new_user


# In[111]:


model.predict(new_user)


# In[112]:


model


# In[113]:


import pickle


# In[114]:


with open (file="model.pkl",mode="wb") as f:
    pickle.dump(model,f)


# In[115]:


pip install streamlit


# In[116]:


import streamlit as st


# In[117]:


st.title("Welcome To Telecom Project")
st.sidebar.header('User Input Parameters')


# In[118]:


def user_input_features():
    gen = st.sidebar.selectbox("Gender",("Male","Female"))
    ss = st.sidebar.selectbox("SeniorCitizen",(0,1))
    dep = st.sidebar.selectbox("Dependents",("Yes","No"))
    ten = st.slider("Tenure",min_value=0,max_value=75,step=1)
    
    isr = st.sidebar.selectbox('InternetService',('DSL', 'Fiber optic' ,'No'))
    osr = st.sidebar.selectbox('OnlineSecurity',('No', 'Yes',"No internet service"))
    ob = st.sidebar.selectbox('OnlineBackup',('No', 'Yes',"No internet service"))
    dp = st.sidebar.selectbox('DeviceProtection',('No', 'Yes',"No internet service"))

    ts = st.sidebar.selectbox('TechSupport',('No', 'Yes',"No internet service"))
    
    cr = st.sidebar.selectbox('Contract',('Month-to-month', 'One year' ,'Two year'))
    
    pb = st.sidebar.selectbox('PaperlessBilling',('Yes','No'))
    pm = st.sidebar.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'))
    
    
    mc = st.sidebar.number_input("Insert the MonthlyCharges",min_value=10,max_value=1000,step=1)
    tc = st.sidebar.number_input("Insert TotalCharges",min_value=10,max_value=1000,step=1)
    
    
    new = {"gender":gen,
         'SeniorCitizen': ss,
         'Dependents':dep,
         'tenure': ten,
         'InternetService': isr,
         'OnlineSecurity': osr,
         'OnlineBackup': ob,
         'DeviceProtection': dp,
         'TechSupport': ts,
         'Contract': cr,
         'PaperlessBilling': pb,
         'PaymentMethod': pm,
         'MonthlyCharges': mc,
         'TotalCharges': tc,
            }
    features = pd.DataFrame(new,index = [0])
    return features 
    
df = user_input_features()
st.write(df)


import pickle
    
with open("model.pkl",mode="rb") as f:
    model = pickle.load(f)
    
st.write("Model loaded")

result = model.predict(df)

st.subheader('Predicted Result')

if result[0]=="No":
    st.write("Customer will not Churn")
    
else:
    st.write("Customer will Churn")


# In[ ]:




