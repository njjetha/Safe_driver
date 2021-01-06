#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,accuracy_score
#from sklearn.model_selection import StratifiedKFold,GridSearchCV
#import missingno as mssno
#seed =45


# In[3]:


from sklearn.linear_model import LogisticRegressionCV


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[5]:


from sklearn.ensemble import RandomForestClassifier


# In[7]:


train=pd.read_csv("train.csv")


# In[8]:


test=pd.read_csv("test.csv")


# In[41]:


train=pd.read_csv("/content/drive/My Drive/Colab Notebooks/Safe Driver /train.csv")
test=pd.read_csv("/content/drive/My Drive/Colab Notebooks/Safe Driver /test.csv")


# # **EDA**

# In[10]:


train.shape


# In[43]:


test.shape


# In[9]:


train=train.dropna()#remove the row which has single missing values 


# In[45]:


train.isna().any()#no missing values in the dataset


# In[12]:


train.dtypes # dtype is either int64 or float64


# In[13]:


train.describe()


# In[14]:


(train==-1).sum()


# In[15]:


train['target'].value_counts()


# In[16]:


sns.barplot(x="target",y=train['target'].value_counts(),data=train)


# * In our trainig dataset there are 59 columns including the id and target.
# * Their is no missing value in the training dataset 
# * Features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc)
# * The names of the variables indicate certain properties: "Ind" is related to individual or driver, “reg” is related to region, “car” is related to car itself and “calc” is an calculated feature.’ Here we will refer to these properties as groups
# * In addition, feature names include the postfix **bin** to indicate binary features and **cat** to indicate categorical features. Features without these designations are either continuous or ordinal.
# *The target columns signifies whether or not a claim was filed for that policy holder.
# * The value -1 represent the missing value 
# 
# 

# In[10]:


target=train['target'].values


# In[11]:


#segregrating columns 
def groupFeatures(features):
    binary_features = []
    categorical_features = []
    numeric_features = []
    for col in features:
        if 'bin' in col:
            binary_features.append(col)
        elif 'cat' in col:
           categorical_features.append(col)
        elif 'id' in col or 'target' in col:
            continue
        else:
            numeric_features.append(col)
    return binary_features, categorical_features,numeric_features 


# In[12]:


feature_list = list(train.columns)
bin_feature, cat_feature, num_feature = groupFeatures(feature_list)
print("# of binary feature : ", len(bin_feature))
print("# of categorical feature : ", len(cat_feature))
print("# of other feature : ", len(num_feature))


# In[20]:


correlation=train[bin_feature].corr()
fig= plt.subplots(figsize=(20,20))
sns.heatmap(correlation, cmap=None, vmax=1.0, center=0, fmt='.2f',               square=True, linewidths=.5, annot=True, cbar_kws={'shrink':.75})
plt.show()


# In[21]:


y=['ps_ind_06_bin','ps_ind_07_bin','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin']
correlation=train[y].corr()
fig= plt.subplots(figsize=(5,5))
sns.heatmap(correlation, cmap=None, vmax=1.0, center=0, fmt='.2f',               square=True, linewidths=.5, annot=True, cbar_kws={'shrink':.75})
plt.show()


# In[22]:


correlation=train[num_feature].corr()
fig= plt.subplots(figsize=(20,20))
sns.heatmap(correlation, cmap=None, vmax=1.0, center=0, fmt='.2f',               square=True, linewidths=.5, annot=True, cbar_kws={'shrink':.75})
plt.show()


# In[23]:


x=['ps_car_12','ps_car_13','ps_reg_01','ps_reg_03','ps_reg_02','ps_car_15']
correlation=train[x].corr()
fig= plt.subplots(figsize=(5,5))
sns.heatmap(correlation, cmap=None, vmax=1.0, center=0, fmt='.2f',               square=True, linewidths=.5, annot=True, cbar_kws={'shrink':.75})
plt.show()


# In[ ]:


for col in cat_feature:
    plt.figure()
    # Calculate the percentage of target=1 per category value
    cat_perc = train[[col, 'target']].groupby([col],as_index=False).mean()
    cat_perc['target']=cat_perc['target']*100
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(x=col, y='target', data=cat_perc, order=cat_perc[col])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(col, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()


# In[ ]:


for col in num_feature:
    plt.figure()
    # Calculate the percentage of target=1 per numeric value
    num_perc = train[[col, 'target']].groupby([col],as_index=False).mean()
    num_perc['target']=num_perc['target']*100
    num_perc.sort_values(by='target', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(x=col, y='target', data=num_perc, order=num_perc[col])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(col, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()


# In[ ]:


for col in bin_feature:
    plt.figure()
    # Calculate the percentage of target=1 per binary value
    bin_perc = train[[col, 'target']].groupby([col],as_index=False).mean()
    bin_perc.sort_values(by='target', ascending=False, inplace=True)
    bin_perc['target']=bin_perc['target']*100
    
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(x=col, y='target', data=bin_perc, order=bin_perc[col])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(col, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show()


# In[ ]:


#For binary features having 'ind' 
plt.figure()
fig, ax = plt.subplots(3, 4)
fig.set_size_inches((20,10))
cnt = 0
for i in range(3) :
  for j in range(4) :
    if i==2 and j==3:
        break
    sns.countplot(x=bin_feature[cnt], data = train, ax=ax[i,j])
    cnt+=1
plt.show()


# In[ ]:


#For binary features having 'calc'
plt.figure()
fig, ax = plt.subplots(3, 2)
fig.set_size_inches((20,10))
cnt = 11
for i in range(3) :
  for j in range(2) :
    sns.countplot(x=bin_feature[cnt], data = train, ax=ax[i,j])
    cnt+=1
plt.show()


# Here we cant remove any columns as all the 'calc_bin' features contributing to our target variables. 

# In[ ]:


#For categorical features having 'ind'
fig, ax = plt.subplots(1,3)
fig.set_size_inches((20,10))
for i in range(3) :
  sns.countplot(x=cat_feature[i], data = train, ax=ax[i])
plt.show()


# In[ ]:


#For categorical features having 'car'
plt.figure()
fig, ax = plt.subplots(4,3)
fig.set_size_inches((20,10))
count=3
for i in range(4):
    for j in range(3):
      sns.countplot(x=cat_feature[i], data = train, ax=ax[i,j])
    count+=1
plt.show()


# In[ ]:


#For numeric features having 'ind'(indexing from 0 to 3)
fig, ax = plt.subplots(2,2)
fig.set_size_inches((16,8))
count = 0
for i in range(2) :
  for j in range(2) :
    sns.countplot(x = num_feature[cnt], data = train, ax = ax[i,j])
    count+=1
plt.show()


# In[ ]:


#For numeric features having 'reg'(indexing from 4 to 6)
fig, ax = plt.subplots(1,3)
fig.set_size_inches((20,5))
count = 4
for i in range(3) :
    sns.countplot(x = num_feature[cnt], data = train, ax = ax[i])
    count+=1
plt.show()


# In[ ]:


#For numeric features having 'car'(indexing from 7 to 11)
fig, ax = plt.subplots(3,2)
fig.set_size_inches((20,5))
count = 7
for i in range(3) :
    for j in range(2):
        if i==2 and j==1:
            break
        sns.countplot(x = num_feature[count], data = train, ax = ax[i,j])
        count+=1
plt.show()


# In[ ]:


#For numeric features having 'calc'(indexing from 12 to 25)
fig, ax = plt.subplots(5,3)
fig.set_size_inches((15,20))
count = 12
for i in range(5) :
    for j in range(3):
        if i==4 and j==2:
            break
        sns.countplot(x = num_feature[count], data = train, ax = ax[i,j])
        count+=1
plt.show()


# In[ ]:


from scipy.stats import skew 


# In[ ]:


for col in train.columns:
  print(col,"   ",skew(train[col]))


# In[ ]:


from scipy.stats import norm
for col in train.columns:
    plt.figure()
    sns.distplot(train[col],fit=norm) 
    plt.show()
    print(skew(train[col]))


# # **Data Cleaning**

# In[13]:


train_null=train
train_null = train_null.replace(-1, np.NaN)


# In[14]:


test_null=test
test_null=test_null.replace(-1,np.NaN)


# In[15]:


train_null = train_null.loc[:, train_null.isnull().any()]
train_null.isna().sum()


# In[16]:


test_null =test_null.loc[:, test_null.isnull().any()]
test_null.isna().sum()


# In[17]:


percent_missing = train_null.isnull().sum() * 100 / len(train_null)
trainnull= pd.DataFrame({'column_name': train_null.columns,'percent_missing': percent_missing})
trainnull.sort_values(by='percent_missing',inplace=True)


# In[18]:


trainnull


# In[52]:


sns.catplot(x="percent_missing",y="column_name",data=trainnull,kind="bar")


# * ps_car_03_cat and ps_car_05_cat have a large proportion of records with missing values.
# * ps_reg_03 (continuous) has missing values for 18% of all records. 
# * ps_car_11 (ordinal) has only 1 records with misisng values.
# * ps_car_14 (continuous) has missing values for 7% of all records. 

# In[19]:


col_drop=['ps_car_03_cat', 'ps_car_05_cat']
train.drop(col_drop, inplace=True, axis=1)
test.drop(col_drop, inplace=True, axis=1)


# In[20]:


mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')


# In[21]:


train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
train['ps_car_02_cat']=mode_imp.fit_transform(train[['ps_car_02_cat']]).ravel()
train['ps_car_01_cat']=mode_imp.fit_transform(train[['ps_car_01_cat']]).ravel()
train['ps_ind_04_cat']=mode_imp.fit_transform(train[['ps_ind_04_cat']]).ravel()
train['ps_ind_02_cat']=mode_imp.fit_transform(train[['ps_ind_02_cat']]).ravel()
train['ps_ind_05_cat']=mode_imp.fit_transform(train[['ps_ind_05_cat']]).ravel()
train['ps_car_07_cat']=mode_imp.fit_transform(train[['ps_car_07_cat']]).ravel()
train['ps_car_09_cat']=mode_imp.fit_transform(train[['ps_car_09_cat']]).ravel()


# In[22]:


test['ps_reg_03'] = mean_imp.fit_transform(test[['ps_reg_03']]).ravel()
test['ps_car_14'] = mean_imp.fit_transform(test[['ps_car_14']]).ravel()
test['ps_car_11'] = mode_imp.fit_transform(test[['ps_car_11']]).ravel()
test['ps_car_02_cat']=mode_imp.fit_transform(test[['ps_car_02_cat']]).ravel()
test['ps_car_01_cat']=mode_imp.fit_transform(test[['ps_car_01_cat']]).ravel()
test['ps_ind_04_cat']=mode_imp.fit_transform(test[['ps_ind_04_cat']]).ravel()
test['ps_ind_02_cat']=mode_imp.fit_transform(test[['ps_ind_02_cat']]).ravel()
test['ps_ind_05_cat']=mode_imp.fit_transform(test[['ps_ind_05_cat']]).ravel()
test['ps_car_07_cat']=mode_imp.fit_transform(test[['ps_car_07_cat']]).ravel()
test['ps_car_09_cat']=mode_imp.fit_transform(test[['ps_car_09_cat']]).ravel()
test['ps_car_12']=mean_imp.fit_transform(test[['ps_car_12']]).ravel()


# In[23]:


print("Train: ",train.shape)
print("Test: ",test.shape)


# In[24]:


cat_feature=['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_car_01_cat','ps_car_02_cat','ps_car_04_cat','ps_car_06_cat',
 'ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat']


# In[25]:


X_train = train.drop(['target','id'],axis=1)
Y_train= train['target']


# In[26]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train_res, Y_train_res = oversample.fit_resample(X_train, Y_train)


# In[27]:


def oneHotEncode_dataframe(df, features):
    for feature in features:
        temp_onehot_encoded = pd.get_dummies(df[feature])
        column_names = ["{}_{}".format(feature, x) for x in temp_onehot_encoded.columns]
        temp_onehot_encoded.columns = column_names
        df = df.drop(feature, axis=1)
        df = pd.concat([df, temp_onehot_encoded], axis=1)
    return df


# In[28]:


X_train_res = oneHotEncode_dataframe(X_train_res, cat_feature)


# In[29]:


#feature list don't contain the target ,id and all categorical features 
feature=bin_feature+num_feature
print(len(feature))
#print(len(cat_feature))


# In[30]:


#X_train = train.drop(['target','id'],axis=1)
#y_train= train['target']
X_out = test.drop(['id'],axis=1)
scaler=StandardScaler()


# In[31]:


X_train_res.loc[:, feature] = scaler.fit_transform(X_train_res[feature])
X_out = oneHotEncode_dataframe(X_out, cat_feature)
X_out.loc[:, feature] = scaler.fit_transform(X_out[feature])


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X_train_res, Y_train_res, train_size = 0.7)


# #Logistic Regression Model

# In[33]:


#num_trees = 50
#rf = RandomForestClassifier(n_estimators=num_trees, n_jobs=4, min_samples_split=10, verbose=2, oob_score = True)
rf=LogisticRegressionCV(max_iter=3000)
rf.fit(X=X_train, y=y_train)


# In[34]:


rf_predictions = rf.predict_proba(X_test)[:, 1]

print(roc_auc_score(y_test, rf_predictions))

# get accuracy score (predict the class)
rf_predictions_class = rf.predict(X_test)
print(accuracy_score(y_test, rf_predictions_class, normalize=True))


# In[35]:


print('Confusion matrix\n',confusion_matrix(y_test,rf_predictions_class))


# In[36]:


feature_names = list(X_test.columns.values)
feature_importances = list(rf.coef_[0])
feature_list = []

for i in range(0,len(feature_names)):
    feature_list.append((feature_names[i], feature_importances[i]))
    
feature_select25 = pd.DataFrame(sorted(feature_list, reverse=True, key=lambda x: x[1])[:10])
feature_select25.columns = ['feature_select', 'feature_importance']
print(feature_select25)


# In[37]:


X_train_select25 = X_train[feature_select25['feature_select']]
X_out_select25 = X_out[feature_select25['feature_select']]
X_test_select25 = X_test[feature_select25['feature_select']]


# In[38]:


lr = LogisticRegression()
lr.fit(X_train_res, Y_train_res)


# In[39]:


result = lr.predict_proba(X_out)
result


# In[40]:


id=test['id']
submit=pd.DataFrame({'id':id,'target':result[:,1]})
submit=submit[['id','target']]


# In[41]:


submit.to_csv("submit.csv", index = False)


# In[42]:


submit.head()


# ## Without SMOTE

# In[43]:


X_train = train.drop(['target','id'],axis=1)
Y_train= train['target']


# In[45]:


X_train_res = oneHotEncode_dataframe(X_train, cat_feature)


# In[46]:


X_out = test.drop(['id'],axis=1)
scaler=StandardScaler()


# In[48]:


X_train_res.loc[:, feature] = scaler.fit_transform(X_train_res[feature])
X_out = oneHotEncode_dataframe(X_out, cat_feature)
X_out.loc[:, feature] = scaler.fit_transform(X_out[feature])


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X_train_res, Y_train, train_size = 0.7)


# In[51]:


rf=LogisticRegressionCV(max_iter=3000)
rf.fit(X=X_train, y=y_train)


# In[52]:


rf_predictions = rf.predict_proba(X_test)[:, 1]

print(roc_auc_score(y_test, rf_predictions))

# get accuracy score (predict the class)
rf_predictions_class = rf.predict(X_test)
print(accuracy_score(y_test, rf_predictions_class, normalize=True))


# In[53]:


print('Confusion matrix\n',confusion_matrix(y_test,rf_predictions_class))


# In[56]:


lr = LogisticRegression()
lr.fit(X_train_res, Y_train)


# In[57]:


result = lr.predict_proba(X_out)
result


# In[58]:


id=test['id']
submit=pd.DataFrame({'id':id,'target':result[:,1]})
submit=submit[['id','target']]


# In[59]:


submit.to_csv("submit.csv", index = False)


# In[60]:


submit.head()


# In[ ]:




