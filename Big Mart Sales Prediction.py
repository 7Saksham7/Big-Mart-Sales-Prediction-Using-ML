#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install evalml')
get_ipython().system('pip install pandas-profiling')


# In[2]:


import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import evalml
from evalml import AutoMLSearch
import logging
import os


# In[4]:


train_data=pd.read_csv("Train.csv")
test_data=pd.read_csv("Test.csv")


# In[5]:


train_data.head()


# In[6]:


train_data.describe()


# In[8]:


train_data.info()


# In[9]:


#join train and test dataset
#create source column having train and test values to perform train test split later
train_data["source"]="train"
test_data["source"]="test"

data=pd.concat([train_data,test_data],ignore_index=True)


# In[10]:


print("Train dataset shape-",train_data.shape)
print("Test dataset shape-",test_data.shape)
print("Combine dataset shape-",data.shape)


# In[11]:


data.head()


# In[12]:


data.tail()


# In[13]:


#Number of Null values in Dataset
data.isnull().sum()


# In[14]:


#Percentage of null values in dataset
data.isnull().sum()/data.shape[0]*100


# In[15]:


sns.boxplot(data=data["Item_Weight"],orient="h")


# In[16]:


data["Item_Identifier"].unique()


# In[17]:


item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')


# In[18]:


item_avg_weight


# In[19]:


def compute_weights(cols):
    weight=cols[0]
    identifier=cols[1]
    
    if pd.isnull(weight):
        return item_avg_weight["Item_Weight"][item_avg_weight.index==identifier]
    else:
        return weight


# In[20]:


# Filling null values of Item_Weight with mean
data["Item_Weight"]=data[["Item_Weight","Item_Identifier"]].apply(compute_weights,axis=1).astype(float)


# In[21]:


data["Item_Weight"].isnull().sum()


# In[22]:


outlet_size_mode=data.pivot_table(values="Outlet_Size",columns="Outlet_Type",aggfunc=lambda x:x.mode())
outlet_size_mode


# In[23]:


#Define Function to compute outlet size

def compute_size_mode(cols):
    size=cols[0]
    Type=cols[1]
    
    if pd.isnull(size):
        return outlet_size_mode.loc["Outlet_Size"][outlet_size_mode.columns==Type][0]
    else:
        return size


# In[24]:


#Fill null values with mode

data["Outlet_Size"]=data[["Outlet_Size","Outlet_Type"]].apply(compute_size_mode,axis=1)


# In[25]:


data["Outlet_Size"].isnull().sum()


# In[26]:


visibility_item_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
visibility_item_avg


# In[27]:


# def function to compute mean value of item_Visibility according to Item_Itentifier

def compute_visibility_mean(cols):
    visibility=cols[0]
    identifier=cols[1]
    
    if visibility==0:
        return visibility_item_avg["Item_Visibility"][visibility_item_avg.index==identifier]
    else:
        return visibility


# In[28]:


#Compute visibilty values
data["Item_Visibility"]=data[["Item_Visibility","Item_Identifier"]].apply(
                            compute_visibility_mean,axis=1).astype(float)


# In[29]:


sum(data.Item_Visibility==0)


# In[30]:


data["Outlet_Years"]=2013-data["Outlet_Establishment_Year"]


# In[31]:


data["Outlet_Years"].describe()


# In[32]:


data["Item_Type_Combined"]=data["Item_Identifier"].apply(lambda x:x[0:2])


# In[33]:


data["Item_Type_Combined"]=data["Item_Type_Combined"].map({"FD":"Food","NC":"Non-Consumable","DR":"Drinks"})


# In[34]:


data["Item_Type_Combined"].value_counts()


# In[35]:


data["Item_Fat_Content"]=data["Item_Fat_Content"].replace({"LF":"Low Fat","low fat":"Low Fat","reg":"Regular"})


# In[36]:


data["Item_Fat_Content"].value_counts()


# In[37]:


data.loc[data["Item_Type_Combined"]=="Non-Consumable","Item_Fat_Content"]="Non-Edible"


# In[38]:


data["Item_Fat_Content"].value_counts()


# In[39]:


item_visibility_func=lambda x: x["Item_Visibility"]/visibility_item_avg["Item_Visibility"][visibility_item_avg.index==x["Item_Identifier"]][0]
data["Item_Visibility_MeanRatio"]=data.apply(item_visibility_func,axis=1).astype(float)


# In[40]:


data.head()


# In[41]:


data["Item_Visibility_MeanRatio"].describe()


# In[42]:


le=LabelEncoder()


# In[43]:


data.columns


# In[44]:


#Apply label Encoding to categorical variables


# In[45]:


data.head()


# In[46]:


cat_var=["Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Type_Combined"]
for i in cat_var:
    data[i]=le.fit_transform(data[i])


# In[47]:


data.head()


# In[48]:


"""
Item_Fat_Content-
                Low Fat=0
                Non-Edible=1
                Regular=2
Outlet_Size-
            High=0
            Medium=1
            Small=2

Outlet_Location_Type-
                    Tier 1=0
                    Tier 2=1
                    Tier 3=2
                    
Outlet_Type-
            Grocery Store=0
            Supermarket Type1=1
            Supermarket Type2=2
            Supermarket Type3=3

Item_Type_Combined-
                Drinks=0
                Food=1
                Non-Consumable=2
                
"""
pass


# In[49]:


data=pd.get_dummies(data,columns=["Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Type_Combined"])


# In[50]:


data.dtypes


# In[51]:


data.drop(["Item_Identifier","Item_Type","Outlet_Identifier","Outlet_Establishment_Year"],axis=1,inplace=True)


# In[52]:


train_df=data.loc[data["source"]=="train"]
test_df=data.loc[data["source"]=="test"]


# In[53]:


train_df.drop(["source"],axis=1,inplace=True)
test_df.drop(["source","Item_Outlet_Sales"],axis=1,inplace=True)


# In[54]:


train_df.head()


# In[55]:


test_df.head()


# In[57]:


train_df.to_csv("train_modified.csv",index=False)
test_df.to_csv("test_modified.csv",index=False)


# In[58]:


profile=ProfileReport(train_df,title="Stores Sales Analysis",explorative=True)


# In[59]:


profile.to_file("EDA Report.html")


# In[60]:


profile.to_notebook_iframe()


# In[62]:


train_data_mod=pd.read_csv("train_modified.csv")


# In[63]:


x_train_mod=train_data_mod.drop(["Item_Outlet_Sales"],axis=1)
y_train_mod=train_data_mod["Item_Outlet_Sales"]


# In[64]:


x_train_mod.head()


# In[65]:


y_train_mod.head()


# In[66]:


automl = AutoMLSearch(X_train=x_train_mod,y_train=y_train_mod,problem_type="regression",objective="root mean squared error",optimize_thresholds=True)
automl.search()


# In[67]:


automl.rankings


# In[68]:


automl.best_pipeline


# In[69]:


best_pipeline=automl.best_pipeline


# In[70]:


automl.describe_pipeline(automl.rankings.iloc[0]["id"])


# In[71]:


best_pipeline.score(x_train_mod,y_train_mod,objectives=["root mean squared error","mse","mae","r2"])


# In[73]:


# Create the Handler for logging records/messages to a file
file_handler = logging.FileHandler("log_file.log")


# In[74]:


#set the format of the log records and the logging level to DEBUG
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


# In[75]:


# a function  to create and save logs in the log files
def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


# In[76]:


logger = log(path=".",file="log_file.log")


# In[77]:


logger.info("Start data training with random forest model")


# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


x_train,x_test,y_train,y_test=train_test_split(x_train_mod,y_train_mod,random_state=101,test_size=0.2)


# In[81]:


from sklearn.ensemble import RandomForestRegressor


# In[82]:


reg_rf = RandomForestRegressor()
reg_rf.fit(x_train, y_train)
logger.info("Train {}".format("RandomForestRegressor"))


# In[83]:


reg_rf.score(x_test, y_test)


# In[84]:


y_pred=reg_rf.predict(x_test)


# In[85]:


sns.histplot(y_test-y_pred,kde=True)
plt.show()


# In[86]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[87]:


from sklearn import metrics


# In[97]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
logger.info("MAE: {}\n\t\t\t\t\t\t\t   MSE:{}\n\t\t\t\t\t\t\t   RMSE:{}".format(metrics.mean_absolute_error(y_test, y_pred),metrics.mean_squared_error(y_test, y_pred),np.sqrt(metrics.mean_squared_error(y_test, y_pred))))


# In[93]:


from sklearn.model_selection import RandomizedSearchCV


# In[94]:


logger.info("-"*50)
logger.info("Start Hyper-Parameter Tuning")


# In[95]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[96]:


# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[98]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[99]:


rf_random.fit(x_train,y_train)


# In[100]:


logger.info(rf_random.fit(x_train,y_train))


# In[101]:


rf_random.best_params_


# In[102]:


logger.info("Best Paramteres:{}".format(rf_random.best_params_))


# In[103]:


prediction = rf_random.predict(x_test)


# In[104]:


metrics.r2_score(y_test,prediction)


# In[105]:


plt.figure(figsize = (8,8))
sns.histplot(y_test-prediction,kde=True)
plt.show()


# In[106]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[107]:


print('R2:', metrics.r2_score(y_test,prediction))
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
logger.info("R2: {}\n\t\t\t\t\t\t\t   MAE: {}\n\t\t\t\t\t\t\t   MSE:{}\n\t\t\t\t\t\t\t   RMSE:{}".format(metrics.r2_score(y_test,prediction),metrics.mean_absolute_error(y_test, prediction),metrics.mean_squared_error(y_test, prediction),np.sqrt(metrics.mean_squared_error(y_test, prediction))))


# In[108]:


logger.info("-"*50)


# In[109]:


import pickle
# open a file, where you want to store the data
file = open('rf_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)
file.close()


# In[110]:


with open("rf_model.pkl","rb") as file1:
    model=pickle.load(file1)


# In[111]:


test_df.head()


# In[112]:


test_df['Item_Outlet_Sales'] = (model.predict(test_df)).tolist()


# In[113]:


test_df.head()


# In[114]:


#Save Output dataframe in csv file
test_df.to_csv("Test_Output.csv",index=False)


# In[115]:


submission=pd.concat([test_data["Item_Identifier"],test_data["Outlet_Identifier"],pd.DataFrame(test_df["Item_Outlet_Sales"].tolist(),columns=["Item_Outlet_Sales"])],axis=True)


# In[116]:


submission.head()


# In[117]:


submission.to_csv("Submission.csv",index=False)


# In[118]:


# END


# In[ ]:




