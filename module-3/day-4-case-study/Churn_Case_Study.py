#!/usr/bin/env python
# coding: utf-8

# # Churn Case Study
# 
# ## Context
# "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]
# 
# 
# <img src="https://images.pexels.com/photos/3078/home-dialer-siemens-telephone.jpg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" style="width:400px">
# 
# **Client**: Telco Company in the USA offering triple play (phone, internet and TV).
# 
# New competitor entered offering triple play, resulting in increased churn.
# 
# Want better way to spot potential churning customers and suggested actions what to do.

# ## **Assignment**
# 
# - Define the business problem
# - Determine which evaluation metric you find appropriate:
#    - accuracy
#    - precision
#    - recall
#    - f1 score
# - Determine which type of slice/segment/type of churn you are interested
# - Run "data prep code"
# - Use logistic regression to create 2-3 model specifications
#   - model 1 (vanilla model): uses cleaned data as is, find best cutoff using chosen metric
#   - model 2: create at least **2 new features** and add them to the model
#   - model 3 (if time, a 'reach' model): increase the LASSO penalty to decrease the feature set
# - Pick the "best" model and find the "best" threshold
# - Use "best" model to identify the drivers of churn in your segment analysis and make recommendations for the company
# - Each group will have 5 minutes to present their recommendations to the rest of the class. Make sure to share:
#    - segment you chose
#    - evaluation metric you chose based on the business problem
#    - evaluation metric of "best" model's threshold & threshold
#    - what drives churn and what are your recommendations
#    - **if you had more time** what would you work on?

# ## Data
# 
# <img src="https://images.pexels.com/photos/53621/calculator-calculation-insurance-finance-53621.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" style = "width:400px" >
# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
# 
# The data set includes information about:
# 
# - Customers who left within the last month – the column is called Churn
# - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# - Customer account information 
#      - how long they’ve been a customer (tenure is in months)
#      - contract, payment method, paperless billing, monthly charges, and total charges
#      - all "totals" are over the length of the contract
# - Demographic info about customers – gender, age range, and if they have partners and dependents
# - Usage
#     - information about their usage patterns
#     - again, usage totals are over length of contract

# ## Concept: Churn
# 
# #### Type of churn:
# 
# **Voluntary** – they left after contract was up
# 
# **Involuntary** – we fired them
# 
# **Early churn** – left early, broke contract
# 
# ### Churn is a survival problem:
# - Predicting who will churn next month is really hard
# - Predicting who may churn over next 3 months is easier
# 
# <img src = "./img/funnel.png" style="width:800px">
# 
# There are many reasons to churn &#8594; **feature engineering is king**

# ### Solutions need to be tied to root problems
# 
# <img src = "./img/solution.png" style="width:800px">

# ### Different solutions have different time frames
# 
# <img src = "./img/time.png" style="width:800px">

# ## Remember:
# 
# #### You will not be paid to create intricate models
# ### You will be paid to **Solve Problems**

# # Get Started!
# 
# ## Part 1: Business problem
# 
# #### End Users: Client: Telco Company
# 
# 
# #### True business problem: Identifying causes of churn in the customer "journey". 
# 
# 
# #### Context:
# 
# - **False negative**: Predict that a variable is not going to cause customer churn, when in fact it does. 
#     - **Outcome**: Missed opportunity to retain customer.
# - **False positive** Predict that a variable is going to cause customer churn, when in fact it does not. 
#     - **Outcome**: Offer incentive to stay to customers who are not at risk of churning.

# ## Part 2: Evaluation Metric
# Which metric (of the ones we've explore so far) would make sense to primarily use as we evaluate our models?
# 
# - Accuracy
# - Precision
# - **Recall**
# - F1-Score

# ## Part 3: Segment choice
# 
# - We are most interested in voluntary churn.

# ## Part 4: Data Prep Code

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression


# In[2]:


pd.set_option('display.max_columns',None)


# In[22]:


# Import pacakges
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load dataset
url_link = 'https://docs.google.com/spreadsheets/d/1TAWfdKnWYiCzKUeDyGL6NzIOv7AxFt_Sfzzax464_FQ/export?format=csv&gid=882919979'
telco = pd.read_csv(url_link)

# Drop nas
telco.dropna(inplace=True)

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(telco.drop(columns=['customerID','Churn']), np.where(telco.Churn =="Yes", 1, 0), test_size=0.33, random_state=42)

# Separate out numeric from categorical variables
cat_var = telco.select_dtypes(include='object')
cat_var.drop(columns=['customerID','Churn'], inplace = True)

num_var = telco.select_dtypes(exclude = 'object') 

# Encode categorical variables
ohc = OneHotEncoder(drop='first')
encoded_cat = ohc.fit_transform(X_train[cat_var.columns.tolist()]).toarray()

# Add feature names to encoded vars
encoded=pd.DataFrame(encoded_cat, columns=ohc.get_feature_names(cat_var.columns.tolist()))
encoded.reset_index(inplace=True, drop=True)
X_train.reset_index(inplace=True, drop=True)

# Reassemble entire training dataset
clean_X_train = pd.concat([X_train[num_var.columns.tolist()] , encoded], axis=1,  sort=False)
clean_X_train.shape

encoded_cat = ohc.transform(X_test[cat_var.columns.tolist()]).toarray()
# Add feature names to encoded vars
encoded=pd.DataFrame(encoded_cat, columns=ohc.get_feature_names(cat_var.columns.tolist()))
encoded.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
# Reassemble entire training dataset
clean_X_test = pd.concat([X_test[num_var.columns.tolist()] , encoded], axis=1,  sort=False)


# In[23]:


scaler = StandardScaler()


# In[24]:


X_train_scaled = scaler.fit_transform(clean_X_train)
X_test_scaled = scaler.transform(clean_X_test)


# ## Part 5: Create models

# In[25]:


import pandas as pd
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


logreg = LogisticRegression()


# In[27]:


vanilla_mod = logreg.fit(X_train_scaled, y_train)


# In[28]:


y_pred_train = logreg.predict(X_train_scaled)
y_pred_test = logreg.predict(X_test_scaled)


# In[29]:


metrics = {"Accuracy": accuracy_score,
           "Recall": recall_score,
           "Precision": precision_score,
           "F1-Score": f1_score}

for name, metric_function in metrics.items():
    print(f"{name}:"); print("="*len(name))
    print(f"TRAIN: {metric_function(y_train, y_pred_train):.4f}")
    print(f"TEST: {metric_function(y_test, y_pred_test):.4f}")
    print("*" * 15)


# In[30]:


logreg.predict_proba(X_test_scaled)


# In[ ]:





# ## Part 6: Pick model & find best threshold

# In[ ]:





# ## Part 7: What drives churn?

# In[ ]:





# ## Part 8: What are your recommendations?

# In[ ]:





# In[ ]:




