
# #  Logistic Regression

# In[1]:


# importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1. Data Exploration

# In[2]:


df = pd.read_csv('Titanic_train.csv')
df.head()


# In[3]:


df.dtypes


# In[7]:


df.info()


# In[6]:


df.describe()


# In[8]:


# checking for null values in the df

df.isnull().sum()


# In[9]:


# deleting columns which have no use for prediction like "Name"

df.columns


# In[10]:


df1 = df.drop('Name', axis=1)
df1.columns


# In[12]:


df1.head()


# In[17]:


# deleting 'Ticket' column as well since there are seperate columns for ticket class and fare making ticket number useless
df1 = df1.drop('Ticket', axis=1)
df1.head()


# In[41]:


#grouping pclass and survived
survival_rates = df.groupby('Pclass')['Survived'].mean()
plt.bar(survival_rates.index, survival_rates.values)
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rates by Passenger Class')
plt.show()


# In[42]:


# survival based on sex
survived_by_sex = df.groupby(['Sex', 'Survived']).size().unstack()

survived_by_sex.plot(kind='bar', stacked=True)
plt.xlabel('Sex')
plt.ylabel('Number of Passengers')
plt.title('Survival by Sex')
plt.legend(title='Survived')
plt.show()


# In[47]:


#relatiuonship between age and fare

plt.scatter(df['Age'], df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Relationship Between Age and Fare')
plt.show()


# # 2. Data Preprocessing

# In[18]:


# checking for missing values
df1.isnull().sum()


# In[20]:


# dropping rows where embarked value is missing since its insignificant amount of rows

df1 = df1.dropna(subset=['Embarked'])
df1.isnull().sum()


# In[24]:


# exploring age column 

df1.describe()


# In[26]:


# plotting histogram for age column 

plt.hist(df1['Age'])
plt.xlabel('Age (years)')
plt.ylabel('Number of passengers')
plt.title('Distribution of Passenger Ages on the Titanic')
plt.grid(True)  # Add gridlines for better readability (optional)
plt.show()


# In[27]:


# filling missing values in age column with mean
df1['Age'].fillna(df['Age'].mean(), inplace=True)
df1.isnull().sum()


# In[28]:


# dropping cabin column as it wont influence outcome and also has a significant amount of missing values

df1 = df1.drop('Cabin', axis=1)
df1.head()


# In[29]:


df1.dtypes


# In[31]:


df1['Sex'].nunique()


# In[33]:


df2 = df1.copy()
df2['Age'] = df2['Age'].astype(int)
df2.dtypes


# In[34]:


df2['Fare'] = df2['Fare'].astype(int)
df2.dtypes


# In[38]:


df1['Embarked'].nunique()


# In[40]:


#Creating dummy variables for 'Embarked' and 'Sex'
df2 = pd.get_dummies(df2, columns=['Embarked', 'Sex'], drop_first=True)
df2.dtypes # Setting drop_first=True ensures that one category is dropped from each new dummy variable set to avoid multicollinearity issues in logistic regression models


# # 3. Model Building

# In[48]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[49]:


X = df2.drop('Survived', axis=1)
y = df2['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[50]:


model = LogisticRegression()
model.fit(X_train, y_train)


# # 4. Model Evaluation

# In[57]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve


# In[53]:


y_pred = model.predict(X_test)


# In[54]:


# Calculating accuracy, precision, recall, F1-score, and ROC-AUC score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC-ROC:", auc)


# In[56]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr, _ = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Titanic Survival Prediction')
plt.grid(True)
plt.legend()
plt.show()


# # 5. Interpretation

# In[63]:


coefficients = model.coef_
odds_ratios = np.exp(coefficients)
print(odds_ratios)


# Positive Coefficient: higher values of that feature are more likely to be associated with survival.
# Negative Coefficient: Higher values of that feature are likely linked to a lower chance of survival.

# # 6. Deployment with Streamlit

# In[65]:


import joblib
joblib.dump(model, 'logistic_regression_model.pkl')


# In[66]:


import streamlit as st
model = joblib.load('logistic_regression_model.pkl')


# In[67]:


# Function to make predictions
def predict_survival(Pclass, Age, SibSp, Parch, Fare, Sex, Embarked):
    data = np.zeros((1, 9))
    data[0, 0] = Pclass
    data[0, 1] = Age
    data[0, 2] = SibSp
    data[0, 3] = Parch
    data[0, 4] = Fare
    data[0, 5] = Sex 
    
    # Create the dummy variables for Embarked
    if Embarked == 0:  # 'C'
        data[0, 6] = 1
    elif Embarked == 1:  # 'Q'
        data[0, 7] = 1
    else:  # 'S'
        data[0, 8] = 1

    prediction = model.predict(data)
    return prediction[0]


# In[68]:
st.title('Titanic Survival Prediction')
st.header('Enter passenger details:')
Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Age = st.slider('Age', 0, 80, 30)
SibSp = st.slider('Number of Siblings/Spouses aboard', 0, 8, 0)
Parch = st.slider('Number of Parents/Children aboard', 0, 6, 0)
Fare = st.slider('Fare', 0, 500, 50)
Sex = st.selectbox('Sex', ['male', 'female'])
Embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])


# In[69]:


# Encode categorical features
Sex = 1 if Sex == 'male' else 0
Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]


# In[70]:
if st.button('Predict'):
    result = predict_survival(Pclass, Age, SibSp, Parch, Fare, Sex, Embarked)
    if result == 1:
        st.success('The passenger is likely to survive.')
    else:
        st.error('The passenger is not likely to survive.')

