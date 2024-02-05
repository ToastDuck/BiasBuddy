
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shap

shap.initjs()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# This dataframe contains gender, age and race which are all protected variables
data= pd.read_csv("Salary.csv", delimiter = ",")
data.head()

# data.dtypes
# If the column for gender has only two unique values, pass it into LLM, ask the user the question "why are there only two options". "Are you representing groups other than male and female?"
# Checking for all of the gender values
# WILL HAVE TO PARAMETERIZE EVERYTHING FOR THE APP
# genders = data['Gender'].unique()

# for gender in genders:
#     print(gender)
# if len(genders) <= 2:
#     print("Why only two genders? Are there groups you're missing?")
#     print("Your model may suffer from [insert type of bias here]")
    
# # Preprocessing for random forest
# Calculate the mean
# Adding the target column - Person is a 'target' if their salary is above the mean
# Calculate the salary mean 
# Rounded to nearest whole number
salary_mean = round(data['Salary'].mean())
# print("Salary Mean: ")
# print(salary_mean)

# Add target column to dataframe 
data['Target'] = np.where(data['Salary'] > salary_mean, 1, 0)

# data[data['Gender']=='Female']
# Making Gender Numerical
# ADD A COLUMN WHICH REPRESENTS GENDER NUMERICALLY
data['Num_Gender'] = np.where(data['Gender']=='Male', 1, 0)


# DROP ALL CATEGORICAL DATA
data = data.select_dtypes(exclude=['object'])

# DROP THE SALARY
data = data.drop('Salary', axis=1)

# the only 'categorical' data left is gender, which we encoded
# NOTE this doesnt work for dataframes with more genders 
data.head()

# # Splitting the Data

# Split the data into features (X) and target (y)

# TODO SHOULD TEST DROPPING SALARY FROM THE MODEL
# Our independent variables
X = data.drop('Target', axis=1)
# Our dependent variable
y = data['Target']

# Split the data into training and test sets
# Test set is 20% of the total dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X_train and X_test are train and test predictors??
one_hot_encoded_training_predictors = pd.get_dummies(X_train)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)

# %% [markdown]
# # Fitting and evaluating the model

# %% [code] {"execution":{"iopub.status.busy":"2024-02-03T19:13:08.429298Z","iopub.execute_input":"2024-02-03T19:13:08.429987Z","iopub.status.idle":"2024-02-03T19:13:08.943073Z","shell.execute_reply.started":"2024-02-03T19:13:08.429929Z","shell.execute_reply":"2024-02-03T19:13:08.941916Z"}}
# Create an instance of the random forest model and fit it to our training data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Making predictions based on model
y_pred = rf.predict(X_test)

# Checking the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# %% [markdown]
# # Setting up SHAP explainer

# %% [code] {"execution":{"iopub.status.busy":"2024-02-03T19:13:41.593739Z","iopub.execute_input":"2024-02-03T19:13:41.594209Z","iopub.status.idle":"2024-02-03T19:13:48.692363Z","shell.execute_reply.started":"2024-02-03T19:13:41.594175Z","shell.execute_reply":"2024-02-03T19:13:48.691166Z"}}
explainer = shap.Explainer(rf)
shap_values = explainer.shap_values(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2024-02-03T19:14:05.080250Z","iopub.execute_input":"2024-02-03T19:14:05.080682Z","iopub.status.idle":"2024-02-03T19:14:05.335552Z","shell.execute_reply.started":"2024-02-03T19:14:05.080648Z","shell.execute_reply":"2024-02-03T19:14:05.334174Z"}}
shap.summary_plot(shap_values, X_test)

# print(shap_values)

# print("What are these")
# print(np.abs(shap_values.values).mean(axis=0))

vals= np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X_train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])

# feature_importance.sort_values(by='feature_importance_vals',ascending=False,inplace=True)
# print(feature_importance.head())
# print(feature_importance.shape)
feature_importance.sort_values(by='feature_importance_vals',ascending=False,inplace=True)
# print(feature_importance.head())

data_sum = feature_importance.sum()
# print(data_sum.head())
total_importance = data_sum.iloc[1]
# # Convert SHAP values to percentages 
# ROUNDED TO TWO DECIMAL PLACE
feature_importance['percentages'] = round(feature_importance['feature_importance_vals'] / total_importance,2)

# print(feature_importance.head())

# Prompt generator:
feature_names = feature_importance['col_name'].values
feature_percentages = feature_importance['percentages'].values

# for feature in feature_names:
#     print(feature)

for i in range(len(feature_percentages)):
    feature_percentages[i] = int(feature_percentages[i] * 100)

# for percentage in feature_percentages:
#     print(percentage)

feature_importance_string = "Importance Scores: "
for i in range(len(feature_percentages)):
    feature_importance_string += f"{feature_names[i]}" + " " + f"{feature_percentages[i]}%, "


prompt = "A machine learning model ranks how important some features are to its predictions based on their 'importance score'. Use the following importance scores to determine whether or not the model is biased on the metric of gender. "
prompt += feature_importance_string
# print(prompt)

# Above is an example of parametarizing everything
# For the demo we might stick to the hardcoded prompt since I dont have time to change the feature names which is messing up the model
hardcoded_prompt = "A machine learning model ranks how important some features are to its predictions based on their 'importance score'. Use the following importance scores to determine whether or not the model is biased on the metric of gender.  Importance scores: years of experience = 59%, age = 18%, education level = 14%, gender = 6%, senior = 3%."
# print(hardcoded_prompt)
print("AAAAAAAAAAAAAAAAA")