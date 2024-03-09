#!/usr/bin/env python
# coding: utf-8

# I will do a case study on Obesity Data Set Obtained from Kaggle. The data contains information about 2112 Individuals and their habbits. 
# The goal is to create a predictive model which will predict the Obesity Level
# 
# The flow of the case study is as below :
# * Reading the data in python
# * Defining the problem statement
# * Identifying the Target variable
# * Looking at the distribution of Target variable
# * Basic Data exploration
# * Rejecting useless columns
# * Visual Exploratory Data Analysis for data distribution (Histogram and Barcharts)
# * Feature Selection based on data distribution
# * Outlier treatment
# * Missing Values treatment
# * Visual correlation analysis
# * Statistical correlation analysis (Feature Selection)
# * Converting data to numeric for ML
# * Sampling and K-fold cross validation
# * Trying multiple classification algorithms
# * Selecting the best Model
# * Deploying the best model in production
# 

# In[1]:


import warnings 
warnings.filterwarnings("ignore")


# In[2]:


# Reading the data in python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandoc
import nbconvert
ObesityData= pd.read_csv("C:/Users/RITWIK/OneDrive/Desktop/IVY/Ivy ML/Project/ObesityDataSet.csv")

# Deleting Duplicates
ObesityData=ObesityData.drop_duplicates()
ObesityData.shape


# In[3]:


ObesityData.head()


# In[4]:


ObesityData.info()


# # Defining The Problem Statement
# #### Whether a person is Obese or not
# 
# * Target Variable : Obese Data
# * Predictors " Gender, Smoke, Number of main meals etc
#     

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Creating Bar chart as the Target variable is Categorical
GroupedData = ObesityData.groupby("Obese Data").size()
GroupedData.plot(kind="bar", figsize=(4,3))


# ## Basic Data Exploration

# In[6]:


# Observing the summarized information of data
# Data types, Missing values based on number of non-null values Vs total rows etc.
# Remove those variables from data which have too many missing values (Missing Values > 30%)
# Remove Qualitative variables which cannot be used in Machine Learning
ObesityData.info()


# In[7]:


ObesityData.describe(include="all")


# In[8]:


# Finging unique values for each column
# TO understand which column is categorical and which one is Continuous
# Typically if the numer of unique values are < 20 then the variable is likely to be a category otherwise continuous
ObesityData.nunique()


# ## Basic Data Exploration
# Based on the basic exploration above, you can now create a simple report of the data, noting down your 
# observations regaring each column. Hence, creating a initial roadmap for further analysis. 
# 
# The selected columns in this step are not final, further study will be done and then a final list will be created
# 
# Age , Height, Weight,'Frequency of consumption of vegetables','Number of main meals ','Consumption of water daily','Physical activity frequency','Time using technology devices ' are <b>continous variable</b>.
# 
# Gender, family_history_with_overweight ,Obese Data,Frequent consumption of high caloric food , Consumption of food between meals , Smoke,
# Calories consumption monitoring , Consumption of Alcohol, MTRANS  are <b>categorical variable</b>. 
# 

# # Visual Exploratory Data Analysis
# * Categorical variables: Bar plot
# * Continuous variables: Histogram

# ### Visualize distribution of all the Categorical Predictor variables in the data using bar plots

# In[9]:


ObesityData.columns


# ## Plotting all the categorical columns in Box Plot

# In[ ]:





# In[10]:


cat = ObesityData.dtypes[ObesityData.dtypes=='object'].index
num = ObesityData.dtypes[ObesityData.dtypes!='object'].index


# In[11]:


cat= ["Gender"," Consumption of food between meals",'family_history_with_overweight','Frequent consumption of high caloric food','Smoke','Calories consumption monitoring','Consumption of Alcohol', 'MTRANS', 'Obese Data']


# In[12]:


#bivariate plot-political.knowledge
fig = plt.figure(figsize=(50,30))
c = 1
for i in cat:
    plt.subplot(3, 3, c)
    plt.title('{}, subplot: {}{}{}'.format(i, 3, 3, c))
    plt.xlabel(i)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #sns.barplot(el_df[i],el_df['age'])
    sns.countplot(x= ObesityData[i], data=ObesityData)
    plt.rcParams.update({'font.size': 20})
    c = c + 1

plt.show()


# ## Bar Charts Interpretation
# These bar charts represent the frequencies of each category in the Y-axis and the category names in the X-axis.
# 
# In the ideal bar chart each category has comparable frequency. Hence, there are enough rows for each category in the data for the ML algorithm to learn.
# 
# If there is a column which shows too skewed distribution where there is only one dominant bar and the other categories are present in very low numbers. These kind of columns may not be very helpful in machine learning. We confirm this in the correlation analysis section and take a final call to select or reject the column.
# 
# In this data, all the categorical columns except have satisfactory distribution to be considered for machine learning.
# 
# <b>Selected Categorical Variables</b>: All the categorical variables are selected for further analysis.

# ## Converting Target Variable into Numeric

# In[13]:


#ObesityData['Obese Data'].replace({'Normal_Weight':0, 'Overweight_Level_I':1,'Overweight_Level_II':2,'Obesity_Type_I':3,'Insufficient_Weight':4,'Obesity_Type_II':5,'Obesity_Type_III':6}, inplace=True)


# ### Visualize distribution of all the Continuous Predictor variables in the data using histograms

# In[14]:


#bivariate plot-political.knowledge using Seaborn lib
fig = plt.figure(figsize=(50,30))
c = 1
for i in num:
    plt.subplot(3, 3, c)
    plt.title('{}, subplot: {}{}{}'.format(i, 3, 3, c))
    plt.xlabel(i)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #sns.barplot(el_df[i],el_df['age'])
    sns.histplot(x= ObesityData[i], data=ObesityData)
    plt.rcParams.update({'font.size': 20})
    c = c + 1

plt.show()


# In[15]:


# Plotting histograms of multiple columns together
# Observe that ApplicantIncome and CoapplicantIncome has outliers
#ObesityData.hist(['Age','Height', 'Weight','Frequency of consumption of vegetables','Number of main meals ','Consumption of water daily','Physical activity frequency','Time using technology devices ' ], figsize=(25,10))


# ## Histogram Interpretation
# Histograms shows us the data distribution for a single continuous variable.
# 
# The X-axis shows the range of values and Y-axis represent the number of values in that range.
# 
# The ideal outcome for histogram is a bell curve or slightly skewed bell curve. If there is too much skewness, then outlier treatment should be done and the column should be re-examined, if that also does not solve the problem then only reject the column.

# In[16]:


# Checking Null Values
ObesityData.isnull().sum()


# # Feature Selection
# Now its time to finally choose the best columns(Features) which are correlated to the Target variable.
# This can be done directly by measuring the correlation values or ANOVA/Chi-Square tests. However, it is always helpful to visualize the relation between the Target variable and each of the predictors to get a better sense of data.
# 
# I have listed below the techniques used for visualizing relationship between two variables as well as measuring the strength statistically.

# ## Visual exploration of relationship between variables
# * Continuous Vs Continuous ---- Scatter Plot
# * Categorical Vs Continuous---- Box Plot
# * Categorical Vs Categorical---- Grouped Bar Plots
# 
# ## Statistical measurement of relationship strength between variables
# * Continuous Vs Continuous ---- Correlation matrix
# * Categorical Vs Continuous---- ANOVA test
# * Categorical Vs Categorical--- Chi-Square test

# In this case study the Target variable is categorical, hence below two scenarios will be present
# * Categorical Target Variable Vs Continuous Predictor
# * Categorical Target Variable Vs Categorical Predictor

# # Relationship exploration: Categorical Vs Continuous -- Box Plots
# When the target variable is Categorical and the predictor variable is Continuous we analyze the relation using bar plots/Boxplots and measure the strength of relation using Anova test

# In[17]:


cat


# In[18]:


num


# In[19]:


ContinuousColsList=['Age','Height', 'Weight','Frequency of consumption of vegetables','Number of main meals ','Consumption of water daily','Physical activity frequency','Time using technology devices ' ]


# In[30]:


#**Check for Box plots, Correlation plots for the continuous columns**
import matplotlib.pyplot as plt
import seaborn as sns
data_plot=ContinuousColsList
fig=plt.figure(figsize=(30,30))
for i in range(0,len(ContinuousColsList)):
    ax=fig.add_subplot(4,2,i+1)
    sns.boxplot(x=data_plot[i],y='Obese Data', data=ObesityData,showmeans=True, orient="h")
    ax.set_title(ContinuousColsList[i],color='Red')
    plt.grid()
plt.tight_layout()


# # Box-Plots interpretation
# <b>What should you look for in these box plots? </b>
# 
# These plots gives an idea about the data distribution of continuous predictor in the Y-axis for each of the category in the X-Axis.
# 
# If the distribution looks similar for each category(Boxes are in the same line), that means the the continuous variable has NO effect on the target variable. Hence, the variables are not correlated to each other
# 
# The other chart exhibit opposite characteristics. Means the the data distribution is different(the boxes are not in same line!) for each category of survival. It hints that these variables might be correlated with Survived.
# 
# We confirm this by looking at the results of ANOVA test below

# # Statistical Feature Selection (Categorical Vs Continuous) using ANOVA test
# Analysis of variance(ANOVA) is performed to check if there is any relationship between the given continuous and categorical variable
# * Assumption(H0): There is NO relation between the given variables (i.e. The average(mean) values of the numeric Predictor variable is same for all the groups in the categorical Target variable)
# * ANOVA Test result: Probability of H0 being true

# In[19]:


# Defining a function to find the statistical relationship with all the categorical variables
def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    
    print('##### ANOVA Results ##### \n')
    for predictor in ContinuousPredictorList:
        CategoryGroupLists=inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
    
    return(SelectedPredictors)


# In[20]:


# Calling the function to check which categorical variables are correlated with target
ContinuousVariables=['Age','Height', 'Weight','Frequency of consumption of vegetables','Number of main meals ','Consumption of water daily','Physical activity frequency','Time using technology devices ']
FunctionAnova(inpData=ObesityData, TargetVariable='Obese Data', ContinuousPredictorList=ContinuousVariables)


# # Relationship exploration: Categorical Vs Categorical -- Grouped Bar Charts
# When the target variable is Categorical and the predictor is also Categorical then we explore the correlation between them  visually using barplots and statistically using Chi-square test

# In[21]:


#bivariate plot-political.knowledge
fig = plt.figure(figsize=(50,30))
c = 1
for i in cat:
    plt.subplot(3, 3, c)
    plt.title('{}, subplot: {}{}{}'.format(i, 3, 3, c))
    plt.xlabel(i)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #sns.barplot(el_df[i],el_df['age'])
    sns.countplot(x= ObesityData[i], data=ObesityData, hue = "Obese Data")
    plt.rcParams.update({'font.size': 20})
    c = c + 1

plt.show()


# # Grouped Bar charts Interpretation
# <b>What to look for in these grouped bar charts?</b>
# 
# These grouped bar charts show the frequency in the Y-Axis and the category in the X-Axis. 
# If the ratio of bars is similar across all categories, then the two columns are not correlated. 
# 
# Here we can see all the categorical variables the bar chart is different. Hence the two columns are corellated to each other.
# We confirm this analysis in below section by using Chi-Square Tests.

# ### Statistical Feature Selection (Categorical Vs Categorical) using Chi-Square Test
# 
# Chi-Square test is conducted to check the correlation between two categorical variables
# 
# * Assumption(H0): The two columns are NOT related to each other
# * Result of Chi-Sq Test: The Probability of H0 being True

# In[22]:


# Writing a function to find the correlation of all categorical variables with the Target variable
def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    from scipy.stats import chi2_contingency
    
    # Creating an empty list of final selected predictors
    SelectedPredictors=[]

    for predictor in CategoricalVariablesList:
        CrossTabResult=pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)
        
        # If the ChiSq P-Value is <0.05, that means we reject H0
        if (ChiSqResult[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', ChiSqResult[1])        
            
    return(SelectedPredictors)


# In[23]:


CategoricalVariables=["Gender"," Consumption of food between meals",'family_history_with_overweight','Frequent consumption of high caloric food','Smoke','Calories consumption monitoring','Consumption of Alcohol', 'MTRANS']

# Calling the function
FunctionChisq(inpData=ObesityData, 
              TargetVariable='Obese Data',
              CategoricalVariablesList= CategoricalVariables)


# <b>Finally selected Categorical variables:</b>
# 
# ['Gender',
#  ' Consumption of food between meals',
#  'family_history_with_overweight',
#  'Frequent consumption of high caloric food',
#  'Smoke',
#  'Calories consumption monitoring',
#  'Consumption of Alcohol',
#  'MTRANS']

# # Selecting final predictors for Machine Learning
# Based on the above tests, selecting the final columns for machine learning.
# 
# For this Data, all columns are selected
# 

# In[24]:


ObesityData.columns


# In[25]:


SelectedColumns =['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'Frequent consumption of high caloric food',
       'Frequency of consumption of vegetables', 'Number of main meals ',
       ' Consumption of food between meals', 'Smoke',
       'Consumption of water daily', 'Calories consumption monitoring',
       'Physical activity frequency', 'Time using technology devices ',
       'Consumption of Alcohol', 'MTRANS']


# In[26]:


DataForML = ObesityData[SelectedColumns]
DataForML.head()


# ## Data Pre-processing for Machine Learning
# List of steps performed on predictor variables before data can be used for machine learning
# 1. Converting each Ordinal Categorical columns to numeric
# 2. Converting Binary nominal Categorical columns to numeric using 1/0 mapping
# 3. Converting all other nominal categorical columns to numeric using pd.get_dummies()
# 4. Data Transformation (Optional): Standardization/Normalization/log/sqrt. Important if you are using distance based algorithms like KNN, or Neural Networks

# ## Converting the binary nominal variable to numeric using 1/0 mapping

# In[27]:


# Converting the binary nominal variable sex to numeric
DataForML['Gender'].replace({'Female':0, 'Male':1}, inplace=True)
DataForML['family_history_with_overweight'].replace({'yes':0, 'no':1}, inplace=True)
DataForML['Frequent consumption of high caloric food'].replace({'yes':0, 'no':1}, inplace=True)
DataForML['Smoke'].replace({'yes':0, 'no':1}, inplace=True)
DataForML['Calories consumption monitoring'].replace({'yes':0, 'no':1}, inplace=True)


# ## Converting the nominal variable to numeric using get_dummies()¶

# In[28]:


DataForML.head()


# In[29]:


# Treating all the nominal variables at once using dummy variables
DataForML_Numeric=pd.get_dummies(DataForML)

# Adding Target Variable to the data
DataForML_Numeric['Obese Data']=ObesityData['Obese Data']

# Printing sample rows
DataForML_Numeric.head()


# ## Changing the target variables into numeric 

# In[30]:


print(DataForML_Numeric["Obese Data"].unique())


# In[18]:


DataForML_Numeric['Obese Data'].replace({'Normal_Weight':0, 'Overweight_Level_I':1,'Overweight_Level_II':2,'Obesity_Type_I':3,'Insufficient_Weight':4,'Obesity_Type_II':5,'Obesity_Type_III':6}, inplace=True)


# In[32]:


DataForML_Numeric.head()


# # Machine Learning: Splitting the data into Training and Testing sample
# We dont use the full data for creating the model. Some data is randomly selected and kept aside for checking how good the model is. This is known as Testing Data and the remaining data is called Training data on which the model is built. Typically 70% of data is used as Training data and the rest 30% is used as Tesing data.

# In[33]:


# Printing all the column names for our reference
DataForML_Numeric.columns


# In[34]:


# Separate Target Variable and Predictor Variables
TargetVariable='Obese Data'
Predictors=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
       'Frequent consumption of high caloric food',
       'Frequency of consumption of vegetables', 'Number of main meals ',
       'Smoke', 'Consumption of water daily',
       'Calories consumption monitoring', 'Physical activity frequency',
       'Time using technology devices ',
       ' Consumption of food between meals_Always',
       ' Consumption of food between meals_Frequently',
       ' Consumption of food between meals_Sometimes',
       ' Consumption of food between meals_no',
       'Consumption of Alcohol_Always', 'Consumption of Alcohol_Frequently',
       'Consumption of Alcohol_Sometimes', 'Consumption of Alcohol_no',
       'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
       'MTRANS_Public_Transportation', 'MTRANS_Walking']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
# Split X and Y into training and test set in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)


# In[35]:


# Sanity check for the sampled data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # Logistic Regression

# In[36]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
# choose parameter Penalty='l1' or C=1
# choose different values for solver 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
clf = LogisticRegression(C=5,penalty='l2', solver='newton-cg')

# Printing all the parameters of logistic regression
# print(clf)

# Creating the model on Training Data
LOG=clf.fit(X_train,y_train)

# Generating predictions on testing data
prediction=LOG.predict(X_test)
# Printing sample values of prediction in Testing data
TestingData=pd.DataFrame(data=X_test, columns=Predictors)
TestingData['Obese Data']=y_test
TestingData['Predicted_Obese Data']=prediction
print(TestingData.head())


# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(prediction, y_test))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(LOG, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# ## Decision Tree

# In[37]:


#Decision Trees
from sklearn import tree
#choose from different tunable hyper parameters
clf = tree.DecisionTreeClassifier(max_depth=4,criterion='entropy')

# Printing all the parameters of Decision Trees
print(clf)

# Creating the model on Training Data
DTree=clf.fit(X_train,y_train)
prediction=DTree.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(DTree.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(DTree, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# ## Plotting A Decision Tree

# In[38]:


# Load libraries
from IPython.display import Image
from sklearn import tree
import pydotplus

# Create DOT data
dot_data = tree.export_graphviz( DTree,out_file=None, 
                                feature_names=Predictors, class_names= True )

# printing the rules
#print(dot_data)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png(), width=3000,height=3000)
# Double click on the graph to zoom in


# # Random Forest

# In[39]:


# Random Forest (Bagging of multiple Decision Trees)
from sklearn.ensemble import RandomForestClassifier
# Choose different hyperparameter values of max_depth, n_estimators and criterion to tune the model
clf = RandomForestClassifier(max_depth=4, n_estimators=100,criterion='gini')

# Printing all the parameters of Random Forest
print(clf)

# Creating the model on Training Data
RF=clf.fit(X_train,y_train)
prediction=RF.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RF, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(RF.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# ### Plotting one of the Decision Trees in Random Forest

# In[40]:


# Plotting a single Decision Tree from Random Forest
# Load libraries
from IPython.display import Image
from sklearn import tree
import pydotplus

# Create DOT data for the 6th Decision Tree in Random Forest
dot_data = tree.export_graphviz(clf.estimators_[5] , out_file=None, feature_names=Predictors, class_names=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png(), width=3000,height=4000)
# Double click on the graph to zoom in


# ## Adaboost

# In[41]:


# Adaboost 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Choosing Decision Tree with 1 level as the weak learner
DTC=DecisionTreeClassifier(max_depth=4)
clf = AdaBoostClassifier(n_estimators=100, base_estimator=DTC ,learning_rate=0.01)

# Printing all the parameters of Adaboost
print(clf)

# Creating the model on Training Data
AB=clf.fit(X_train,y_train)
prediction=AB.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(AB, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(AB.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# ### Plotting one of the Decision trees from Adaboost

# In[42]:


# PLotting 5th single Decision Tree from Adaboost
# Load libraries
from IPython.display import Image
from sklearn import tree
import pydotplus

# Create DOT data for the 6th Decision Tree in Adaboost
dot_data = tree.export_graphviz(clf.estimators_[5] , out_file=None, feature_names=Predictors, class_names=True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png(), width=5000,height=5000)
# Double click on the graph to zoom in


# ## XGBOOST

# In[43]:


# Xtreme Gradient Boosting (XGBoost)
from xgboost import XGBClassifier
clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=200, objective='binary:logistic', booster='gbtree')

# Printing all the parameters of XGBoost
print(clf)

# Creating the model on Training Data
XGB=clf.fit(X_train,y_train)
prediction=XGB.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(XGB, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(XGB.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# ## Plotting a single decision tree out of XGBoost

# In[44]:


from xgboost import plot_tree
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(XGB, num_trees=10, ax=ax)


# # Standardization/Normalization of data
# You can choose not to run this step if you want to compare the resultant accuracy of this transformation with the accuracy of raw data. 
# 
# However, if you are using KNN or Neural Networks, then this step becomes necessary.

# In[45]:


### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## KNN

# In[46]:


# K-Nearest Neighbor(KNN)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)

# Printing all the parameters of KNN
print(clf)

# Creating the model on Training Data
KNN=clf.fit(X_train,y_train)
prediction=KNN.predict(X_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(KNN, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# Plotting the feature importance for Top 10 most important columns
# There is no built-in method to get feature importance in KNN


# # Deployment of the Model
# 
# Based on the above trials you select that algorithm which produces the best average accuracy. In this case, multiple algorithms have produced similar kind of average accuracy. Hence, we can choose any one of them. 
# 
# I am choosing <b>XGBoost</b> as the final model since it is very fast for this data!
# 
# In order to deploy the model we follow below steps
# 1. Train the model using 100% data available
# 2. Save the model as a serialized file which can be stored anywhere
# 3. Create a python function which gets integrated with front-end(Tableau/Java Website etc.) to take all the inputs and returns the prediction

# In[47]:


# Xtreme Gradient Boosting (XGBoost)
from xgboost import XGBClassifier
clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=200, objective='binary:logistic', booster='gbtree')

# Printing all the parameters of XGBoost
print(clf)

# Creating the model on Full Data
X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values
XGB=clf.fit(X,y)
prediction=XGB.predict(X)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y, prediction))
print(metrics.confusion_matrix(y, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))

# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(XGB, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))

# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(XGB.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# ## Ploting a Decision Tree with the whole data

# In[48]:


X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

print(X.shape)
print(y.shape)


# #### Step 1. Retraining the model using 100% data

# In[49]:


#Decision Trees
from sklearn import tree
#choose from different tunable hyper parameters
clf = tree.DecisionTreeClassifier(max_depth=4,criterion='entropy')

# Training the model on 100% Data available
FinalDecisionTreeModel=clf.fit(X,y)


# #### Cross validating the final model accuracy with less predictors

# In[50]:


# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(FinalDecisionTreeModel, X , y, cv=10, scoring='f1_weighted')
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# ## Insights from Data
# 
# * Obesity_Type III is maximum in Females and Obesity_Type II in Males.
# * People with Family History of Obesity has a higher chances of becoming obese.
# * People consuming high caloric food has a higher chances of becoming obese.
# * People who smoke and consumes food between meals has a higher chance of becoming obese.
# * People consuming Alcohol sometimes has higher chance of developing Obesity .
# * Number of people whose age is above 35 are obese.    
# * Factors which helps in reducing obesity are : Higher Physical Activity Frequency, Keeping number of mean meals less than 3 ,     Not smoking , Maintaining a proper diet and calory consumption monitoring .

# In[ ]:




