# %% [markdown]
# # 0. Libraries

# %% [markdown]
# For this notebook, I use a wide range of libraries and packages, all of which can be accessed by clicking on this cell

# %%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import numpy as np
import statistics
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from scipy import stats


# %% [markdown]
# # 1. Exploring the dataset
# ## 1.1 dataset breakdown
# Here we have a dataset of biomarkers that are collected from blood of people that either (1) had diabetes, (2) had developed diabetes during the study, and (3) were entirely healthy. I'm tasked with finding some biomarkers that can help to highlight incident diabetes specifically. 
# 
# First, we will need to load the data and do some quick analysis to see the nature of the data. First we can load in 'test_project_data.csv' and inspect it. 

# %%
diabetes_mtb_full_data = pd.read_csv("test_project_data.csv")

# %% [markdown]
# Quite the dataset to download. Since the data is made up of folks that were enrolled with diabetes and folks that developed diabetes it would be good to get numbers on their representation in the data. 
# 
# It seems that if you are already prevalent then you are coded as 1 and then incident will be 0. 
# 
# If someone develops diabetes in the study it codes as prevalent 0 and then incident being 1.
# 
# Finally, not being prevalent and not being incident will mean 0 and then 0, indicating a healthy individual.
# 
# There is no way to code for 1 and 1, just wouldn't make sense in this case.

# %%
obs_with_na_inc_diabetes = diabetes_mtb_full_data["incident_diabetes"].isna().sum()
print("Observations with NaN data: {}".format(obs_with_na_inc_diabetes))

# %% [markdown]
# I wanted to make sure the capacity to know who was incident wasn't compromised by a lack of data. I simply took away the observations that had 'NaN'. These samples in the data are essentially useless to our ultimate purpose so best to just excise them from the data.

# %%
diabetes_mtb_inc_cleaned = diabetes_mtb_full_data.dropna(subset=['incident_diabetes','prevalent_diabetes','sex','BMI','age','diabetes_followup_time'])

obs_with_na_inc_diabetes = diabetes_mtb_inc_cleaned["incident_diabetes"].isna().sum()
print("Observations with NaN data: {}".format(obs_with_na_inc_diabetes))

# %% [markdown]
# ## 1.1.1 'status' variable 
# 
# Something that will help make life easier later on is the encoding of the class of observations as 'Incident', 'Prevalent', and 'Healthy' in a variable called 'status'. Based on the rules we defined earlier, we can create the new column by looping through each row and returning the appropriate class. 

# %%
# creating a new column based on different column rules 

def classify(row):
    if (row["prevalent_diabetes"] == 0) & (row["incident_diabetes"] == 1):
        return 'Incident'
    elif (row["prevalent_diabetes"] == 0) & (row["incident_diabetes"] == 0):
        return 'Healthy'
    else:
        return 'Prevalent'

diabetes_mtb_inc_cleaned["status"] =  diabetes_mtb_inc_cleaned.loc[:,["prevalent_diabetes","incident_diabetes"]].apply(classify, axis=1)

# %%
#### quick test
diabetes_mtb_inc_cleaned[diabetes_mtb_inc_cleaned["status"] == "Incident"]

# %% [markdown]
# ## 1.2 demographic breakdowns of the classes of observations
# Next, it would be good to have an idea of the covariates and how they might impact the data. Let's go one by one. 
# 
# ### 1.2.1 Sex 

# %%
sns.set(style='whitegrid')
 
sns.countplot(x="status",
                hue="sex",
                data=diabetes_mtb_inc_cleaned)

# %% [markdown]
# Already, we can see that there is an imbalance in the dataset. Healthy has way more observations comparatively to Incident and Prevalent

# %% [markdown]
# ### 1.2.2 BMI 

# %%
sns.set(style='whitegrid')
 
sns.violinplot(x="status",
                y="BMI",
                data=diabetes_mtb_inc_cleaned)

# %% [markdown]
# ### 1.2.3 Age 

# %%
sns.set(style='whitegrid')
 
sns.violinplot(x="status",
                y="age",
                data=diabetes_mtb_inc_cleaned)

# %%
sns.set(style='whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
 
sns.histplot(hue="status",
                x="diabetes_followup_time",
                data=diabetes_mtb_inc_cleaned[diabetes_mtb_inc_cleaned['status']=="Healthy"],
                ax=axes[0])


sns.histplot(hue="status",
                x="diabetes_followup_time",
                data=diabetes_mtb_inc_cleaned[diabetes_mtb_inc_cleaned['status']=="Incident"],
                ax=axes[1])


sns.histplot(hue="status",
                x="diabetes_followup_time",
                data=diabetes_mtb_inc_cleaned[diabetes_mtb_inc_cleaned['status']=="Prevalent"],
                ax=axes[2])

# Adjust layout
plt.tight_layout()
plt.show()

# %% [markdown]
# Nothing too out of the ordinary here as it seems from the plots of the demographic information. Interesting to see a slightly lower BMI for healthy observations, which makes intuitive sense since folks that develop diabetes probably have a higher body mass. 

# %% [markdown]
# ## 1.3 Data imputation

# %% [markdown]
# Before delving into analyzing the data, there was a note mentioned that some of the 'mtb' data might not have a reading. This can be due to the blood biomarker having a value that is below the level of detection. Once it is below the level of detection, there is now way of knowing the variabilitu int the concentration of the biomarker in the blood. I pondered how to salvage some of this data since NaNs would be not the best to have when doing downstream analyses. 
# 
# ### 1.3.1 Which biomarkers have the lowest detection levels
# I wanted to know how pervasive the LOD/censored data was for each putative biomarker. I looped through the columns, focusing only on the biomarkers and calculated how much of the data was censored. 

# %%
proportions = []
masked_data = []

for series_name, series in diabetes_mtb_inc_cleaned.items():
    if series_name[0:3] == "mtb":
        proportion_nan = series.isna().sum()/len(series)
        if proportion_nan > 0.25:
            masked_data.append(series_name)
        proportions.append(proportion_nan)

diabetes_mtb_inc_cleaned = diabetes_mtb_inc_cleaned.drop(columns=masked_data)
plt.hist(proportions)
plt.xlabel('% of Nan')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# Even though there are many columns that have NaN values, I will keep the biomarkers that have at least 75% of their data accounted for. I think it will save time when performing feature selection in the later parts of the notebook. 

# %% [markdown]
# # 2. Normalizing and Imputing Data
# So the mtb data seem to have a large variance in data just by the sight of it. one biomarker might have adrastically different variance than the others it will be almagated with in downstream applications, so it would behoove us to perform some feature scaling so that thedata all falls in a consistent range. 
# 
# We have a few options at our disposal, but I decided to use min-max feature scaling since NaN values will be turned into '0.0' after the transformation. I looped through all the biomarker in the dataset and replaced all Nan values in a biomarker with the lowest recorded value in that column. Then we apply the scaler to the column and update the dataframe. 
# 

# %%
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

mtb_columns = []
for column in diabetes_mtb_inc_cleaned.columns: 
    if column[0:3] == "mtb": 
        mtb_columns.append(column)

df_min_max_scaled = diabetes_mtb_inc_cleaned.loc[:,:]

for col in mtb_columns:
    df_min_max_scaled[col] = df_min_max_scaled[col].fillna(df_min_max_scaled[col].min())
    df_min_max_scaled[col] = scaler.fit_transform(df_min_max_scaled[[col]])



# %% [markdown]
# # 3. Which blood biomarkers are associated with incident diabetes?

# %% [markdown]
# ## 3.1 Undersampling the data
# My first question is simply to find the blood biomarkers that possess some type of effect on the development of incidnet diabetes. I can't answer the question easily if there is a large imbalance in the data, like what we currently have at the moment. To deal with this, I plan to resample from the dataset in a way where all the classes are equally represented. That means splitting up the data by Healthy, Incident, and Prevalent observations. Since Healthy is the clearly larger of the classes, I will undersample from that group. 

# %%
# status_map goes 0,1,2 because I want to discriminate between the 3 classes to do appropriate resampling 
status_map = {"Healthy": 0, "Incident": 1, "Prevalent":2}
df_diabetes_vs_healthy = df_min_max_scaled.loc[:,:]

df_diabetes_vs_healthy["status"] = df_min_max_scaled["status"].map(status_map)
df_diabetes_vs_healthy["status"]

df_majority = df_diabetes_vs_healthy[(df_diabetes_vs_healthy["status"]==0)] 
df_minority = df_diabetes_vs_healthy[(df_diabetes_vs_healthy["status"]==1)] 
df_minority_prev = df_diabetes_vs_healthy[(df_diabetes_vs_healthy["status"]==2)] 

df_majority_undersampled = resample(df_majority,replace=False,n_samples= len(df_minority["status"]), random_state=1234)  
df_sampled = pd.concat([df_majority_undersampled, df_minority,df_minority_prev])
df_sampled['status'] = [0 if x == 2 else x for x in df_sampled["status"]]

sns.set(style='whitegrid')
 
sns.countplot(x="status",
                hue="sex",
                data=df_sampled)


# %% [markdown]
# ## 3.2 ANOVA 
# My first initial take on finding the biomarkers that can classify Incident diabetes is to run an ANOVA test on each biomarker with covariates included and report the biomarkers with low p.values at the significance level of 0.05. ANOVA will essentially test that for the biomarker of interest, do the means and distributions of the biomarker concentrations differentiate Incident to the Control. 

# %%
anova_pval_table= []
print(df_sampled)
i = 0
for column in df_sampled: 
    if column[0:3] == "mtb":
        result = statsmodels.formula.api.ols("status ~ {} + BMI + sex + age + diabetes_followup_time ".format(column), data=df_sampled).fit()
        table = statsmodels.api.stats.anova_lm(result,typ=2)
        row = [column,table.loc[column,"PR(>F)"]]
        anova_pval_table.append(row)
        i += 1
        if i % 100 == 0:
            print("models are {}% done".format((i/len(df_sampled.columns))*100))


# %% [markdown]
# I wanted to also pair the p value data with a strength of association test for the biomarker and the target variable. Point Biserial correlation appeared to fulfill the need I had for a metric that compares continuous biomarker data to categorical incidental diabetes data. 

# %%
### Associations 
anova_results_healthy_diabetes = pd.DataFrame(anova_pval_table, columns=['biomarker','anova_pval'])

point_biserial = []
for marker in anova_results_healthy_diabetes["biomarker"]:
    r_pb, p_value = stats.pointbiserialr(df_sampled[marker], df_sampled['status'])
    point_biserial.append(r_pb)

anova_results_healthy_diabetes["point_biserial"] = point_biserial
anova_results_healthy_diabetes.to_csv("anova_results_healthy_diabetes.csv")


# %% [markdown]
# ## 3.3 Biomarker selection 

# %% [markdown]
# Below I'm showing the biomarkers that have a high statistical significance while also exhibiting a strong association to the target variable. 

# %%
anova_biomarkers = anova_results_healthy_diabetes[(anova_results_healthy_diabetes['anova_pval'] < 0.05/len(df_sampled.columns)) & ((anova_results_healthy_diabetes['point_biserial'] < -0.2) | (anova_results_healthy_diabetes['point_biserial'] > 0.2) )]
print(anova_biomarkers)

# %% [markdown]
# I am interested in how filtering the biomarkers in general might impact the type of discrimination of the data. I am utilizing Linear Discriminant Analysis as a Supervised method to separate the data based on the biomarkers that seem to have an affect. 

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

countor_val = []
for pval in list(np.arange(0.0001,0.1,0.001)):
    for biserial in [0.01,0.05,0.1,0.15,0.2]:

        anova_biomarkers = anova_results_healthy_diabetes[(anova_results_healthy_diabetes['anova_pval'] < pval) & ((anova_results_healthy_diabetes['point_biserial'] < -biserial) | (anova_results_healthy_diabetes['point_biserial'] > biserial) )]

        X = df_sampled[anova_biomarkers["biomarker"]]  # Features
        y = df_sampled['status']  # Target variable (class labels)

        # Step 1: Fit LDA model
        try:
            lda = LinearDiscriminantAnalysis(n_components=1)  # 2 components for 2D plot
            X_lda = lda.fit_transform(X, y)

            # Step 2: Create a DataFrame with LDA components and target variable
            df_lda = pd.DataFrame(X_lda, columns=['LD1'])
            df_lda['target'] = y

            mean_0 = statistics.mean(df_lda[df_lda['target'] == 0]['LD1'])
            mean_1 = statistics.mean(df_lda[df_lda['target'] == 1]['LD1'])

            # Calculate standard deviations
            std_0 = statistics.stdev(df_lda[df_lda['target'] == 0]['LD1'])
            std_1 = statistics.stdev(df_lda[df_lda['target'] == 1]['LD1'])

            # Calculate pooled standard deviation
            pooled_std = ((std_0 ** 2 + std_1 ** 2) / 2) ** 0.5

            # Calculate separability (Cohen's d)
            separability = abs(mean_0 - mean_1) / pooled_std
            #print(f'Separability (Cohen\'s d): {separability}')
            countor_val.append([pval,biserial,separability])
        except:
            continue

slot = pd.DataFrame(countor_val,columns=["pval","biserial","cohen_d"])
print(slot)


# %%
slot[slot['cohen_d'] == slot['cohen_d'].max()]

# %% [markdown]
# ## 3.4 Biomarker's impact 

# %% [markdown]
# By testing all the filterings using a double for loop we were able to find the most advantageous pairing. Now I want to visualize the impact

# %%
anova_biomarkers = anova_results_healthy_diabetes[(anova_results_healthy_diabetes['anova_pval'] < 0.0001) & ((anova_results_healthy_diabetes['point_biserial'] < -0.1) | (anova_results_healthy_diabetes['point_biserial'] > 0.1) )]
X = df_sampled[anova_biomarkers["biomarker"]] 
y = df_sampled['status']  

lda = LinearDiscriminantAnalysis(n_components=1)  
X_lda = lda.fit_transform(X, y)

df_lda = pd.DataFrame(X_lda, columns=['LD1'])
df_lda['target'] = y

plt.figure(figsize=(8, 6))
sns.histplot(df_lda, x='LD1', hue='target', palette='coolwarm', kde=True, element='step', stat='density', common_norm=False)
plt.title('LDA: 1D Linear Discriminant')
plt.xlabel('Linear Discriminant 1 (LD1)')
plt.grid(True)
plt.show()

# %% [markdown]
# Not the most drastic of separations. 

# %% [markdown]
# # 4 Alternative approach for biomarker discovery 
# ## 4.1 Logistic Regression
# 
# I wanted to try a different method of calculating p-values, through the use of using a logistic regression model instead of ANOVA to see if the model had an impact

# %%
logistic_pval_healthy_vs_diabetes = []
df_sampled['sex'] = [0 if x == 'male' else 1 for x in df_sampled['sex']]

for column in df_sampled.columns: 
    if column[0:3] == "mtb":
        logit_model = statsmodels.api.Logit(df_sampled['status'], df_sampled.loc[:,[column,"BMI","age","diabetes_followup_time",'sex']].astype(float))
        result = logit_model.fit()
        row = [column,result.pvalues.loc[column],result.params.loc[column]]
        logistic_pval_healthy_vs_diabetes.append(row)

# %%
logit_results_healthy_diabetes = pd.DataFrame(logistic_pval_healthy_vs_diabetes, columns=['biomarker','logistic_pval','coefficient'])
logit_results_healthy_diabetes.to_csv("logit_results_healthy_diabetes.csv")
logit_results_healthy_diabetes[logit_results_healthy_diabetes['logistic_pval'] < 0.05]

# %% [markdown]
# ## 4.2 Association Tests with the logistic regression
# Here I'm recalculating the point biserial correlation

# %%
### Associations 
from sklearn.metrics import roc_auc_score
from scipy import stats

point_biserial = []
for marker in logit_results_healthy_diabetes["biomarker"]:
    r_pb, p_value = stats.pointbiserialr(df_sampled.loc[:,marker], df_sampled['status'])
    point_biserial.append(r_pb)


# %%
logit_results_healthy_diabetes = pd.DataFrame(logistic_pval_healthy_vs_diabetes, columns=['biomarker','logistic_pval','coefficient'])
logit_results_healthy_diabetes["point_biserial"] = point_biserial
logit_results_healthy_diabetes.to_csv("logit_results_healthy_diabetes.csv")
logit_biomarkers = logit_results_healthy_diabetes[(logit_results_healthy_diabetes['logistic_pval'] < 0.05/10000) & ((logit_results_healthy_diabetes['point_biserial'] < -0.2) | (logit_results_healthy_diabetes['point_biserial'] > .2) )]
logit_liberal_biomarkers = logit_results_healthy_diabetes[logit_results_healthy_diabetes['logistic_pval'] < 0.05]
logit_results_healthy_diabetes[(logit_results_healthy_diabetes['logistic_pval'] < 0.05/10) & ((logit_results_healthy_diabetes['point_biserial'] < -0.05) | (logit_results_healthy_diabetes['point_biserial'] > 0.05) )].to_csv("logit_results_healthy_diabetes.csv")
print(logit_biomarkers)

# %% [markdown]
# ## 4.3 LDA for the logistic regression biomarkers

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

countor_val = []
for pval in list(np.arange(0.0001,0.1,0.001)):
    for biserial in [0.01,0.05,0.1,0.15,0.2]:

        logit_biomarkers = logit_results_healthy_diabetes[(logit_results_healthy_diabetes['logistic_pval'] < pval) & ((logit_results_healthy_diabetes['point_biserial'] < -biserial) | (logit_results_healthy_diabetes['point_biserial'] > biserial) )]

        X = df_sampled[logit_biomarkers["biomarker"]]  # Features
        y = df_sampled['status']  # Target variable (class labels)

        # Step 1: Fit LDA model
        try:
            lda = LinearDiscriminantAnalysis(n_components=1)  # 2 components for 2D plot
            X_lda = lda.fit_transform(X, y)

            # Step 2: Create a DataFrame with LDA components and target variable
            df_lda = pd.DataFrame(X_lda, columns=['LD1'])
            df_lda['target'] = y

            mean_0 = statistics.mean(df_lda[df_lda['target'] == 0]['LD1'])
            mean_1 = statistics.mean(df_lda[df_lda['target'] == 1]['LD1'])

            # Calculate standard deviations
            std_0 = statistics.stdev(df_lda[df_lda['target'] == 0]['LD1'])
            std_1 = statistics.stdev(df_lda[df_lda['target'] == 1]['LD1'])

            # Calculate pooled standard deviation
            pooled_std = ((std_0 ** 2 + std_1 ** 2) / 2) ** 0.5

            # Calculate separability (Cohen's d)
            separability = abs(mean_0 - mean_1) / pooled_std
            #print(f'Separability (Cohen\'s d): {separability}')
            countor_val.append([pval,biserial,separability])
        except:
            continue

slot = pd.DataFrame(countor_val,columns=["pval","biserial","cohen_d"])
print(slot)


# %%
slot[slot['cohen_d'] == slot['cohen_d'].max()]

# %%
logit_biomarkers = logit_results_healthy_diabetes[(logit_results_healthy_diabetes['logistic_pval'] < 0.0031) & ((logit_results_healthy_diabetes['point_biserial'] < -0.1) | (logit_results_healthy_diabetes['point_biserial'] > 0.1) )]
X = df_sampled[logit_biomarkers["biomarker"]] 
y = df_sampled['status']  

lda = LinearDiscriminantAnalysis(n_components=1)  
X_lda = lda.fit_transform(X, y)

df_lda = pd.DataFrame(X_lda, columns=['LD1'])
df_lda['target'] = y

plt.figure(figsize=(8, 6))
sns.histplot(df_lda, x='LD1', hue='target', palette='coolwarm', kde=True, element='step', stat='density', common_norm=False)
plt.title('LDA: 1D Linear Discriminant')
plt.xlabel('Linear Discriminant 1 (LD1)')
plt.grid(True)
plt.show()

# %% [markdown]
# It appears that the biomarkers, while impressive through p-value and association, seem to struggle in separating the data in terms of the Incident Diabetes patients. Printing the biomarkers I found from doing the logistic regression approach and the anova approach.

# %%
print(logit_biomarkers)
print(anova_biomarkers)

# %% [markdown]
# # 5 Using a prediction model to classify incident diabetes
# ## 5.1 Random Forest
# 
# For the relevant biomarkers, I am using the subset of biomarkers that were discovered in the ANOVA feature selection portion of the notebook. I split my data into training and testing datasets and then proceed to train a Random Forest model to classify Incident observations. 

# %%
X = df_sampled.loc[:,anova_biomarkers["biomarker"]]
y = df_sampled.loc[:,"status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# %% [markdown]
# I start off by doing a hyper parameter tuning of the model just to see which set of parameters would demonstrate the best precision as a model.

# %%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score

# Define the hyperparameter distributions
param_distributions = {
    'n_estimators': [100, 300, 500, 1000],            
    'max_depth': [10, 20, 30, None],                  
    'min_samples_split': [2, 5, 10],                  
    'min_samples_leaf': [1, 2, 4],                    
    'max_features': ['sqrt', 'log2', None],           
    'criterion': ['gini', 'entropy'],                 
    'class_weight': ['balanced', None]                
}

model = RandomForestClassifier(random_state=42)

scorer = make_scorer(precision_score, average='binary') 
random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_distributions, 
    n_iter=50,           
    scoring=scorer,      
    cv=3,                
    verbose=2,          
    random_state=42
)

# Fit the random search on training data
random_search.fit(X_train, y_train)

# Best model and parameters
print("Best parameters found: ", random_search.best_params_)
print("Best precision score: ", random_search.best_score_)

# %% [markdown]
# Here is a Confusion matrix highlighting the ability of the Random Forest model to predict the Incident Diabetes in the dataset.

# %%
# Train the Random Forest model
model = RandomForestClassifier(
    **random_search.best_params_
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Other', 'Incident'], yticklabels=['Other', 'Incident'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

# %%
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Assuming your model is already trained, and predictions are made
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report (Precision, Recall, F1-Score, Support)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Specificity (True Negative Rate)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)  # Recall is the same as sensitivity

print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity (Recall/TPR): {sensitivity:.2f}")
print(f"Specificity (TNR): {specificity:.2f}")

# %% [markdown]
# The model output suggests that the model is doing better than flipping a coin when choosing between Incident diabetes and not. However there is room for improvement, such as increasing both the Type I and II error the model produces from the biomarker features. 


