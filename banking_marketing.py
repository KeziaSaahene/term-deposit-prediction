# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

data = pd.read_csv('bank-additional-full.csv', sep= ';') 

# making a copy of data dataframe
df = data.copy()

# previewing the dataset
print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print(df.columns)


# exploratory data analysis

# checking for duplicates 
duplicate_count= df.duplicated().sum() # this counts the duplicated rows in the dataframe
print(f"There are {duplicate_count} duplicate rows")
df= df.drop_duplicates() # removing duplicate rows

# checking for missing values
print(df.isna().sum())


numerical_columns= ['age',  'duration', 'campaign', 'pdays',
       'previous']


# plotting numerical columns
for i, col in enumerate(numerical_columns):
    plt.subplot(4, 3, i+1)
    plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()

plt.show()

# finding outliers
outliers_dict = {}

for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_dict[col] = outliers.shape[0]

for col, count in outliers_dict.items():
    print(f"{col}: {count} outliers")


# dropping outliers for campaign column
# calculate interquartile range for campaign column
Q1 = df['campaign'].quantile(0.25)
Q3 = df['campaign'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# filter out outliers from campaign column
df = df[(df['campaign'] >= lower_bound) & (df['campaign'] <= upper_bound)]

print(df.describe())


# creating age groups
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 24, 34, 44, 54, 64, 100],
                         labels=['Young', 'Young Adult', 'Adult', 'Middle-aged', 'Senior', 'Retired'])

# plotting features by the target
# plotting subscription count by age group
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='age_group', hue='y', order=df['age_group'].value_counts().index)
plt.title("Term Deposit Subscription Count by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.legend(title="Subscribed")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# plotting subscription count by education level
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='education', hue='y', order=df['education'].value_counts().index)
plt.title("Subscription Count by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Number of Clients")
plt.legend(title="Subscribed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# plotting subscription count by marital status
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='marital', hue='y', order=df['marital'].value_counts().index)
plt.title("Subscription Count by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Number of Clients")
plt.legend(title="Subscribed")
plt.tight_layout()
plt.show()


# plotting subscription count by credit default status
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='default', hue='y', order=df['default'].value_counts().index)
plt.title("Subscription Count by Credit Default Status")
plt.xlabel("Default Status")
plt.ylabel("Number of Clients")
plt.legend(title="Subscribed")
plt.tight_layout()
plt.show()


# plotting subscription count by housing loan status
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='housing', hue='y', order=df['housing'].value_counts().index)
plt.title("Subscription Count by Housing Loan Status")
plt.xlabel("Housing Loan")
plt.ylabel("Number of Clients")
plt.legend(title="Subscribed")
plt.tight_layout()
plt.show()


# plotting subscription count by personal loan status
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='loan', hue='y', order=df['loan'].value_counts().index)
plt.title("Subscription Count by Personal Loan Status")
plt.xlabel("Personal Loan")
plt.ylabel("Number of Clients")
plt.legend(title="Subscribed")
plt.tight_layout()
plt.show()


# plotting subscription count by previous campaign outcome
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='poutcome', hue='y', order=df['poutcome'].value_counts().index)
plt.title("Subscription Count by Previous Campaign Outcome")
plt.xlabel("Previous Outcome")
plt.ylabel("Number of Clients")
plt.legend(title="Subscribed")
plt.tight_layout()
plt.show()


# label encoding
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome' , 'age_group']

# encoding target variable 
df['y'] = df['y'].map({'yes': 1, 'no': 0})

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())

# viewing data types
print("\nData Types:\n", df.dtypes)



# checking correlation with target variable y
correlation = df.corr()['y'].sort_values(ascending=False).drop('y')

plt.figure(figsize=(10,6))
sns.barplot(
    x=correlation.values,
    y=correlation.index,
    hue=correlation.index,
    palette='coolwarm',
    dodge=False,
    legend=False
)
plt.title("Correlation with y")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Features")
plt.tight_layout()  #this adjusts spacing automatically
plt.show()



# choosing features
X = df.drop(['y', 'loan', 'marital'], axis=1)   #features
y = df['y']    #target



# train/test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)


# Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

# evaluating Logistic Regression model
y_pred_logreg = log_reg.predict(X_test)
print("\n=== Logistic Regression Results ===")
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("ROC AUC Score:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1]))



# Random Forest model
# training Random Forest model
model = RandomForestClassifier(n_estimators = 100, random_state=42, class_weight= 'balanced')
model.fit(X_train, y_train)



# evaluating the Random Forest model
y_pred = model.predict(X_test)
print("\n=== Random Forest Results ===")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# adjusting prediction threshold to 0.3 for better recall
y_probs = model.predict_proba(X_test)[:, 1]
y_pred_thresh_03 = (y_probs >= 0.3).astype(int)

print("\n=== Random Forest (Threshold = 0.3) Results ===")
print("Classification Report:\n", classification_report(y_test, y_pred_thresh_03))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_thresh_03))
print("ROC AUC Score:", roc_auc_score(y_test, y_probs))