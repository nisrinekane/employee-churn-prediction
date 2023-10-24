import pandas as pd
from sklearn import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('HR-Employee-Attrition.csv')

# check first few rows:
# print(df.head())

# check the missing values:
# print(df.isnull().sum()) # no missing values

# encode binary categories:
label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transorm(df['Attrition'])
df['Over18'] = label_encoder.fit_transorm(df['Over18'])
df['OverTime'] = label_encoder.fit_transorm(df['OverTime'])

# one-hot encoding for multi category features:
df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'])

# split data into training and testing sets:
X = df.drop(['Attrition'], axis=1) # features
y = df['Attrition'] # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale so that one feature doesn't dominate the others:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)