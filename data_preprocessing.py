import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(filename):
    df = pd.read_csv(filename)

    # Encoding binary categories
    label_encoder = LabelEncoder()
    df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

    # Feature Engineering
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])
    avg_distance = df['DistanceFromHome'].mean()
    df['HighDistance'] = df['DistanceFromHome'].apply(lambda x: 1 if x > avg_distance else 0)

    # One-hot encoding for all categorical features
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_features + ['AgeGroup'])

    # Splitting the data
    X = df.drop(['Attrition'], axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    column_names = X_train.columns.tolist()

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, column_names

