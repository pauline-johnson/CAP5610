import pandas as pd
import random as random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Set female to 1 and male to 0 in Sex and rename column to Gender for
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
train = train.rename(columns={'Sex': 'Gender'})

# Fill in missing ages using mean and standard deviation.
mean = train['Age'].mean()
stddev = train['Age'].std()
train['Age'] = train['Age'].apply(lambda x: random.uniform(mean-stddev, mean+stddev) if np.isnan(x) else x)

# Fill in embarkation port in training dataset with mode of that column
embarked_mode = train['Embarked'].mode()[0]
train['Embarked'] = train['Embarked'].fillna(embarked_mode)

# Fill in single missing fare in test data
fare_mode = train['Fare'].mode()[0]
train['Fare'] = train['Fare'].fillna(fare_mode)

# 20 : Fill in fares with ordinal values
train['Fare'] = train['Fare'].apply(lambda x: 0 if -0.001 <= x < 7.91 else x)
train['Fare'] = train['Fare'].apply(lambda x: 1 if 7.91 <= x < 14.454 else x)
train['Fare'] = train['Fare'].apply(lambda x: 2 if 14.454 <= x < 31 else x)
train['Fare'] = train['Fare'].apply(lambda x: 3 if 31 <= x < 512.329 else x)

train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

X = train.iloc[:, 1:8].values
y = train.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svc_linear = SVC(kernel='linear', random_state=0, gamma='scale')
svc_linear.fit(X_train, y_train)

svc_poly = SVC(kernel='poly', random_state=0, gamma='scale')
svc_poly.fit(X_train, y_train)

svc_rbf = SVC(kernel='rbf', random_state=0, gamma='scale')
svc_rbf.fit(X_train, y_train)

prediction1 = svc_linear.predict(X_test)
print('linear kernel accuracy:', accuracy_score(y_test, prediction1))

prediction2 = svc_poly.predict(X_test)
print('polynomial kernel accuracy:', accuracy_score(y_test, prediction2))

prediction3 = svc_rbf.predict(X_test)
print('rbf kernel accuracy:', accuracy_score(y_test, prediction3))