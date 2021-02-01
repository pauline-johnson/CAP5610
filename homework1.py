
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random as random
import numpy as np

train = pd.read_csv('train.csv')

# print(df.head())

# 11
# Dataframes with only survivors and only those who died
survived_df = train[train['Survived'] == 1]
died_df = train[train['Survived'] == 0]

# Plot of survived = 1 by age
f = plt.figure(1)
plt.hist(survived_df['Age'].dropna())
plt.title('Survivors by Age')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')

# Plot of survived = 0 by age
g = plt.figure(2)
plt.hist(died_df['Age'].dropna())
plt.title('Fatalities by Age')
plt.xlabel('Age')
plt.ylabel('Number of Fatalities')

# 12
# Plot of survived = 1 and pclass = 1 by age
survived_1_df = survived_df[survived_df['Pclass'] == 1]
survived_2_df = survived_df[survived_df['Pclass'] == 2]
survived_3_df = survived_df[survived_df['Pclass'] == 3]
died_1_df = died_df[died_df['Pclass'] == 1]
died_2_df = died_df[died_df['Pclass'] == 2]
died_3_df = died_df[died_df['Pclass'] == 3]

# Plot of survived = 1 and pclass = 1
f = plt.figure(3)
plt.hist(survived_1_df['Age'].dropna())
plt.title('Pclass = 1 | Survived = 1')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')

# Plot of survived = 1 and pclass = 2
g = plt.figure(4)
plt.hist(survived_2_df['Age'].dropna())
plt.title('Pclass = 2 | Survived = 1')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')

# Plot of survived = 1 and pclass = 3
h = plt.figure(5)
plt.hist(survived_3_df['Age'].dropna())
plt.title('Pclass = 3 | Survived = 1')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')

# Plot of survived = 0 and pclass = 1
i = plt.figure(6)
plt.hist(died_1_df['Age'].dropna())
plt.title('Pclass = 1 | Survived = 0')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')

# Plot of survived = 0 and pclass = 2
j = plt.figure(7)
plt.hist(died_2_df['Age'].dropna())
plt.title('Pclass = 2 | Survived = 0')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')

# Plot of survived = 0 and pclass = 3
k = plt.figure(8)
plt.hist(died_3_df['Age'].dropna())
plt.title('Pclass = 3 | Survived = 0')
plt.xlabel('Age')
plt.ylabel('Number of Survivors')

plt.show()

# 16 : Set female to 1 and male to 0 in Sex and rename column to Gender
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
train = train.rename(columns={'Sex': 'Gender'})

# 17 : Fill in missing ages using mean and standard deviation.
mean = train['Age'].mean()
stddev = train['Age'].std()
# print(train.isna().sum())
train['Age'] = train['Age'].apply(lambda x: random.uniform(mean-stddev, mean+stddev) if np.isnan(x) else x)
# print(train.isna().sum())

# 18 : Fill in embarkation port in training dataset with mode of that column
embarked_mode = train['Embarked'].mode()[0]
# print(train.isna().sum())
train['Embarked'] = train['Embarked'].fillna(embarked_mode)
# print(train.isna().sum())

# 19 : Fill in single missing fare in test data
test = pd.read_csv('test.csv')
# print(test['Fare'][152]) # check before filling in
fare_mode = test['Fare'].mode()[0]
test['Fare'] = test['Fare'].fillna(fare_mode)
# print(test['Fare'][152]) # check after filling in

# 20 : Fill in fares with ordinal values
# print(train['Fare'])
train['Fare'] = train['Fare'].apply(lambda x: 0 if -0.001 <= x < 7.91 else x)
train['Fare'] = train['Fare'].apply(lambda x: 1 if 7.91 <= x < 14.454 else x)
train['Fare'] = train['Fare'].apply(lambda x: 2 if 14.454 <= x < 31 else x)
train['Fare'] = train['Fare'].apply(lambda x: 3 if 31 <= x < 512.329 else x)
# print(train['Fare'])





#plt.show()


#f = plt.figure(1)
