import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('TITANIC.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

df= df.drop(columns='Cabin', axis=1)
df['Age']= df['Age'].fillna(df['Age'].mean())
df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum())
print(df.describe())
print(df['Survived'].value_counts())

sns.set()
sns.countplot(x= 'Survived', data= df)
plt.show()

print(df['Sex'].value_counts())
sns.countplot(x= 'Sex', data= df)
plt.show()
sns.countplot(x='Sex', hue= 'Survived', data= df)
plt.show()

print(df['Pclass'].value_counts())
sns.countplot(x= 'Pclass', data= df)
plt.show()
sns.countplot(x='Pclass', hue= 'Survived', data= df)
plt.show()

df= df.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}})

X= df.drop(columns=['PassengerId', 'Name', 'Ticket','Survived'], axis=1)
Y= df['Survived']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state= 2)
print(X.shape, X_train.shape, X_test.shape)

model= LogisticRegression(max_iter=175)
model.fit(X_train, Y_train)

X_train_prediction= model.predict(X_train)
training_accuracy= accuracy_score(Y_train, X_train_prediction)
print('Training Accuracy: ', training_accuracy*100)

X_test_prediction= model.predict(X_test)
test_accuracy= accuracy_score(Y_test, X_test_prediction)
print('Testing Accuracy: ', test_accuracy*100)