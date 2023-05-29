import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("training_data/train.csv")
print(train_data.head())

test_data = pd.read_csv("training_data/test.csv")
print(test_data.head())

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
output.to_csv("predicted_data/submission.csv", index=False)
print(output)
