import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

# object to process data and assign integers to attributes
label_encoder = preprocessing.LabelEncoder()

buying = label_encoder.fit_transform(list(data["buying"]))
maint = label_encoder.fit_transform(list(data["maint"]))
door = label_encoder.fit_transform(list(data["door"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
cls = label_encoder.fit_transform(list(data["class"]))

predict = "class"

# X is features, y is labels
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# splits data into training data and testing data (90/10 split)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# print(x_train, y_test)
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train,y_train)

prediction = model.score(x_test, y_test)
print(prediction)

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for i in range(len(x_test)):
    print("Data:", x_test[i], "\n   Actual:", y_test[i], "\n   Predicted:", predicted[i])






