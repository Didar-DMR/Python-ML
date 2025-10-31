
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

td = pd.read_csv("Titanic-Dataset.csv")

x = td.drop(columns=["PassengerId", "Survived", "Cabin"]).values
y = td.iloc[:, 1].values

print(x[:5])
print(20*"----")
print(y[:5])

print("Missing values before imputation:")
print(td.isnull().sum())

numerical_cols = td.select_dtypes(include=np.number).columns
td[numerical_cols] = td[numerical_cols].fillna(td[numerical_cols].mean())

print("\nMissing values after imputation:")
print(td.isnull().sum())

mode_embarked = td['Embarked'].mode()[0]
td['Embarked'] = td['Embarked'].fillna(mode_embarked)

print("Missing values after handling 'Cabin' and 'Embarked':")
print(td.isnull().sum())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(handle_unknown='ignore'), [1, 2, 6, 8])], remainder="passthrough")
x = ct.fit_transform(x).toarray()

print(x)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30]
}

rf_classifier = RandomForestClassifier(random_state=0)

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(x_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

best_rf_model = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

y_pred_best_rf = best_rf_model.predict(x_test)

print("\nEvaluation of the best RandomForestClassifier on the test set:")
print("Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_best_rf))


#####################

classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(cm)
print(acc*100)
print(classification_report(y_test, y_pred))