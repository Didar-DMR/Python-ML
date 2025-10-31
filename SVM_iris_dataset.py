
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()

x=pd.DataFrame(iris.data, columns=iris.feature_names)
y=pd.DataFrame(iris.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.svm import SVC

model = SVC(kernel= 'linear', random_state=0)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names = iris.target_names)

print(cm)
print(accuracy_score(y_test, y_pred))
print(report)
