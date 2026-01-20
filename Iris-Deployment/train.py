from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

#Train the model
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

#We added 'protocol=2' for cross-platform compatibility
joblib.dump(model, 'iris_model.pkl', protocol=2)
print("New model saved successfully!")