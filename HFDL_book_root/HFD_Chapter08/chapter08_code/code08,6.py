# Example: Sentiment analysis ensemble:

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

model1 = LogisticRegression()
model2 = SVC(probability=True)
model3 = DecisionTreeClassifier()

ensemble_model = VotingClassifier(
    estimators=[('lr', model1), ('svm', model2), ('dt', model3)],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)
accuracy = ensemble_model.score(X_test, y_test)
print(f"Ensemble accuracy: {accuracy:.2f}")
