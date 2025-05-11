import lime
import lime.lime_tabular
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Загрузка набора данных ирисов
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

# Обучение логистической регрессии
logistic_model = LogisticRegression(solver='liblinear')
logistic_model.fit(X_train, y_train)

# Создание объекта LIME для табличных данных
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

# Выбор примера из тестового набора для интерпретации
i = 25  # пример для интерпретации
exp = explainer.explain_instance(X_test[i], logistic_model.predict_proba, num_features=2)

print('Interpretation:\n', exp.as_list())  # exp.as_list()