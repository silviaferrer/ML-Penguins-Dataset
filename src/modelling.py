from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.visualization import plot_confusion_matrix

def evaluate_model(model, X_train, y_train, X_test, y_test, path_output=None):

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    plot_confusion_matrix(y_test=y_test, y_pred=predicted,path_output=path_output)



