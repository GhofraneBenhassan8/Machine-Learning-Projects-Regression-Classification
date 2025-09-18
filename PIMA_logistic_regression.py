import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Chargement des données/Visualisation des données
df = pd.read_csv("PIMA.csv")
print(df.head())
print(df.describe())
print(df.info())


sns.pairplot(df, hue='Outcome')
plt.show()

# Séparation en variables X et y
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Séparation en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Fonction de coût (log loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h)) / m
    return cost[0][0]

# Descente de gradient
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (X.T @ (h - y)) / m
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Ajout de la colonne de biais
X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
theta_init = np.zeros((X_train_bias.shape[1], 1))

# Entraînement
alpha = 0.1
iterations = 1000
theta_opt, cost_history = gradient_descent(X_train_bias, y_train, theta_init, alpha, iterations)

# Courbe de la fonction de coût
plt.plot(range(iterations), cost_history)
plt.xlabel("Itérations")
plt.ylabel("Coût")
plt.title("Évolution du coût")
plt.grid(True)
plt.show()

# Prédictions
def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

y_pred_train = predict(X_train_bias, theta_opt).astype(int)
y_pred_test = predict(X_test_bias, theta_opt).astype(int)

# Évaluation
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test))

conf_mat = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Non Diabétique", "Diabétique"], yticklabels=["Non Diabétique", "Diabétique"])
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.show()
