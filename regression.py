# a. Poser le modèle et estimer les paramètres avec descente de gradient

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
data = pd.read_csv(r"C:\Users\39\OneDrive\Desktop\2AGE\TP ML\airfoil_self_noise.txt", sep="\t", header=None)
X = data.iloc[:, :-1].values  # 5 premières colonnes
y = data.iloc[:, -1].values   # dernière colonne (valeur cible)

# Normalisation des données
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std

# Ajout du biais (colonne de 1)
X = np.c_[np.ones(X.shape[0]), X]

# Découpage apprentissage / test
n_train = int(0.8 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Fonction coût
def compute_cost(X, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum((X @ theta - y)**2)

# Descente de gradient
def gradient_descent(X, y, theta, alpha, n_iter):
    m = len(y)
    cost_history = []

    for _ in range(n_iter):
        grad = (1/m) * X.T @ (X @ theta - y)
        theta -= alpha * grad
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Initialisation
theta = np.zeros(X.shape[1])
alpha = 0.01
n_iter = 500

theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, n_iter)

# b. Afficher la fonction coût à chaque itération

plt.plot(cost_history)
plt.xlabel("Itérations")
plt.ylabel("Coût")
plt.title("Évolution du coût")
plt.grid(True)
plt.show()
# c. Tester différentes valeurs du pas d’apprentissage (alpha)
alphas = [0.001, 0.01, 0.1]
plt.figure()

for a in alphas:
    theta = np.zeros(X.shape[1])
    _, cost = gradient_descent(X_train, y_train, theta, a, n_iter)
    plt.plot(cost, label=f"alpha={a}")

plt.xlabel("Itérations")
plt.ylabel("Coût")
plt.legend()
plt.title("Comparaison des pas d’apprentissage")
plt.grid(True)
plt.show()

# d. Vérifier le résultat analytiquement (moindres carrés)

theta_analytique = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
print("Theta (descente de gradient):", theta)
print("Theta (analytique):", theta_analytique)

#e. Prédiction et évaluation
# Prédictions
y_train_pred = X_train @ theta
y_test_pred = X_test @ theta

# Performances
print("Train MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_test_pred))

plt.figure()
plt.plot(y_test[:100], label="Vraies valeurs")
plt.plot(y_test_pred[:100], label="Prédictions")
plt.legend()
plt.title("Prédiction vs Vraie sortie (test)")
plt.show()

