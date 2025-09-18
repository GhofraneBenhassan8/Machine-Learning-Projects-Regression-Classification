# importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

#chargement des données
data = pd.read_csv(r"C:\Users\39\OneDrive\Desktop\2AGE\TP ML\airfoil_self_noise.txt", sep='\t', header=None)

X = data.iloc[:, :-1].values  # Variables explicatives
y = data.iloc[:, -1].values   # Variable cible (bruit)

# standarisation ( moyenne = 0 , écart type = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#80% apprentissage, 20% test
split = int(0.8 * len(X))
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]



#PARTIE_3

#a. Estimation des paramètres pour différentes valeurs de λ

# Liste de valeurs de régularisation lambda

# les valeurs de lambda de 10^(-3) jusqu'à 10^(3) échelle logarithmique

lambdas = np.logspace(-3, 3, 20)
train_errors = []
test_errors = []

for lam in lambdas:
    ridge = Ridge(alpha=lam)
    ridge.fit(X_train, y_train)

    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# b. Sélection de la meilleure valeur de λ

# Visualisation de l’erreur MSE en fonction de λ

plt.figure(figsize=(8, 5))
plt.plot(lambdas, train_errors, label="Train", marker='o')
plt.plot(lambdas, test_errors, label="Test", marker='s')
plt.xscale('log')
plt.xlabel("Lambda (log)")
plt.ylabel("Erreur quadratique moyenne (MSE)")
plt.title("Ridge : MSE vs Lambda")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# meilleur lambda (celui qui correspond à l'MSE la plus faible)
best_lambda_index = np.argmin(test_errors)
best_lambda = lambdas[best_lambda_index]
print(f"Meilleur lambda : {best_lambda:.4f}")


#PARTIE_4

#a. Visualisation des coefficients

# régression avec le meilleur lambda 
ridge_best = Ridge(alpha=best_lambda)
ridge_best.fit(X_train, y_train)
coefficients = ridge_best.coef_

# affichage de coefficients 
feature_names = [
    "Fréquence (Hz)",
    "Angle d'attaque (°)",
    "Longueur de corde (m)",
    "Vitesse d'écoulement (m/s)",
    "Épaisseur de déplacement (m)"
]

plt.figure(figsize=(8, 5))
plt.bar(feature_names, np.abs(coefficients))
plt.ylabel("Valeur absolue des coefficients")
plt.title("Importance des variables")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# affichage de la variable la plus influente (celle ayant la valeur la plus importante)
most_important_index = np.argmax(np.abs(coefficients))
print(f"Variable la plus influente : {feature_names[most_important_index]}")



#b. Régression Ridge avec 2 variables principales


# Sélection des 2 plus grandes valeurs absolues de coefficients
top_2_indices = np.argsort(np.abs(coefficients))[-2:]
print("Variables sélectionnées :", [feature_names[i] for i in top_2_indices])

X_train_top2 = X_train[:, top_2_indices]
X_test_top2 = X_test[:, top_2_indices]

ridge_top2 = Ridge(alpha=best_lambda)
ridge_top2.fit(X_train_top2, y_train)

y_pred_train_top2 = ridge_top2.predict(X_train_top2)
y_pred_test_top2 = ridge_top2.predict(X_test_top2)

mse_train_top2 = mean_squared_error(y_train, y_pred_train_top2)
mse_test_top2 = mean_squared_error(y_test, y_pred_test_top2)

print(f"MSE (train, 2 var) : {mse_train_top2:.2f}")
print(f"MSE (test, 2 var) : {mse_test_top2:.2f}")


#c. Visualisation 3D des prédictions (hyperplan)


# Grille pour surface de régression
x_surf, y_surf = np.meshgrid(
    np.linspace(X_train_top2[:, 0].min(), X_train_top2[:, 0].max(), 50),
    np.linspace(X_train_top2[:, 1].min(), X_train_top2[:, 1].max(), 50)
)
z_surf = ridge_top2.intercept_ + ridge_top2.coef_[0]*x_surf + ridge_top2.coef_[1]*y_surf

# visualisation (apprentissage)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_top2[:, 0], X_train_top2[:, 1], y_train, color='blue', label="Train")
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='red')
ax.set_xlabel(feature_names[top_2_indices[0]])
ax.set_ylabel(feature_names[top_2_indices[1]])
ax.set_zlabel("Niveau sonore")
ax.set_title("Hyperplan de régression (apprentissage)")
plt.legend()
plt.show()

# visualisation (test)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_top2[:, 0], X_test_top2[:, 1], y_test, color='green', label="Test")
ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.5, color='red')
ax.set_xlabel(feature_names[top_2_indices[0]])
ax.set_ylabel(feature_names[top_2_indices[1]])
ax.set_zlabel("Niveau sonore")
ax.set_title("Hyperplan de régression (test)")
plt.legend()
plt.show()



# PARTIE_5

# prédictions avec toutes les variables
y_pred_train_full = ridge_best.predict(X_train)
y_pred_test_full = ridge_best.predict(X_test)

mse_train_full = mean_squared_error(y_train, y_pred_train_full)
mse_test_full = mean_squared_error(y_test, y_pred_test_full)
r2_train_full = r2_score(y_train, y_pred_train_full)
r2_test_full = r2_score(y_test, y_pred_test_full)

print("=== Performances avec toutes les variables ===")
print(f"MSE (train) : {mse_train_full:.2f} | R² (train) : {r2_train_full:.2f}")
print(f"MSE (test)  : {mse_test_full:.2f} | R² (test)  : {r2_test_full:.2f}")

# prédictions avec 2 variables
y_pred_train_top2 = ridge_top2.predict(X_train_top2)
y_pred_test_top2 = ridge_top2.predict(X_test_top2)

mse_train_top2 = mean_squared_error(y_train, y_pred_train_top2)
mse_test_top2 = mean_squared_error(y_test, y_pred_test_top2)
r2_train_top2 = r2_score(y_train, y_pred_train_top2)
r2_test_top2 = r2_score(y_test, y_pred_test_top2)

print("\n=== Performances avec 2 variables ===")
print(f"MSE (train) : {mse_train_top2:.2f} | R² (train) : {r2_train_top2:.2f}")
print(f"MSE (test)  : {mse_test_top2:.2f} | R² (test)  : {r2_test_top2:.2f}")

# Ce modèle montre clairement qu’on perd en performance (R2 diminue) en simplifiant trop, bien que les variables soient les plus importantes selon les coefficients. 


