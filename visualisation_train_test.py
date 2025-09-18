import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Charger le dataset depuis un fichier local
data = pd.read_csv(r"C:\Users\39\OneDrive\Desktop\2AGE\TP ML\airfoil_self_noise.txt", sep='\t', header=None)

# Renommer les colonnes pour plus de clarté
data.columns = ['Frequency', 'Angle of attack', 'Chord length',
                'Free-stream velocity', 'Section side displacement thickness',
                'Scaled Sound Pressure Level']

# Afficher des informations sur le dataset
print(data.info())
print(data.describe())

# Standardisation des données (optionnel)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# PCA (optionnel, si une réduction de dimensions est nécessaire)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Découpage des données en ensembles d'entraînement et de test
# Utilisation des 80% premiers exemples comme ensemble d'entraînement et les 20% restants pour le test
train_size = int(0.8 * len(data))
train_data = data[:train_size]  # Les premières lignes (80%) pour l'entraînement
test_data = data[train_size:]  # Les dernières lignes (20%) pour le test

# Afficher la taille des ensembles
print(f"Ensemble d'entraînement : {len(train_data)} exemples")
print(f"Ensemble de test : {len(test_data)} exemples")

# Visualisation des densités pour chaque variable
features = ['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity',
            'Section side displacement thickness', 'Scaled Sound Pressure Level']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(data[feature], label="Dataset complet", color="blue")
    sns.kdeplot(train_data[feature], label="Train Data", color="green")
    sns.kdeplot(test_data[feature], label="Test Data", color="red")
    plt.title(f'Density of {feature}')
    plt.legend()

plt.tight_layout()
plt.show()
