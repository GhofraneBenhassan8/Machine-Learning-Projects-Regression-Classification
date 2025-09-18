import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv(r"C:\Users\39\OneDrive\Desktop\TP ML\airfoil_self_noise.txt", sep='\t', header=None)


data.columns = ['Frequency', 'Angle of attack', 'Chord length',
                'Free-stream velocity', 'Section side displacement thickness',
                'Scaled Sound Pressure Level']

print(data.info())
print(data.describe())


features = ['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity',
            'Section side displacement thickness', 'Scaled Sound Pressure Level']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(data[feature])
    plt.title(f'Density of {feature}')

plt.tight_layout()
plt.show()
