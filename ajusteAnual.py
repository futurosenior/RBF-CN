import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dados.csv')

X = np.arange(1, 13)  
y = data.iloc[:-1, 1:].sum()  

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

centro = np.linspace(1, 12, 12) 
largura = np.repeat(1.0, len(centro)) 

rbf_values_list = []
for c, s in zip(centro, largura):
    rbf_values_list.append(rbf(X, c, s))
rbf_values = np.array(rbf_values_list).T

pesos = np.linalg.pinv(rbf_values).dot(y)

X_pred = np.linspace(1, 12, 1000)  
rbf_values_pred = np.array([rbf(X_pred, c, s) for c, s in zip(centro, largura)])
y_pred = pesos.dot(rbf_values_pred)

plt.figure(figsize=(10, 6))
plt.plot(X, y, 'bo', label='Dados anuais')
plt.plot(X_pred, y_pred, 'r-', label='Ajuste de curvas anual')
plt.legend()
plt.xlabel('Mês')
plt.ylabel('Intensidade')
plt.title('Ajuste de curvas anual')
plt.show()


