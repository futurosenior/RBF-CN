import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dados.csv', skipfooter=1, engine='python')

X = pd.to_numeric(data['Horas'])
y = np.array(data['Jun'])  

def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

centro = np.linspace(0, 24, 10)  
largura = np.repeat(2.0, len(centro))  

rbf_values_list = []
for c, s in zip(centro, largura):
    rbf_values_list.append(rbf(X, c, s))
rbf_values = np.array(rbf_values_list).T

pesos = np.linalg.pinv(rbf_values).dot(y)

X_pred = np.linspace(0, 24, 1000)  
rbf_values_pred = np.array(list(map(lambda c, s: rbf(X_pred, c, s), centro, largura)))
y_pred = pesos.dot(rbf_values_pred)

plt.figure(figsize=(10, 6))
plt.plot(X, y, 'bo', label='Dados mensais de Junho')
plt.plot(X_pred, y_pred, 'r-', label='Ajuste de curvas mensal de Junho')
plt.legend()
plt.xlabel('Hora')
plt.ylabel('Intensidade')
plt.title('Ajuste de curvas mensal - Junho')
plt.show()
