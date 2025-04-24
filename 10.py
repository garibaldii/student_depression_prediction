from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

df['Sleep Duration'] = df['Sleep Duration'].str.strip("'\"")

categorical_columns = [
    'Gender', 'City', 'Profession', 'Sleep Duration',
    'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness'
]

encoder = OneHotEncoder(sparse_output=False)
X_categorical = encoder.fit_transform(df[categorical_columns])

X_numeric = df.drop(columns=categorical_columns + ['Depression', 'id'])
X = np.concatenate([X_numeric.to_numpy(), X_categorical], axis=1)
Y = df['Depression']

print(df)
input('Aperte enter para continuar! 😊\n')

print('Matriz de entrada (treinamento):\n', X)
input('Aperte enter para continuar! 😊\n')

print('Vetor de entrada (treinamento):\n', Y)
input('Aperte enter para continuar! 😊\n')

print(f'Quantidade de linhas e colunas: {X.shape}')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print('\n Formato dados de treinamento: ', X_train.shape, Y_train.shape)
print('\n Formato dados de teste: ', X_test.shape, Y_test.shape)
input('Aperte enter para continuar! 😊\n')

modelo = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=500,
    tol=1e-3,
    activation='relu',
    learning_rate='constant',
    verbose=True
)

modelo.fit(X_train, Y_train)

pred = modelo.predict(X_test)
print(f'{pred.dtype}\nVetor de previsões:\n{pred}')
print(f'Comparação Y_teste e vetor previsto:\n{Y_test}\n{pred}')
input('Aperte enter para continuar! 😊\n')

accuracy = accuracy_score(Y_test, pred)
print('Acurácia:', accuracy)
input('Aperte enter para continuar! 😊\n')

cm = confusion_matrix(Y_test, pred)
print("Matriz de Confusão:")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues', colorbar=True)
plt.title("Modelo com 10 neurônios (1 camada)")
plt.show()

print("\nClasses = ", modelo.classes_)
print("Erro = ", modelo.loss_)
print("Amostras visitadas = ", modelo.t_)
print("Atributos de entrada = ", modelo.n_features_in_)
print("N ciclos = ", modelo.n_iter_)
print("N de camadas = ", modelo.n_layers_)
print("Tamanhos das camadas ocultas: ", modelo.hidden_layer_sizes)
print("N de neurônios na saída = ", modelo.n_outputs_)
print("Função de ativação da saída = ", modelo.out_activation_)
