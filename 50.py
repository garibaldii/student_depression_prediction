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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=42)

modelo = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=1000,
    tol=1e-8,
    activation='logistic',
    learning_rate='constant',
    verbose=True
)
modelo.fit(X_train, Y_train)
pred = modelo.predict(X_test)

print("Acurácia:", accuracy_score(Y_test, pred))
print("Classes = ", modelo.classes_)    
print("Erro = ", modelo.loss_)   
print("Amostras visitadas = ", modelo.t_)  
print("Atributos de entrada = ", modelo.n_features_in_)   
print("N ciclos = ", modelo.n_iter_)    
print("N de camadas = ", modelo.n_layers_) 
print("Tamanhos das camadas ocultas: ", modelo.hidden_layer_sizes)
print("N de neurônios na saída = ", modelo.n_outputs_) 
print("Função de ativação da saída = ", modelo.out_activation_) 

cm = confusion_matrix(Y_test, pred)
print("Matriz de Confusão:\n", cm)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Oranges', colorbar=True)
plt.title("Modelo com 50 neurônios (1 camada)")
plt.show()
