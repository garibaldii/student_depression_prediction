from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('dataset.csv')
print(df)
input('Aperte enter para continuar! 😊\n')

X = df.loc[:, 'Gender':'Family History of Mental Illness']
print('Matriz de entrada (treinamento):\n', X)
input('Aperte enter para continuar! 😊\n')

Y = df['Depression']
print('Vetor de entrada (treinamento):\n', Y)
input('Aperte enter para continuar! 😊\n')

print(f'Quantidade de linhas e colunas: {X.shape}' )

encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform(df.loc[:, 'Gender':'Family History of Mental Illness'])
print('Matriz de entrada codificado (treinamento):\n', X)
input('Aperte enter para continuar! 😊\n')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print('\n Formato dados de treinamento: ', X_train.shape, Y_train.shape)
print('\n Formato dados de teste: ', X_test.shape, Y_test.shape)
input('Aperte enter para continuar! 😊\n')

neural_network = MLPClassifier(
    verbose=True,
    hidden_layer_sizes=(2, 8),
    max_iter=2000,
    tol=1e-5,
    activation='relu'
)

neural_network.fit(X_train, Y_train)

prediction = neural_network.predict(X_test)
print(f'{prediction.dtype}\nVetor de previsões:\n{prediction}')
print(f'Comparação Y_teste e vetor previsto:\n{Y_test}\n{prediction}')
input('Aperte enter para continuar! 😊\n')

accuracy = accuracy_score(Y_test, prediction)
print('Acurácia:', accuracy)
input('Aperte enter para continuar! 😊\n')

confusion_matrix_ = confusion_matrix(Y_test, prediction)
print("Matriz de Confusão:")
print(confusion_matrix_)

visual_confusion_matrix = confusion_matrix(Y_test, prediction, labels=neural_network.classes_)
display = ConfusionMatrixDisplay(
    confusion_matrix = visual_confusion_matrix,
    display_labels = neural_network.classes_
)
display.plot()
plt.show()

print("\nClasses = ", neural_network.classes_)    
print("Erro = ", neural_network.loss_)   
print("Amostras visitadas = ", neural_network.t_)  
print("Atributos de entrada = ", neural_network.n_features_in_)   
print("N ciclos = ", neural_network.n_iter_)    
print("N de camadas = ", neural_network.n_layers_) 
print("Tamanhos das camadas ocultas: ", neural_network.hidden_layer_sizes)
print("N de neurons saida = ", neural_network.n_outputs_) 
print("F de ativação = ", neural_network.out_activation_)  