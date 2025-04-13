# 🤖 IA para Detecção de Depressão em Estudantes

Este projeto utiliza aprendizado de máquina para detectar indícios de depressão a partir de dados de um questionário com atributos psicossociais. O modelo foi treinado com um classificador MLP (Multi-Layer Perceptron) da biblioteca `scikit-learn`.

## 📦 Estrutura do Projeto

- `dataset.csv` → Base de dados com respostas dos participantes. Disponível [aqui](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset).
- `main.py` → Script principal que realiza todo o pipeline de pré-processamento, treino e avaliação.
- `requirements.txt` → Lista de dependências do projeto.
- `.gitignore` → Arquivos a serem ignorados pelo Git.

---

## 🚀 Como Rodar o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/leonfagundes27/student_depression_prediction.git
cd student_depression_prediction
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
```

### 3. Ative o ambiente virtual

- **Windows**
```bash
venv\Scripts\activate
```

- **Linux/macOS**
```bash
source venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 🧠 Bibliotecas Utilizadas

- **pandas** → Manipulação e análise de dados tabulares.
- **scikit-learn** → Ferramentas de aprendizado de máquina.
- **matplotlib** → Geração de gráficos (matriz de confusão).
- **numpy** → Suporte a arrays e operações numéricas.
- **OneHotEncoder** → Codificação de variáveis categóricas.

---

## 🔍 Visão Geral do Pipeline

1. Leitura e visualização do dataset
2. Codificação das variáveis categóricas
3. Separação em conjuntos de treino/teste
4. Treinamento com MLPClassifier
5. Avaliação com acurácia e matriz de confusão

---

## 📊 Matriz de Confusão

Aqui será exibida a matriz de confusão gerada após o treinamento do modelo:

![Matriz de Confusão](https://github.com/leonfagundes27/Assets/blob/main/Images/Matriz_confusão.png)

> 💡 Dica: a imagem será salva automaticamente com `plt.savefig("confusion_matrix.png")` no script, se desejar persistir a figura.

---

## Resultado da v1

Abaixo está o resultado da primeira versão do primeiro treinamento:
![Primeira versão](https://github.com/leonfagundes27/Assets/blob/main/Images/v1-ia.png)

---

## 🎯 Métricas e Detalhes

- Acurácia do modelo
- Número de camadas e neurônios
- Função de ativação
- Total de ciclos de treinamento
- Taxa de erro final

---

## 🧪 Exemplo de Execução

Ao rodar o script, você verá saídas no console indicando os dados processados, estrutura da rede neural e métricas de performance.

---

## 📎 Autor

Desenvolvido por [Seu Nome](https://github.com/seu-usuario) 😊  
Sinta-se à vontade para contribuir ou sugerir melhorias!

---

## 📜 Licença

Este projeto está sob a licença MIT.
