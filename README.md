# 🤖 IA para Detecção de Depressão em Estudantes

Este projeto utiliza aprendizado de máquina para detectar indícios de depressão a partir de dados de um questionário com atributos psicossociais. O modelo foi treinado com diferentes arquiteturas de MLPs (Multi-Layer Perceptrons) utilizando a biblioteca `scikit-learn`.

---

## 📄 Relatório dos Modelos

📎 **[Clique aqui para acessar o relatório completo dos 10 modelos (PDF)](https://github.com/leonfagundes27/Assets/blob/main/Images/Relat%C3%B3rio%20(2).pdf)**  
O relatório contém detalhes sobre a estrutura de cada rede neural, métricas de desempenho e análise dos resultados.

---

## 📊 Tabela Comparativa de Acurácia

Abaixo está a tabela com os resultados obtidos por cada modelo:

<img src="https://github.com/leonfagundes27/Assets/blob/main/Images/tabela.png" style="width:70%;">

---

## 📦 Estrutura do Projeto

- `dataset.csv` → Base de dados com respostas dos participantes. Disponível [aqui](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset).
- Arquivos `.py` → Scripts com arquiteturas distintas de MLP:
  - `10.py`, `20.py`, `50.py`, `100.py`, `200.py`
  - `10+10.py`, `20+20.py`, `50+50.py`, `100+100.py`, `200+200.py`
- `requirements.txt` → Lista de dependências do projeto.
- `.gitignore` → Arquivos ignorados pelo Git.

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

![Matriz de Confusão](https://github.com/leonfagundes27/Assets/blob/main/Images/Matriz_confusão.png)

> 💡 A imagem será salva automaticamente com `plt.savefig("confusion_matrix.png")`.

---

## Resultado da v1

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
