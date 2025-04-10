# Curso: Análise e Desenvolvimento de Sistemas - UNIFACISA
# Autor: Felipe Monteiro - 10/04 19/27

#importando os dados para a maquina local
import streamlit as st # para criar a interface web
from sklearn.datasets import load_iris # para carregar o dataset iris
from sklearn.model_selection import train_test_split # para dividir os dados
from sklearn.neighbors import KNeighborsClassifier # para o modelo KNN
from sklearn.metrics import accuracy_score, classification_report # para avaliar o modelo
import pandas as pd # para manipulação de dados

# Criando objetos que vão aparecer na tela do sistema
st.title("Classificação de Flores Iris!")

# Carregando a base de dados para um dataframe e separando as features de seleção e classificação
iris = load_iris()

X = iris.data
y = iris.target

# Exibindo os doados na tela em formato de tabela
st.write("### Dados Iris")
st.dataframe(pd.DataFrame(X, columns=iris.feature_names).assign(target=y))

# Separando a base em 70% para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
k = st.slider("Escolha o valor de K para o KNN", 1, 15, 5) # criando um slider para o valor de K

# Fazer o treinamento do Modelo de ML
model = KNeighborsClassifier(n_neighbors=k) # criando o modelo KNN
model.fit(X_train, y_train) # treinando o modelo
y_pred = model.predict(X_test) # fazendo a predição

# Avaliação de Modelo de MAchine Learning com Acuraria e o relat'rorio de classificação
acc = accuracy_score(y_test, y_pred) # calculando a acuracia
st.success(f"Acurácia do modelo: {acc:.2f}") # exibindo a acuracia

# Exibir o modelo de classificação de Machine Learning
st.text("### Relatório de Classificação")
st.text(classification_report(y_test, y_pred, target_names=iris.target_names)) # exibindo o relatório de classificação