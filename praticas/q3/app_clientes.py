# app_clientes.py
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🛍️ Agrupamento de Clientes do E-commerce")

df = pd.read_csv("Ecommerce_data.csv")

df.drop('product_name', axis=1, inplace=True)
df.drop('brand_name', axis=1, inplace=True)

X = df[["Cost", "Sales"]]

st.write("### Dados dos Clientes")
st.dataframe(df.head())

# Elbow method
sse = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(X)
    sse.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), sse, marker='o')
ax.set_title("Método do Cotovelo")
ax.set_xlabel("Número de Clusters")
ax.set_ylabel("SSE")
st.pyplot(fig)

k = st.slider("Escolha o número de clusters:", 2, 10, 4)
model = KMeans(n_clusters=k)
df["Cluster"] = model.fit_predict(X)

fig2, ax2 = plt.subplots()
sns.scatterplot(x="Cost", y="Sales", hue="Cluster", data=df, palette="tab10", ax=ax2)
st.pyplot(fig2)
