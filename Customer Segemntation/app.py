import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained KMeans model
model = joblib.load('cust_seg_model.pkl')

# Load dataset for ranges and cluster plotting
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    X = df.iloc[:, [3, 4]].values  # Annual Income, Spending Score
    return df, X

df, X = load_data()

st.title("Customer Segmentation App")
st.write("Predict which customer segment a new customer belongs to, based on Annual Income and Spending Score.")

# User inputs
income = st.slider(
    label="Annual Income (k$)",
    min_value=int(X[:, 0].min()),
    max_value=int(X[:, 0].max()),
    value=int(X[:, 0].mean())
)
score = st.slider(
    label="Spending Score (1-100)",
    min_value=int(X[:, 1].min()),
    max_value=int(X[:, 1].max()),
    value=int(X[:, 1].mean())
)

if st.button("Predict Segment"):
    features = np.array([[income, score]])
    cluster = model.predict(features)[0]
    st.success(f"Customer belongs to segment: **Cluster {cluster}**")

    # Display cluster centers
    centers = model.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=['Annual Income (k$)', 'Spending Score'])
    st.subheader("Cluster Centers")
    st.table(centers_df)

    # Plot clusters with the new customer
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(model.n_clusters):
        ax.scatter(
            X[model.labels_ == i, 0],
            X[model.labels_ == i, 1],
            s=50,
            label=f"Cluster {i}"
        )
    # Centroids
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=200,
        c='black',
        marker='X',
        label='Centroids'
    )
    # New customer
    ax.scatter(
        income, score,
        s=300,
        c='yellow',
        edgecolors='black',
        marker='*',
        label='New Customer'
    )
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
