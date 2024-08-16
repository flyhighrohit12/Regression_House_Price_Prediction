import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'top_5_features' not in st.session_state:
    st.session_state['top_5_features'] = None

st.title('House Price Prediction App')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    st.session_state['data'] = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["How to Use", "Exploratory Data Analysis", "Model Evaluation", "Prediction"])

with tab1:
    st.header("How to Use This App")
    st.write("""
    Welcome to the House Price Prediction App! This application uses Linear Regression to predict house prices based on various features.

    Here's how to use the app:
    1. Upload a CSV file containing your dataset.
    2. Explore the data in the 'Exploratory Data Analysis' tab.
    3. Train and evaluate the model in the 'Model Evaluation' tab.
    4. Make predictions in the 'Prediction' tab.

    The dataset should include various features of houses and their sale prices.
    """)

with tab2:
    if st.session_state['data'] is not None:
        data = st.session_state['data']
        st.header("Exploratory Data Analysis")
        st.write("First few rows of the data:")
        st.write(data.head())

        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())

        # Histogram of Sale Prices
        st.subheader("Distribution of Sale Prices")
        fig, ax = plt.subplots()
        sns.histplot(data['SalePrice'], ax=ax)
        st.pyplot(fig)

        # Correlation heatmap
        st.subheader("Heatmap of Top Correlations")
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        correlations = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
        top_correlations = correlations.head(6).index.tolist()
        st.session_state['top_5_features'] = top_correlations[1:6]  # Exclude SalePrice itself
        fig, ax = plt.subplots()
        sns.heatmap(numeric_data[top_correlations].corr(), annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Scatter matrix for top features, using alternative approach
        st.subheader("Scatter Plots for Top Correlated Features")
        fig, axs = plt.subplots(len(top_correlations), len(top_correlations), figsize=(15, 15))
        for i, feature_i in enumerate(top_correlations):
            for j, feature_j in enumerate(top_correlations):
                if i == j:
                    axs[i, j].hist(numeric_data[feature_i].dropna())
                else:
                    axs[i, j].scatter(numeric_data[feature_j], numeric_data[feature_i], alpha=0.5)
                if j == 0:
                    axs[i, j].set_ylabel(feature_i)
                if i == len(top_correlations) - 1:
                    axs[i, j].set_xlabel(feature_j)
        plt.tight_layout()
        st.pyplot(fig)

with tab3:
    if st.session_state['data'] is not None and st.session_state['top_5_features'] is not None:
        data = st.session_state['data']
        st.header("Model Evaluation and Training")

        train_size = st.slider('Select training data percentage', 60, 90, 80, 5)
        test_size = (100 - train_size) / 100

        if st.button('Train Model'):
            X = data[st.session_state['top_5_features']]
            y = data['SalePrice']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = Pipeline([
                ('preprocessor', ColumnTransformer([
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ]), st.session_state['top_5_features'])
                ])),
                ('regressor', LinearRegression())
            ])
            model.fit(X_train, y_train)
            with open('trained_model.pkl', 'wb') as file:
                pickle.dump(model, file)
            st.success("Model trained and saved successfully!")

with tab4:
    st.header("Prediction")
    try:
        with open('trained_model.pkl', 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully.")

        input_data = {}
        for feature in st.session_state['top_5_features']:
            input_data[feature] = st.number_input(f'Enter value for {feature}:', value=float(data[feature].mean()))

        if st.button('Predict House Price'):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.write(f'Predicted House Price: ${prediction:,.2f}')
    except FileNotFoundError:
        st.error("The model has not been trained yet. Please train the model in the 'Model Evaluation' tab.")
