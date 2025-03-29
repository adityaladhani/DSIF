

api_url = "http://127.0.0.1:8502"

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import datetime
import base64
import plotly.graph_objects as go

st.title("Fraud Detection App")

# Display site header
#image = Image.open("../images/dsif header.jpeg")

image_path = "../images/dsif header 2.jpeg"
try:
    # Open and display the image
    img = Image.open(image_path)
    st.image(img, use_column_width=True)  # Caption and resizing options
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")
prediction_data = None

def process_data(data):
    """Processes the data regardless of the input source."""
    if isinstance(data, pd.DataFrame):
        st.write("Data from CSV:")
        st.write(data)
        # Perform calculations or transformations on the DataFrame
        # Example: Calculate the mean of each column
        if 'transaction_amount' in data.columns and 'customer_age' in data.columns and 'customer_balance' in data.columns:
            st.write(f"Mean transaction amount: {data['transaction_amount'].mean()}")
            st.write(f"Mean customer age: {data['customer_age'].mean()}")
            st.write(f"Mean customer balance: {data['customer_balance'].mean()}")

    elif isinstance(data, dict):
        st.write("Data from Textboxes:")
        st.write(data)
        # Perform calculations or transformations on the dictionary
        # Example: Calculate the sum of the values
        if all(key in data for key in ("transaction_amount", "customer_age", "customer_balance")):
          try:
            sum_of_values = float(data['transaction_amount']) + float(data['customer_age']) + float(data['customer_balance'])
            st.write(f"Sum of values: {sum_of_values}")
          except ValueError:
            st.write("Please enter numeric values in the textboxes.")
    else:
      st.write("No data to process.")

st.title("Transaction Data Input")

input_method = st.radio("Select Input Method:", ("Textbox Input", "CSV Upload"))

if input_method == "Textbox Input":
    transaction_amount = st.number_input("Transaction Amount")
    customer_age = st.number_input("Customer Age")
    customer_balance = st.number_input("Customer Balance")

    if transaction_amount and customer_age and customer_balance:
        data = {
            "transaction_amount": transaction_amount,
            "customer_age": customer_age,
            "customer_balance": customer_balance
        }
        process_data(data)
        prediction_data = data

if input_method == "CSV Upload":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            df['transaction_amount_to_balance_ratio'] = df['transaction_amount'] / df['customer_balance']
            process_data(df)
            results = []
            fraudulent_data = []

            for index, row in df.iterrows():
                prediction_data = {
                    "transaction_amount": float(row["transaction_amount"]),
                    "transaction_time": row["transaction_time"],
                    "customer_age": int(row["customer_age"]),
                    "customer_balance": float(row["customer_balance"]),
                }
                try:
                    response = requests.post(f"{api_url}/predict/", json=prediction_data)
                    response.raise_for_status()
                    result = response.json()
                    results.append(result)

                    if result["fraud_prediction"] == 1:
                        fraudulent_row = row.to_dict()
                        fraudulent_row["fraud_prediction"] = 1
                        fraudulent_data.append(fraudulent_row)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error processing row {index}: {e}")

            st.session_state['csv_results'] = results

            if fraudulent_data:
                fraud_df = pd.DataFrame(fraudulent_data)
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"fraudulent_transactions_{now}.csv"
                csv = fraud_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()

                st.download_button(
                    label="Download Fraudulent Transactions",
                    data=b64,
                    file_name=filename,
                    mime="text/csv",
                )
            else:
                st.info("No fraudulent transactions found.")

                # Scatter Plot:
            st.subheader("Interactive Scatter Plot")
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist() #get numerical columns.

            x_axis = st.selectbox("Select X-axis", numerical_cols)
            y_axis = st.selectbox("Select Y-axis", numerical_cols)

            fig = go.Figure(data=go.Scatter(x=df[x_axis], y=df[y_axis], mode='markers')) #create plotly figure.
            fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
            st.plotly_chart(fig)

        except pd.errors.EmptyDataError:
            st.error("Uploaded CSV file is empty.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it is correctly formatted.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
          

if st.button("Show Feature Importance"):
    import matplotlib.pyplot as plt
    response = requests.get(f"{api_url}/feature-importance")
    feature_importance = response.json().get('feature_importance', {})

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

if st.button("Predict and show prediction confidence"):
    # Make the API call

    response = requests.post(f"{api_url}/predict/",
                            json=prediction_data)
    result = response.json()
    confidence = result['confidence']

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    # Confidence Interval Visualization
    labels = ['Not Fraudulent', 'Fraudulent']
    fig, ax = plt.subplots()
    ax.bar(labels, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)

if st.button("Predict and show SHAP values"):
    response = requests.post(f"{api_url}/predict/",
                             json=prediction_data)
    result = response.json()

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not fraudulent")
    else:
        st.write("Prediction: Fraudulent")

    ######### SHAP #########
    # Extract SHAP values and feature names
    shap_values = np.array(result['shap_values'])
    features = result['features']

    # Display SHAP values
    st.subheader("SHAP Values Explanation")

    # Bar plot for SHAP values
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('SHAP Value (Impact on Model Output)')
    st.pyplot(fig)
