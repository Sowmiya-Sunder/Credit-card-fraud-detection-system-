import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit interface
st.title("Credit Card Fraud Detection Model")

# Display instructions for the user
st.write(f"Please enter exactly {X_train.shape[1]} feature values, separated by commas.")
st.write("Example input: 1.0, -0.5, 0.3, ...")

# Input from user
input_df = st.text_input('Enter all required feature values (comma-separated):')

# Submit button
submit = st.button("Submit")

if submit:
    if not input_df:
        st.error("Input is empty. Please provide the feature values.")
    else:
        try:
            # Split and convert the input to a list of floats
            input_values = [float(i.strip()) for i in input_df.split(',')]

            # Check if the input length matches the model's expected number of features
            if len(input_values) != X_train.shape[1]:
                st.error(f"Incorrect number of inputs. Expected {X_train.shape[1]} feature values.")
            else:
                # Reshape the input and make prediction
                features = np.array(input_values).reshape(1, -1)
                prediction = model.predict(features)

                # Output prediction result
                if prediction[0] == 0:
                    st.success("LEGITIMATE TRANSACTION")
                else:
                    st.warning("FRAUD TRANSACTION")
        except ValueError:
            st.error("Invalid input. Please ensure all inputs are valid numbers.")


