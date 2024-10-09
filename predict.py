import numpy as np
import pandas as pd
import pickle

# Function to load the trained model
def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict sales using the loaded model and provided features
def predict_sales(model, features):
    # Ensure that features are in the correct shape for the model
    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    # Make predictions using the model
    prediction = model.predict(features)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Example feature values (adjust based on your dataset)
    example_features = {
        'Item_Identifier': 'FDA15',             # Categorical
        'Item_Weight': 12.60,                    # Numeric
        'Item_Fat_Content': 'Low Fat',           # Categorical
        'Item_Type': 'Dairy',                    # Categorical
        'Item_MRP': 249.81,                      # Numeric
        'Outlet_Identifier': 'OUT027',           # Categorical
        'Outlet_Establishment_Year': 1997,      # Numeric
        'Outlet_Size': 'Medium',                 # Categorical
        'Outlet_Location_Type': 'Tier 1',        # Categorical
        'Outlet_Type': 'Grocery Store',          # Categorical
        'Item_Visibility': 0.016047,             # Numeric (Ensure to include this)
        # Include any other features necessary that were part of the model training
    }

    # Convert the example features to a DataFrame
    features_df = pd.DataFrame([example_features])

    # Predict sales
    try:
        predicted_sales = predict_sales(model, features_df)
        print(f"Predicted sales: {predicted_sales:.2f}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
