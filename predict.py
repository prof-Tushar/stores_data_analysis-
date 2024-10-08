# predict.py
import numpy as np

# Function to predict sales using the loaded model and provided features
def predict_sales(model, features):
    # Ensure that features are in the correct shape for the model
    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    # Make predictions using the model
    prediction = model.predict(features)
    return prediction[0]
