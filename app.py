from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    item_weight = float(request.form['Item_Weight'])
    item_visibility = float(request.form['Item_Visibility'])
    item_mrp = float(request.form['Item_MRP'])
    outlet_size = int(request.form['Outlet_Size'])  # Adjust based on how you handle outlet size
    outlet_location_type = int(request.form['Outlet_Location_Type'])  # Adjust as needed
    outlet_type = int(request.form['Outlet_Type'])  # Adjust based on your encoding
    item_fat_content = int(request.form['Item_Fat_Content'])  # Adjust based on your encoding
    item_type = int(request.form['Item_Type'])  # Adjust based on your encoding

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Item_Weight': [item_weight],
        'Item_Visibility': [item_visibility],
        'Item_MRP': [item_mrp],
        'Outlet_Size': [outlet_size],
        'Outlet_Location_Type': [outlet_location_type],
        'Outlet_Type': [outlet_type],
        'Item_Fat_Content': [item_fat_content],
        'Item_Type': [item_type],
        'Outlet_Identifier': ['OUT001'],  # Set a default or dummy value
        'Item_Identifier': ['I001'],      # Set a default or dummy value
        'Outlet_Establishment_Year': [1990] # Set a default or dummy value
    })

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Return prediction result
    prediction_text = f"Predicted Sales: ${prediction[0]:.2f}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
