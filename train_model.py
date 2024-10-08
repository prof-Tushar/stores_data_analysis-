import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle  # Import pickle for saving the model

# Load the dataset
data = pd.read_csv('BigMart.csv')
print(f"Initial dataset shape: {data.shape}")

# Display unique values in relevant columns
print("Unique values in 'Item_Fat_Content':", data['Item_Fat_Content'].unique())
print("Unique values in 'Item_Type':", data['Item_Type'].unique())
print("Unique values in 'Outlet_Identifier':", data['Outlet_Identifier'].unique())

# Check for outliers and filter the dataset
data = data[data['Item_Outlet_Sales'] < data['Item_Outlet_Sales'].quantile(0.95)]
print(f"Filtered dataset shape after outlier removal: {data.shape}")

# Define features and target
features = data.drop(columns=['Item_Outlet_Sales'])
target = data['Item_Outlet_Sales']

# Check the shapes
print(f"Features shape: {features.shape}")
print(f"Target shape: {target.shape}")

# Check if features and target are empty
if features.empty or target.empty:
    print("No data available for training and testing. Check your preprocessing steps.")
else:
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Define a preprocessing pipeline
    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()

    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a final pipeline with a DecisionTreeRegressor
    model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', DecisionTreeRegressor(random_state=42))])

    # Fit the model
    model.fit(x_train, y_train)

    # Optionally: Evaluate the model (add evaluation code here if needed)
    print("Training completed.")

    # Save the trained model to a pickle file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)  # Save the pipeline model
