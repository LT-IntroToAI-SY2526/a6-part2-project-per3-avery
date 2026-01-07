"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Avery Liu
- 
- 
- 

Dataset: Possum Regression
Predicting: [What you're predicting]
Features: [List your features]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'possum.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    print("=== Possum Data ===")
    data = pd.read_csv(DATA_FILE) #reading the file
    print(f"\nShape: {data.shape[0]} rows, {data.shape[1]} columns") #I'm not actually using all of the columns.
    
    print(f"\nFirst few rows:")
    print(data.head())

    print(f"\nBasic statistics:")
    print(data.describe())
    
    print(f"\nColumn names: {list(data.columns)}")
    
    return data


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    fig.suptitle('Possum Features vs Length', fontsize=16, fontweight='bold')

    axes[0, 0].scatter(data['age'], data['totlngth'], color='red', alpha=0.6)
    axes[0, 0].set_xlabel('Age (years)')
    axes[0, 0].set_ylabel('Length (cm)')
    axes[0, 0].set_title('Age vs Length')
    axes[0, 0].grid(True, alpha=0.3)
    

    axes[0, 1].scatter(data['sex'], data['totlngth'], color='orange', alpha=0.6)
    axes[0, 1].set_xlabel('Sex')
    axes[0, 1].set_ylabel('Length (cm)')
    axes[0, 1].set_title('Sex vs Length')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(data['earconch'], data['totlngth'], color='yellow', alpha=0.6)
    axes[1, 0].set_xlabel('Ear conch length (mm)')
    axes[1, 0].set_ylabel('Total length (cm)')
    axes[1, 0].set_title('Ear conch length vs Total length')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(data['hdlngth'], data['totlngth'], color='green', alpha=0.6)
    axes[1, 1].set_xlabel('Head length (mm)')
    axes[1, 1].set_ylabel('Total length (cm)')
    axes[1, 1].set_title('Head length vs Total length')
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].scatter(data['skullw'], data['totlngth'], color='blue', alpha=0.6)
    axes[2, 0].set_xlabel('Skull width (mm)')
    axes[2, 0].set_ylabel('Total length (cm)')
    axes[2, 0].set_title('Skull width vs Total length')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].scatter(data['taill'], data['totlngth'], color='purple', alpha=0.6)
    axes[2, 1].set_xlabel('Tail length (cm)')
    axes[2, 1].set_ylabel('Total length (cm)')
    axes[2, 1].set_title('Tail length vs Total length')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    # TODO: Save the figure as 'feature_plots.png' with dpi=300
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    # TODO: Show the plot
    plt.show()


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Your code here
    feature_columns = ['age','sex','earconch','hdlngth','skullw','taill']
    
    X = data[feature_columns]
    y = data['totlngth']
    
    print(f"\n==== Feature Preparation ====")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    print(f"\nFeature columns: {list(X.columns)}")

    #now splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Training set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test, feature_columns




def train_model(X_train, y_train, feature_names):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.intercept_)
    print(zip(feature_names, model.coef_))
    print("y = "+str(model.coef_)+"x + "+str(model.intercept_))

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")



    print(f"\n=== Feature Importance ===")
    
    #admittedly i don't understand these 2 lines
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")

    return predictions


def make_prediction(model, feature_names):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    sample = pd.DataFrame([[age, sex, earconch, hdlngth, skullw, taill]], columns=feature_names)
#['age','sex','earconch','hdlngth','skullw','taill']
    predicted_length = model.predict(sample)[0] 

    sex_name = ['male', 'female'][sex]

    print(f"\n=== New Prediction ===")
    print(f"Possum traits: {earconch:.0f} mm ear conch, {hdlngth} mm long head, {skullw} mm wide skull, {taill} cm long tail, {age} years old, {sex}")
    print(f"Predicted total length: ${predicted_length:,.2f}")
    
    return predicted_length

if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualizex
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test, feature_columns = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train, feature_columns) #X.columns

    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)








    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

