# Random Forest Classifier for Iris Dataset

This repository contains a simple implementation of a Random Forest Classifier for the famous Iris dataset. The code uses the scikit-learn library for machine learning tasks.

## Usage

1. **Data Splitting:**
   - The dataset is split into training and testing sets using the `train_test_split` function.

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Model Initialization:**
   - A Random Forest model is initialized with 100 estimators.

    ```python
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ```

3. **Model Training:**
   - The model is trained using the training set.

    ```python
    rf_model.fit(X_train, y_train)
    ```

4. **Prediction:**
   - Predictions are made on the test set.

    ```python
    y_pred = rf_model.predict(X_test)
    ```

## Results

- **Accuracy:** 1.0

- **Confusion Matrix:**
-   [[10 0 0]
    [ 0 9 0]
    [ 0 0 11]]

- **Classification Report:**
                   precision    recall  f1-score   support
Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor   1.00      1.00      1.00         9
Iris-virginica    1.00      1.00      1.00        11

accuracy                           1.00        30
macro avg          1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

