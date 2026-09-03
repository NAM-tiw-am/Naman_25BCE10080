# Machine Learning Fundamentals

## What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables systems to learn and improve from experience without being explicitly programmed. Instead of writing rules manually, ML algorithms find patterns in data and use them to make predictions or decisions.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the model is trained on labeled data — input-output pairs. The algorithm learns to map inputs to correct outputs. Common algorithms include Linear Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM). Use cases include spam detection, image classification, and price prediction.

### Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm discovers hidden patterns or groupings on its own. Common algorithms include K-Means Clustering, DBSCAN, and Principal Component Analysis (PCA). Use cases include customer segmentation, anomaly detection, and recommendation systems.

### Reinforcement Learning
In reinforcement learning, an agent learns by interacting with an environment and receiving rewards or penalties. The agent's goal is to maximize cumulative reward. Notable examples include AlphaGo by DeepMind, self-driving cars, and game-playing AI.

## Key Concepts

### Training and Testing
Data is typically split into training set (70-80%) and testing set (20-30%). The model learns from training data and is evaluated on unseen testing data. Cross-validation uses multiple splits to get a more reliable evaluation. A common technique is K-Fold Cross-Validation where the data is divided into K subsets.

### Overfitting and Underfitting
Overfitting occurs when a model memorizes training data but fails on new data. The model is too complex. Underfitting occurs when a model is too simple to capture the patterns. The solution is to find the right balance — the "bias-variance tradeoff." Regularization techniques like L1 (Lasso) and L2 (Ridge) help prevent overfitting.

### Feature Engineering
Feature engineering is the process of selecting, transforming, and creating input variables (features) that improve model performance. Techniques include normalization (scaling values to 0-1), standardization (zero mean, unit variance), one-hot encoding for categorical variables, and polynomial features for capturing non-linear relationships.

## Evaluation Metrics

For classification: Accuracy, Precision, Recall, F1-Score, and AUC-ROC curve.
For regression: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (coefficient of determination).

## Popular ML Libraries

- Scikit-learn: General-purpose ML in Python. Best for classical algorithms.
- TensorFlow: Google's deep learning framework. Used for neural networks.
- PyTorch: Facebook's deep learning framework. Preferred in research.
- XGBoost: Gradient boosting library. Dominates tabular data competitions.
- Keras: High-level API for building neural networks, now integrated into TensorFlow.
