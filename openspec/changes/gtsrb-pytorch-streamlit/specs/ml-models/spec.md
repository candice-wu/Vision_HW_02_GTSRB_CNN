## ADDED Requirements

### Requirement: Traditional ML Classifiers
The system SHALL implement multiple traditional Machine Learning classifiers (SVM, Random Forest, K-means, KNN, AdaBoost) using `scikit-learn` to classify the GTSRB images.

#### Scenario: Training ML models
- **WHEN** the training data is provided to the ML pipeline
- **THEN** all specified ML models (SVM, RF, KNN, AdaBoost, K-means) fit the data and produce callable prediction models.

### Requirement: PCA Dimensionality Reduction
The system SHALL use Principal Component Analysis (PCA) to reduce the dimensionality of flattened image vectors before training the traditional ML models.

#### Scenario: Applying PCA
- **WHEN** flattened images of size 3072 (32x32x3) are passed to the PCA module
- **THEN** the module outputs reduced feature vectors retaining 95% of the variance, to be consumed by the ML models.
