## ADDED Requirements

### Requirement: Model Accuracy and Loss Metrics
The system SHALL compute and log the accuracy and loss for the CNN model during training and evaluation phases.

#### Scenario: CNN Evaluation
- **WHEN** the evaluation loop runs on the test set
- **THEN** the overall accuracy and categorical loss are reported.

### Requirement: Cross-Model Comparison Metrics
The system SHALL aggregate the test accuracies of the CNN and all ML models to generate comparative visualizations (e.g., bar charts).

#### Scenario: Generating Comparison Report
- **WHEN** all models are evaluated
- **THEN** a bar chart or table is generated comparing the final accuracies of CNN vs SVM, RF, KNN, AdaBoost, and K-means.
