## ADDED Requirements

### Requirement: Random Image Sampling UI
The system SHALL provide a Streamlit UI button that allows the user to randomly sample a traffic sign image from the test dataset.

#### Scenario: Clicking "Random Image"
- **WHEN** the user clicks the sample button
- **THEN** a random traffic sign image is displayed on the screen along with its true label.

### Requirement: Real-time Model Inference
The system SHALL run inference on the sampled image using both the CNN model and the ML models, displaying their predictions.

#### Scenario: Viewing predictions
- **WHEN** an image is sampled
- **THEN** the UI displays the predicted classes and confidence scores (if applicable) for the CNN, SVM, RF, KNN, and AdaBoost models.
