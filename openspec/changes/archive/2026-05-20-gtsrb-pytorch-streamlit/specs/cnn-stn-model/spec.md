## ADDED Requirements

### Requirement: CNN Architecture Definition
The system SHALL define a Convolutional Neural Network (CNN) in PyTorch for classifying the 43 GTSRB traffic sign classes.

#### Scenario: Forward pass of CNN
- **WHEN** an image batch of shape (N, 3, 32, 32) is passed into the CNN
- **THEN** the model outputs logits of shape (N, 43).

### Requirement: Spatial Transformer Network Module
The system SHALL include an STN module (using `affine_grid` and `grid_sample`) that acts as the first component of the CNN, allowing spatial transformations.

#### Scenario: STN transformation
- **WHEN** an image is passed through the STN
- **THEN** the network learns to output a spatially transformed (aligned/cropped) version of the image before passing it to the convolutional layers.
