## ADDED Requirements

### Requirement: Kaggle Dataset Downloading and Parsing
The system SHALL be able to download the GTSRB dataset from Kaggle and parse its structure (train/test directories, CSV labels).

#### Scenario: Downloading raw data
- **WHEN** the dataset initialization script runs
- **THEN** the system verifies if the dataset exists locally, and if not, downloads and extracts it from Kaggle.

### Requirement: PyTorch Dataset and DataLoader
The system SHALL implement a PyTorch Dataset class that loads GTSRB images, applies necessary transformations (resizing to 32x32), and exposes them via a DataLoader.

#### Scenario: Fetching a batch for training
- **WHEN** the DataLoader is iterated
- **THEN** it yields a batch of resized images (tensors) and corresponding class labels.
