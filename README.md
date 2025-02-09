

# MNIST Digit Predictor

This project is a simple web application that predicts handwritten digits using machine learning models trained on the MNIST dataset. The application is built using Streamlit and allows users to upload an image, draw a digit, or capture an image via webcam to get a prediction.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Files](#files)
- [License](#license)

## Overview

The MNIST Digit Predictor application uses either a Random Forest or Support Vector Machine (SVM) model to predict handwritten digits. The model is trained on the MNIST dataset, which consists of 28x28 grayscale images of digits from 0 to 9. The application provides three options for input:
1. **Upload an image**: Users can upload an image of a handwritten digit.
2. **Draw a digit**: Users can draw a digit on a canvas.
3. **Capture image via webcam**: Users can capture an image using their webcam.

The application then resizes the input image to 28x28, converts it to grayscale, and flattens it to a 1D array before passing it to the model for prediction.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/mnist-digit-predictor.git
   cd mnist-digit-predictor


  Create a virtual environment (optional but recommended):
bash
Copy
python3 -m venv venv
source venv/bin/activate
Install the required dependencies:
bash
Copy
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy
streamlit run Streamlit_App.py
Open the app in your browser:
The app should open automatically in your default web browser. If not, navigate to http://localhost:8501.
Usage

Upload an image:
Click on "Upload an image" in the sidebar.
Upload an image of a handwritten digit.
Click the "Predict" button to see the predicted digit.
Draw a digit:
Click on "Draw a digit" in the sidebar.
Use your mouse to draw a digit on the canvas.
Click the "Predict" button to see the predicted digit.
Capture image via webcam:
Click on "Capture image via webcam" in the sidebar.
Allow the app to access your webcam and capture an image.
Click the "Predict" button to see the predicted digit.
Models

The application uses either a Random Forest or Support Vector Machine (SVM) model, depending on which model performs better on the validation set. The model is saved as mnist_model.pkl and loaded in the Streamlit app.

Random Forest: A ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees.
Support Vector Machine (SVM): A supervised learning model that finds the hyperplane that best separates the classes in the feature space.
Files

MNIST_Model.py: This script trains the Random Forest and SVM models on the MNIST dataset, evaluates their performance, and saves the best model as mnist_model.pkl.
Streamlit_App.py: This script contains the Streamlit web application that allows users to interact with the model and get predictions.
requirements.txt: Lists the Python dependencies required to run the project.
License

This project is licensed under the MIT License. See the LICENSE file for more details.

Copy

### Notes:
- Replace `your-username` in the clone command with your actual GitHub username.
- Ensure you have a `requirements.txt` file that lists all the dependencies required to run the project. Here’s an example of what it might look like:

```plaintext
numpy
scikit-learn
streamlit
Pillow
streamlit-drawable-canvas
joblib
