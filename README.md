# Face-Detection-and-Recognition
## Description
This is my workshop project assignment for Computer Vision course. In this project, the dataset is 70 images with 7 individuals and 10 pictures of each. The dataset trained using eigenface method and Support Vector Machine (SVC) algorithm. For the detailed, please refer to `FaceRecognitionEigenface.ipynb`.

## Prerequisites
Python 3.7 or newer

## How to Use
This is step by step to run my code using the existing model, which is the model that i've trained with this 70 images.
1. Clone Repository
   ```
   git clone https://github.com/lituldust/Face-Detection-and-Recognition.git
   cd 'Face-Detection-and-Recognition'
   ```
2. Setup Virtual Environment
   ```
   python -m venv .venv
   .venv/Scripts/activate
   ```
3. Install Packages
   ```
   pip install -r requirements.txt
   ```
4. Real Time Implementation
   Run the `main.py`
   ```
   python main.py
   ```

## Retraining The Model
If you want to retrain the model using your new dataset, you can:
1. Add New Dataset into `images`
2. Follow the Detailed Step in `FaceRecognitionEigenface.ipynb`
3. Save the New Trained Model
4. Run `main.py`
