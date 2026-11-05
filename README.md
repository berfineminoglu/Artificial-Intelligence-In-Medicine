
Automated People Counting
Estimates the number of people in video frames from pixel-based image features and a neural network regressor. The project reproduces the feature extraction methods and result tables of a research paper on crowd counting.

Method
Pre-processing: Each frame is converted to grayscale and thresholded.
Feature extraction: Three methods are compared:
Method 1: Number of foreground pixels
Method 2: Pixel count after morphological closing with disc-shaped structuring elements
Method 3: Closed pixel count combined with an erosion ratio
Regression: A multilayer perceptron (MLPRegressor) learns the mapping from features to people count. 153 frames are split into 102 for training and 51 for testing.
Evaluation: Each method is run 10 times and the average percentage error is reported. Training and test results are plotted against the ground truth.
Tech stack
Python, OpenCV, NumPy, scikit-learn, Matplotlib

How to run
pip install opencv-python numpy scikit-learn matplotlib
python people_count.py
The dataset is in VideoDataset/. Result figures are saved to Final_Output/.

