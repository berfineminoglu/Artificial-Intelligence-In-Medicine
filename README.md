
Retinal Blood Vessel Segmentation
Reproduction of the paper "Retinal blood vessel extraction employing effective image features and combination of supervised and unsupervised machine learning methods" on the DRIVE retinal image dataset.

The pipeline combines a supervised classifier (Decision Tree) with unsupervised clustering (Fuzzy C-Means) to separate blood vessels from the background in fundus images.

Pipeline
Pre-processing: The field of view is masked, and the green, Y (luma) and Lab L channels are extracted. Local contrast is enhanced with CLAHE.
Feature extraction (13 features per pixel): Gabor filter responses at multiple wavelengths and angles, automatic thresholding, top-hat transform, shade correction and other vessel-sensitive features.
Classification: Features are reduced with PCA and classified with a Decision Tree. Fuzzy C-Means clustering produces an unsupervised map, and the two are combined into a hybrid result.
Post-processing: Small noisy regions are removed from the binary vessel map.
Evaluation: Confusion matrix metrics and ROC AUC are computed and reported as tables.
Intermediate results of each stage are saved as figures (Fig3 to Fig12), following the figures of the original paper.

Tech stack
Python, OpenCV, NumPy, scikit-learn (PCA, Decision Tree, ROC AUC), scikit-fuzzy, Pillow

How to run
pip install opencv-python numpy scikit-learn scikit-fuzzy pillow tabulate
python main.py
The script expects the DRIVE dataset in a folder named DRIVE next to main.py.

