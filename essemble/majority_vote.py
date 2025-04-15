import numpy as np
import logging
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os

# Set up logging.
logging.basicConfig(level=logging.INFO)

class MajorityVoteModel:
    def __init__(self, x, y, tx, ty, pca_model_path="pca_model.joblib") -> None:
        self.x = x         # training images
        self.y = y          # training labels
        self.tx = tx      # test images
        self.ty = ty        # test labels
        self.logger = logging.getLogger(self.__class__.__name__)
        self.best_model = None
        self.pca_model_path = pca_model_path
        self.pca = None  
        self.scaler = None  
    
    def extract_hog_features(self, images) -> np.array:
        hog_features = []
        for idx, image in enumerate(images):
            try:
                if image.ndim == 3: 
                    gray_image = rgb2gray(image)
                else:
                    gray_image = image  

                # Extract HOG features
                features = hog(
                    gray_image,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                    visualize=False,
                )
                hog_features.append(features)
            except Exception as e:
                self.logger.error(f"Error processing image index {idx}: {e}")

        return np.array(hog_features)
    
    def apply_pca(self, features, n_components) -> np.array:
        if os.path.exists(self.pca_model_path):
            self.logger.info(f"Loading existing PCA model from {self.pca_model_path}")
            self.pca = load(self.pca_model_path)
        else:
            if isinstance(n_components, float):
                self.logger.info(f"Creating new PCA model to preserve {n_components*100:.2f}% of variance")
            else:
                self.logger.info(f"Creating new PCA model to reduce to {n_components} dimensions")
            
            self.pca = PCA(n_components=n_components)
            self.pca.fit(features)

            dump(self.pca, self.pca_model_path)
            self.logger.info(f"Saved new PCA model to {self.pca_model_path}")
        
        reduced_features = self.pca.transform(features)
        self.logger.info(f"Reduced feature shape: {reduced_features.shape}")
        return reduced_features

    def train_and_eval(self, test_image_paths) -> None:
        features_train = self.x
        features_test = self.tx
        self.logger.info("Extracting HOG features from training images...")
        features_train = self.extract_hog_features(features_train)
        
        self.logger.info("Extracting HOG features from test images...")
        features_test = self.extract_hog_features(features_test)

        self.logger.info(f"Shape of HOG features for training data: {features_train.shape}")
        self.logger.info(f"Shape of HOG features for test data: {features_test.shape}")
        # features_train = features_train.reshape(self.x.shape[0], -1)
        # features_test = features_test.reshape(self.tx.shape[0], -1)


        self.scaler = StandardScaler()
        features_train = self.scaler.fit_transform(features_train)
        features_test = self.scaler.transform(features_test)
        
        # features_train = self.apply_pca(features_train, n_components=0.8)
        # features_test = self.pca.transform(features_test)
        

        clf_rf = RandomForestClassifier(random_state=42)
        clf_xgb = XGBClassifier(random_state=42, objective="multi:softmax",num_class=10)
        
        self.logger.info("Setting up the VotingClassifier with RandomForest and XGBClassifier.")
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', clf_rf),
                ('xgb', clf_xgb)
            ],
            voting='hard'
        )

        param_grid = {
            'rf__n_estimators': [200],
            'rf__max_depth': [6],
            'rf__max_features': ['sqrt'],
            'xgb__n_estimators': [200],
            'xgb__max_depth': [6],
            "xgb__eta": [0.3],
        }


        self.logger.info("Training the ensemble classifier with grid search...")
        grid_search = GridSearchCV(
            voting_clf,
            param_grid,
            cv=10,
            n_jobs=1,
            verbose=4
        )
        grid_search.fit(features_train, self.y)

        self.best_model = grid_search.best_estimator_

        predictions = self.best_model.predict(features_test)
        self.save_result(test_image_paths, predictions, output_filename="output.csv")
        return self.best_model, predictions

    def save_result(self, test_image_paths, predictions, output_filename="output.csv"):
            image_names = [os.path.basename(path) for path in test_image_paths]

            # Create a DataFrame with the required format
            result_df = pd.DataFrame({
                "im_name": image_names,
                "label": predictions
            })

            # Save the DataFrame to a CSV file
            result_df.to_csv(output_filename, index=False)
            self.logger.info(f"Predictions saved to {output_filename}")