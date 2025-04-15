import numpy as np
import logging
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import TSNE, SpectralEmbedding
from skimage.transform import resize
import pandas as pd
from skimage.feature import hog,local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from skimage.color import rgb2gray
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import umap
import os

# Set up logging.
logging.basicConfig(level=logging.INFO)

class ModelXGBoost:
    def __init__(self, x, y, tx, ty, pca_model_path="pca_model.joblib") -> None:
        self.x = x   
        self.y = y     
        self.tx = tx
        self.ty = ty
        self.logger = logging.getLogger(self.__class__.__name__)
        self.best_model = None
        self.pca_model_path = pca_model_path
        self.pca = None 
        self.scaler = None 

    def apply_lbp(self, images, **kwargs):
        radius = kwargs.get("radius", 1)
        n_points = kwargs.get("n_points", 8 * radius)
        method = kwargs.get("method", "uniform")
        return_histogram = kwargs.get("return_histogram", True)
        
        features_list = []
        
        for img in images:
            # If the image is in color (3 channels), convert to grayscale.
            if img.ndim == 3:
                img_gray = rgb2gray(img)
            else:
                img_gray = img

            lbp_image = local_binary_pattern(img_gray, n_points, radius, method)
            
            if return_histogram:
                if method == "uniform":
                    n_bins = n_points + 2
                else:
                    n_bins = int(lbp_image.max() + 1)
                
                # Compute the histogram of LBP values and normalize the histogram.
                hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_bins + 1), density=True)
                features_list.append(hist)
            else:
                features_list.append(lbp_image)
        
        features_array = np.array(features_list)
        return features_array
    
    def extract_hog_features(self, images) -> np.array:
        hog_features = []
        resized_images = [resize(image, (64, 64), anti_aliasing=True) for image in images]
        
        for idx, image in enumerate(resized_images):
            # Compute HOG features
            features = hog(
                image,
                orientations=12,
                pixels_per_cell=(8,8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                visualize=False,
                transform_sqrt=True,
                channel_axis=-1
            )
            hog_features.append(features)
        
        return np.array(hog_features)
    
    def apply_pca(self, features, n_components) -> np.array:
        if isinstance(n_components, float):
            self.logger.info(f"Creating new PCA model to preserve {n_components*100:.2f}% of variance")
        else:
            self.logger.info(f"Creating new PCA model to reduce to {n_components} dimensions")
        
        self.pca = PCA(n_components=n_components)
        self.pca.fit(features)
        # dump(self.pca, self.pca_model_path)
        self.logger.info(f"Saved new PCA model to {self.pca_model_path}")
        reduced_features = self.pca.transform(features)
        self.logger.info(f"Reduced feature shape: {reduced_features.shape}")
        return reduced_features

    def apply_approx_kernel_pca(self, features, **kwargs):
        n_components = kwargs.get("n_components", None)
        if n_components is None:
            raise ValueError("Parameter 'n_components' must be provided in kwargs.")

        kernel = kwargs.get("kernel", "rbf")
        gamma = kwargs.get("gamma", None)
        n_samples = kwargs.get("n_samples", 10000)
        n_jobs = kwargs.get("n_jobs", 1)
        
        # Build the Nystroem transformer for kernel approximation
        nystroem = Nystroem(kernel=kernel, gamma=gamma, n_components=n_samples, random_state=42, n_jobs=n_jobs)
        features_transformed = nystroem.fit_transform(features)
        
        # Apply PCA to the transformed features
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(features_transformed)
        
        # Create a pipeline that encapsulates both transformations
        pipeline = make_pipeline(nystroem, pca)
        
        # Transform the original features using the entire pipeline
        transformed_features = pipeline.transform(features)
        print(f"Approx Kernel PCA: Reduced feature shape: {transformed_features.shape}")
        
        return transformed_features, pipeline
    def apply_kernel_pca(self, features, n_components, **kwargs) -> np.array:
        kpca_model_path = "kernel_pca_model.joblib"
        
        # Ensure required parameters are present
        kwargs["n_components"] = n_components
        # Set default values if not provided in kwargs
        kwargs.setdefault("kernel", "rbf")
        kwargs.setdefault("random_state", 42)
        
        if os.path.exists(kpca_model_path):
            self.logger.info(f"Loading existing Kernel PCA model from {kpca_model_path}")
            kpca = load(kpca_model_path)
        else:
            self.logger.info(f"Creating new Kernel PCA model with parameters: {kwargs}")
            kpca = KernelPCA(**kwargs)
            kpca.fit(features)
#            dump(kpca, kpca_model_path)
            self.logger.info(f"Saved new Kernel PCA model to {kpca_model_path}")
        
        transformed_features = kpca.transform(features)
        self.logger.info(f"Kernel PCA: Reduced feature shape: {transformed_features.shape}")
        return transformed_features, kpca
    
    def train_and_eval(self, test_image_paths, *args, **kwargs):
        # Start with the original images.
        features_train = self.x
        features_test = self.tx

        if kwargs.get("lbp", False):
            self.logger.info("Extracting LBP features from training images...")
            # features_train = self.apply_lbp(features_train,return_histogram=False)
            lbp_features_train = self.apply_lbp(self.x, radius=1, n_points=10, method="uniform")
            self.logger.info(f"Shape of LBP features for training data: {features_train.shape}")

            self.logger.info("Extracting LBP features from test images...")
            # features_test = self.apply_lbp(features_test,return_histogram=False)
            lbp_features_test = self.apply_lbp(self.tx, radius=1, n_points=10, method="uniform")
            self.logger.info(f"Shape of LBP features for test data: {features_test.shape}")


        if kwargs.get("hog", False):
            self.logger.info("Extracting HOG features from training images...")
            features_train = self.extract_hog_features(features_train)
            
            self.logger.info("Extracting HOG features from test images...")
            features_test = self.extract_hog_features(features_test)

            self.logger.info(f"Shape of HOG features for training data: {features_train.shape}")
            self.logger.info(f"Shape of HOG features for test data: {features_test.shape}")

        if kwargs.get("lbp", False) and kwargs.get("hog",False):
            features_train = np.concatenate([features_train, lbp_features_train], axis=1)
            features_test = np.concatenate([features_test, lbp_features_test], axis=1)
        
        if kwargs.get("reshape",False):
            features_train = features_train.reshape(self.x.shape[0], -1)
            features_test = features_test.reshape(self.tx.shape[0], -1)

        # Standardize the features
        self.scaler = StandardScaler()
        features_train = self.scaler.fit_transform(features_train)
        features_test = self.scaler.transform(features_test)

        if kwargs.get("rfe", False):
            self.logger.info("Applying Recursive Feature Elimination (RFE)...")

            estimator = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
            rfe = RFE(estimator=estimator, n_features_to_select=150, step=50)
            features_train = rfe.fit_transform(features_train, self.y)
            features_test = rfe.transform(features_test)
            self.logger.info(f"Shape of features after RFE: training: {features_train.shape}, test: {features_test.shape}")
            
        # Linear PCA
        if kwargs.get("Linear_PCA", False):
            features_train = self.apply_pca(features_train, n_components=120)
            features_test = self.pca.transform(features_test)

        # Kernel PCA
        if kwargs.get("Kernal_PCA", False):
            features_train, kpca = self.apply_kernel_pca(features_train, n_components=150, kernel="rbf", gamma=1e-3, alpha=1e-4, fit_inverse_transform=True, n_jobs=4)
            features_test = kpca.transform(features_test)
            
        # Approximate Kernel PCA    
        if kwargs.get("Approx_Kernal_PCA", False):
            self.logger.info("applying kernal PCA (approximation)")
            features_train, kpca = self.apply_approx_kernel_pca(features_train, n_components=120, kernel="rbf", gamma=1e-3, n_samples=10000, n_jobs=2)
            features_test = kpca.transform(features_test)

        # t-SNE for visualization
        if kwargs.get("TSNE", False):
            self.logger.info("Applying t-SNE to training data...")
            tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
            features_train_tsne = tsne.fit_transform(features_train)               
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(features_train_tsne[:, 0],
                                features_train_tsne[:, 1],
                                c=self.y,  # assuming self.y are your labels
                                cmap="viridis",
                                s=50,
                                alpha=0.7)
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.title("t-SNE Visualization of Training Data")
            plt.colorbar(scatter, label="Labels")
            plt.show()

        # Spectral Embedding for non-linear embedding
        if kwargs.get("Spectral_Embedding", False):
            self.logger.info("Applying Spectral Embedding to features...")
            combined_features = np.concatenate((features_train, features_test), axis=0)
            spectral = SpectralEmbedding(n_components=3, random_state=42)
            combined_embedding = spectral.fit_transform(combined_features)
            n_train = features_train.shape[0]
            features_train = combined_embedding[:n_train]
            features_test = combined_embedding[n_train:]

        # UMAP for non-linear embedding
        if kwargs.get("UMAP", False):
            self.logger.info("Applying UMAP to features...")
            umap_reducer = umap.UMAP(n_components=3, random_state=42)
            features_train = umap_reducer.fit_transform(features_train)
            features_test = umap_reducer.transform(features_test)
                
        # Classification using XGBoost
        clf_xgb = XGBClassifier(random_state=42, objective="multi:softmax", num_class=10)

        param_grid = {
            'n_estimators': [300],
            'max_depth': [12],
            "eta": [0.2],
            "subsample": [1],

        }

        self.logger.info("Training the ensemble classifier with grid search...")
        grid_search = GridSearchCV(
            clf_xgb,
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
        result_df = pd.DataFrame({
            "im_name": image_names,
            "label": predictions
        })
        result_df.to_csv(output_filename, index=False)
        self.logger.info(f"Predictions saved to {output_filename}")