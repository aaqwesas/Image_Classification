import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from majority_vote import MajorityVoteModel
from model_xgboost import ModelXGBoost
from skimage.color import rgb2gray
from skimage import io
from skimage.util import random_noise

class ImageProcessing:
    def __init__(self, path) -> None:
        """
        Initialize the image processing pipeline with the given dataset path.
        """
        self.path = path

    def apply_laplacian(self, img):
        """
        Apply the Laplacian operator to detect edges in the image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(normalized)

    def apply_sobel(self, img):
        """
        Apply the Sobel operator to detect edges in the image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) 
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        threshold = 50
        edges = magnitude > threshold
        return np.array(edges)

    def apply_clahe(self, img):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the image.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        enhanced = cv2.merge((h, s, v))
        return cv2.cvtColor(enhanced, cv2.COLOR_HSV2RGB)
    def augment_image(self,img):
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # Horizontal flip with 50% probability
        if np.random.rand() > 0.5:
            rotated = cv2.flip(rotated, 1)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        bright_adjusted = np.clip(rotated * brightness, 0, 255).astype(np.uint8)
        
        # Optionally add Gaussian noise
        noise = np.random.normal(0, 10, bright_adjusted.shape).astype(np.uint8)
        augmented_img = cv2.add(bright_adjusted, noise)
        
        return augmented_img
    def normalize_image(self, img):
        """
        Normalize the image pixel values to the range [0, 1].
        """
        return img / 255.0

    def preprocess_image(self, img, apply_laplacian=False, apply_sobel=False, apply_clahe=True,augment_image=False):
        """
        Preprocess a single image by applying multiple steps.
        """
        if apply_clahe:
            img = self.apply_clahe(img)

        if apply_laplacian:
            img = self.apply_laplacian(img)

        if apply_sobel:
            img = self.apply_sobel(img)
        
        if augment_image:
            img = self.augment_image(img)

        img = self.normalize_image(img)

        return img

    def load_images(self, image_paths, preprocess_args=None,to_gray=False):
        """
        Load and preprocess all images from the given paths.
        """
        images = []
        for path in image_paths:
            try:
                # Load image
                img = io.imread(path)
                img = np.array(img)
                if to_gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = np.array(img)  # Convert to NumPy array for OpenCV processing

                # Preprocess image
                if preprocess_args:
                    img = self.preprocess_image(img, **preprocess_args)

                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        return np.array(images)

    def main(self):
        """
        Main method to load images, preprocess them, and return the processed data.
        """
        # Define dataset paths
        train_images_folder = os.path.join(self.path, "train_ims")
        test_images_folder = os.path.join(self.path, "test_ims")
        train_csv_path = os.path.join(self.path, "train.csv")
        test_csv_path = os.path.join(self.path, "test.csv")

        # Load train.csv and test.csv
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)

        # Add full paths to images in the dataframes
        train_df["im_path"] = train_df["im_name"].apply(lambda x: os.path.join(train_images_folder, x))
        test_df["im_path"] = test_df["im_name"].apply(lambda x: os.path.join(test_images_folder, x))

        # Extract image paths and labels
        train_image_paths = train_df["im_path"].tolist()
        train_labels = train_df["label"].tolist()
        test_image_paths = test_df["im_path"].tolist()
        test_labels = test_df["label"].tolist()

        # Preprocessing arguments
        preprocess_args = {
            "apply_laplacian": False,
            "apply_sobel": False,
            "apply_clahe": False,
            "augment_image": False,
        }

        # Load and preprocess images
        print("Loading and preprocessing training images...")
        train_images = self.load_images(train_image_paths, preprocess_args=preprocess_args, to_gray=False)
        print("Loading and preprocessing testing images...")
        test_images = self.load_images(test_image_paths, preprocess_args=preprocess_args, to_gray=False)

        # Convert labels to numpy arrays
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        # Print shapes of loaded data
        print(f"Training images shape: {train_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Testing images shape: {test_images.shape}")
        print(f"Testing labels shape: {test_labels.shape}")

        return train_images, train_labels, test_images, test_labels, test_image_paths




if __name__ == "__main__":
    processing = ImageProcessing(path="../data")

    x, y, tx, ty,test_image_paths = processing.main()
    # model = MajorityVoteModel(x, y, tx, ty)
    # model.train_and_eval(test_image_paths)
    model = ModelXGBoost(x, y, tx, ty)
    paras = {
        "reshape" : False,
        "hog": True,
        "lbp" : False,
        "Linear_PCA": False,
        "rfe" : True,
        "Kernal_PCA" : False,
        "Approx_Kernal_PCA" : False,
        "TSNE" : False,
        "Spectral_Embedding": False,
        "UMAP" : False,
    }
    model.train_and_eval(test_image_paths, **paras)