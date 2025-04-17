import os
import cv2
import pandas as pd
import numpy as np
from model_xgboost import ModelXGBoost
from skimage.color import rgb2gray, rgb2hsv
from skimage import io, transform


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

    def normalize_image(self, img):
        """
        Normalize the image pixel values to the range [0, 1].
        """
        return img / 255.0

    def preprocess_image(self, img, **kwargs):
        """
        Preprocess an image by normalizing and optionally converting to HSV.
        """
        img = self.normalize_image(img)

        if kwargs.get("hsv", False):
            # Convert the image from RGB to HSV.
            # Ensure that the image is in proper range for rgb2hsv (i.e. [0, 1]).
            if img.ndim == 3:
                img = rgb2hsv(img)
        return img

    def load_images(self, image_paths, preprocess_args=None):
        """
        Load and preprocess all images from the given paths.
        """
        images = []
        for path in image_paths:
            try:
                # Load image using skimage.io which defaults to RGB order.
                img = io.imread(path)
                img = np.array(img)  # Ensure it is a NumPy array.
                # Preprocess image. Pass in any extra parameters (e.g., hsv=True).
                if preprocess_args:
                    img = self.preprocess_image(img, **preprocess_args)
                else:
                    img = self.preprocess_image(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")

        return np.array(images)

    def augment_image(self, image):
        if np.random.rand() > 0.8:
            image = np.fliplr(image)

        if np.random.rand() > 0.8:
            image = np.flipud(image)

        # Random rotation between -15 and 15 degrees
        angle = np.random.uniform(-15, 15)
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return image

    def oversample_data(self, images, labels, target_count):
        """
        Oversample the dataset with random augmentation to reach the target number of samples.
        """
        current_count = images.shape[0]
        if current_count >= target_count:
            return images, labels

        # Determine how many copies to generate.
        oversample_factor = target_count // current_count  # e.g. 200000/50000 = 4

        augmented_images = [images]
        augmented_labels = [labels]

        # Generate additional samples using augmentation.
        for i in range(oversample_factor - 1):
            aug_imgs = []
            for idx in range(current_count):
                orig_image = images[idx]
                aug_image = self.augment_image(orig_image)
                aug_imgs.append(aug_image)
            augmented_images.append(np.array(aug_imgs))
            augmented_labels.append(labels)

        all_images = np.concatenate(augmented_images, axis=0)
        all_labels = np.concatenate(augmented_labels, axis=0)

        # In case the total doesn't match exactly target_count, randomly subselect extra samples.
        total_count = all_images.shape[0]
        if total_count < target_count:
            extra_needed = target_count - total_count
            idxs = np.random.choice(current_count, extra_needed, replace=True)
            extra_images = images[idxs]
            extra_labels = labels[idxs]
            all_images = np.concatenate([all_images, extra_images], axis=0)
            all_labels = np.concatenate([all_labels, extra_labels], axis=0)
        elif total_count > target_count:
            # If we overshot the target, randomly select target_count samples.
            idxs = np.random.choice(total_count, target_count, replace=False)
            all_images = all_images[idxs]
            all_labels = all_labels[idxs]

        return all_images, all_labels

    def main(self):
        """
        Main method to load images, preprocess them, perform oversampling, and return the processed data.
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
        train_df["im_path"] = train_df["im_name"].apply(
            lambda x: os.path.join(train_images_folder, x)
        )
        test_df["im_path"] = test_df["im_name"].apply(
            lambda x: os.path.join(test_images_folder, x)
        )

        # Extract image paths and labels
        train_image_paths = train_df["im_path"].tolist()
        train_labels = train_df["label"].tolist()
        test_image_paths = test_df["im_path"].tolist()
        test_labels = test_df["label"].tolist()

        # Preprocessing arguments. Set hsv=True to convert to HSV.
        preprocess_args = {"hsv": True}

        # Load and preprocess images
        print("Loading and preprocessing training images...")
        train_images = self.load_images(
            train_image_paths, preprocess_args=preprocess_args
        )
        print("Loading and preprocessing testing images...")
        test_images = self.load_images(
            test_image_paths, preprocess_args=preprocess_args
        )

        # Convert labels to numpy arrays.
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        target_train_count = 250000
        print(
            f"Oversampling training data from {train_images.shape[0]} to {target_train_count} samples..."
        )
        train_images, train_labels = self.oversample_data(
            train_images, train_labels, target_train_count
        )

        # Print data shapes.
        print(f"Training images shape: {train_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Testing images shape: {test_images.shape}")
        print(f"Testing labels shape: {test_labels.shape}")

        return train_images, train_labels, test_images, test_labels, test_image_paths


if __name__ == "__main__":
    processing = ImageProcessing(path="data")
    x, y, tx, ty, test_image_paths = processing.main()

    # Create and train your model.
    model = ModelXGBoost(x, y, tx, ty)
    paras = {
        "hog": True,
        "lbp": True,
        "Linear_PCA": True,
        "Kernal_PCA": False,
        "Approx_Kernal_PCA": False,
        "TSNE": False,
        "Spectral_Embedding": False,
        "UMAP": False,
    }
    model.train_and_eval(test_image_paths, **paras)
