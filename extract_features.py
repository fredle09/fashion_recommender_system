import os
import argparse
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import pickle

# Set logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load the ResNet50 model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def main(image_dir, embeddings_file, filenames_file):
    # Load filenames
    filenames = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    feature_list = []

    # Load existing embeddings if available
    if os.path.exists(embeddings_file):
        feature_list = pickle.load(open(embeddings_file, "rb"))
        processed_files = pickle.load(open(filenames_file, "rb"))
    else:
        processed_files = []

    # Create a set for faster lookup
    processed_set = set(processed_files)

    try:
        # Process each file
        for file in tqdm(filenames):
            if file not in processed_set:
                feature_list.append(extract_features(file, model))
                processed_files.append(file)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving progress...")
        pickle.dump(feature_list, open(embeddings_file, "wb"))
        pickle.dump(processed_files, open(filenames_file, "wb"))
        print("Progress saved. Exiting...")

    else:
        # Final save if no interrupt occurs
        pickle.dump(feature_list, open(embeddings_file, "wb"))
        pickle.dump(processed_files, open(filenames_file, "wb"))
        print("Processing completed and progress saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from images using ResNet50."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=False,
        default="images",
        help="Directory containing images to process.",
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        required=False,
        default="embeddings.pkl",
        help="File to save image embeddings.",
    )
    parser.add_argument(
        "--filenames_file",
        type=str,
        required=False,
        default="filenames.pkl",
        help="File to save processed filenames.",
    )

    args = parser.parse_args()
    main(args.image_dir, args.embeddings_file, args.filenames_file)
