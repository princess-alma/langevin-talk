import os
import gzip
import numpy as np
import requests

# Helper functions to load MNIST from local files
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int64)

def get_mnist_dataset_local(data_dir="./data/MNIST/raw"):
    train_images = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_images = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return (train_images, train_labels), (test_images, test_labels)

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

def download_mnist_files(data_dir="./data/MNIST/raw"):
    os.makedirs(data_dir, exist_ok=True)
    for filename, url in MNIST_URLS.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Saved {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                # Try alternative URL from Kaggle
                alt_url = f"https://www.kaggle.com/datasets/hojjatk/mnist-dataset/download/{filename}"
                print(f"Trying alternative source...")
                continue
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    download_mnist_files()
    
    # Loading MNIST data manually from local .gz files
    (train_images, train_labels), (test_images, test_labels) = get_mnist_dataset_local()
    print(f"Train samples: {len(train_images)} | Test samples: {len(test_images)}")