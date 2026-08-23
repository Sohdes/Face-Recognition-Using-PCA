import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parameter
# -------------------------------------------------
# Path to the dataset
DATASET_PATH = "CroppedYale"
# Path to an external test image
TESTING_IMAGE = "testing.pgm"

# Unified size for all images after preprocessing
IMG_SIZE = (80, 80)
# Number of principal components (Eigenfaces) to keep
K = 80

# Fraction of data used for testing
TEST_RATIO = 0.2
# Distance threshold to classify a face as "Unknown"
THRESHOLD = 20
# -------------------------------------------------
# Function: Load dataset
# Reads all images, preprocesses them, and returns
# flattened vectors along with their labels.
# -------------------------------------------------
def load_images(path):
    images = []
    labels = []
    # Iterate through all subfolders (each folder = one person)
    for person in sorted(os.listdir(path)):
        person_path = os.path.join(path, person)

        if not os.path.isdir(person_path):
            continue
        # Read all images
        for img_name in os.listdir(person_path):

            if not img_name.lower().endswith(".pgm"):
                continue

            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # illumination normalization
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.equalizeHist(img)

            # Normalize pixel to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Flatten to 1D feature vector
            images.append(img.flatten())
            labels.append(person)

    return np.array(images), np.array(labels)


print("Loading dataset...")
# Load dataset into memory
X, y = load_images(DATASET_PATH)
print("Total images:", len(X))


# -------------------------------------------------
# Train/Test split
# Randomly split the dataset into training and testing sets
# -------------------------------------------------
np.random.seed(42)  # Ensures reproducibility
indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(len(X) * (1 - TEST_RATIO))
train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
print("Train images:", len(X_train))
print("Test images:", len(X_test))

# -------------------------------------------------
# Mean face computation
# Subtracting the mean ensures all images are centered.
# -------------------------------------------------
mean_face = np.mean(X_train, axis=0)
A_train = X_train - mean_face
A_test = X_test - mean_face

# -------------------------------------------------
# Covariance trick
# Instead of computing A*A^T (very large), we compute A*A^T in
# the smaller N×N space and later project back to the original space.
# -------------------------------------------------
print("Computing covariance matrix...")
L = A_train @ A_train.T  # N × N covariance surrogate matrix
# -------------------------------------------------
# Eigen decomposition
# Compute eigenvalues and eigenvectors of matrix L
# -------------------------------------------------
print("Eigen decomposition...")

eigvals, eigvecs = np.linalg.eigh(L)

idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
eigvecs = eigvecs[:, :K]

# keep corresponding eigenvalues
eigvals = eigvals[:K]

# -------------------------------------------------
# Compute Eigenfaces
# -------------------------------------------------
print("Computing eigenfaces...")

eigenfaces = A_train.T @ eigvecs
eigenfaces /= np.linalg.norm(eigenfaces, axis=0, keepdims=True) + 1e-8

# -------------------------------------------------
# Whitening (IMPORTANT FIX)
# -------------------------------------------------
sqrt_eig = np.sqrt(eigvals + 1e-8)

# Project training images (whitened space)
weights_train = (A_train @ eigenfaces) / sqrt_eig

# -------------------------------------------------
# Compute intra/inter distances (for threshold)
# -------------------------------------------------
train_projections = weights_train

intra_distances = []
inter_distances = []

for i in range(len(X_train)):
    phi = X_train[i] - mean_face

    w = (phi @ eigenfaces) / sqrt_eig

    distances = np.linalg.norm(train_projections - w, axis=1)

    distances[i] = np.inf  # remove self match
    nn_idx = np.argmin(distances)

    if y_train[nn_idx] == y_train[i]:
        intra_distances.append(np.min(distances))
    else:
        inter_distances.append(np.min(distances))

intra_distances = np.array(intra_distances)
inter_distances = np.array(inter_distances)

# -------------------------------------------------
# Robust threshold (percentile-based)
# -------------------------------------------------
THRESHOLD = np.percentile(intra_distances, 95)

print("Auto Threshold:", THRESHOLD)

# -------------------------------------------------
# Recognition function
# -------------------------------------------------
def recognize(img_vector):

    phi = img_vector - mean_face

    w = (phi @ eigenfaces) / sqrt_eig

    distances = np.linalg.norm(weights_train - w, axis=1)

    idx = np.argmin(distances)
    min_dist = distances[idx]

    if min_dist > THRESHOLD:
        return "Unknown", idx, min_dist

    return y_train[idx], idx, min_dist
# Accuracy
# -------------------------------------------------
print("\nEvaluating on test set...")
correct = 0
for i in range(len(X_test)):
    pred, _, _ = recognize(X_test[i])
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print("Test images:", len(X_test))
print("Correct:", correct)
print("Accuracy:", accuracy * 100, "%")
# -------------------------------------------------
# Test with an sample (testing) image
# -------------------------------------------------
print("\nTesting image:", TESTING_IMAGE)
img = cv2.imread(TESTING_IMAGE, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, IMG_SIZE)
img = cv2.equalizeHist(img)
img = img.astype(np.float32) / 255.0
img_vector = img.flatten()
person, idx, distance = recognize(img_vector)
print("\nPrediction:", person)
print("Distance:", distance)

# Retrieve the closest training image
closest = X_train[idx].reshape(IMG_SIZE)

# -------------------------------------------------
# Visualization: Testing image vs closest match
# -------------------------------------------------
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Testing Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(closest, cmap="gray")
plt.title("Closest Match\n" + person)
plt.axis("off")
plt.show()

# -------------------------------------------------
# Visualization: First 5 Eigenfaces
# -------------------------------------------------
plt.figure(figsize=(10,4))

for i in range(5):
    face = eigenfaces[:, i].reshape(IMG_SIZE)

    # Normalize for better visualization
    face = (face - face.min()) / (face.max() - face.min())

    plt.subplot(1,5,i+1)
    plt.imshow(face, cmap="gray")
    plt.title("Eigenface " + str(i+1))
    plt.axis("off")
plt.show()
