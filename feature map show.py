import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Load the data from the pickle file
file_path = 'face_encodings.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Extract encodings and names
encodings = np.array(data[0])  # List of feature vectors (NumPy arrays)
names = data[1]  # List of corresponding names

# Step 2: Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(encodings)

# Step 3: Create a scatter plot
plt.figure(figsize=(10, 8))
unique_names = list(set(names))

for name in unique_names:
    idx = [i for i, n in enumerate(names) if n == name]
    plt.scatter(reduced_data[idx, 0], reduced_data[idx, 1], label=name, s=50)

plt.title('Feature Map Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
