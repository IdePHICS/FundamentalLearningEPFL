from datasets import load_dataset

import torch
from torchvision import transforms


dataset = load_dataset("DamarJati/Face-Mask-Detection",split='train')

# Step 1: Check if MPS is available and set the device
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

# Step 3: Split the dataset into train and test sets
split_dataset = dataset.train_test_split(test_size=0.2)

# Step 4: Define a transform function
def transform(batch):
    # Resize images to a fixed size, e.g., 128x128 pixels, and convert to numpy array
    resize_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize all images to 128x128
        transforms.ToTensor()           # Convert PIL images or arrays to PyTorch tensors
    ])

    # Apply transformations to images and move them to the device
    batch['image'] = [resize_transform(img).to(device) for img in batch['image']]
    batch['label'] = torch.tensor(batch['label'], dtype=torch.long).to(device)
    return batch

# Apply the transformation to the dataset
split_dataset = split_dataset.map(transform, batched=True)

# Step 5: Set the format to torch
split_dataset.set_format(type='torch', columns=['image', 'label'])

final_dataset = split_dataset.shuffle(seed=2203)
split_dataset.save_to_disk('data')

# Step 6: save in numpy format X_train, y_train, X_test, y_test
# Save the train and test sets as numpy arrays
import numpy as np
train_set = final_dataset['train']
test_set = final_dataset['test']
X_train, y_train = np.array(train_set['image']), np.array(train_set['label'])
X_test, y_test = np.array(test_set['image']), np.array(test_set['label'])

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# Save X_train and y_train as numpy arrays in one single file
np.savez('data/train_full.npz', X=X_train, y=y_train)
np.savez('data/test.npz', X=X_test, y=y_test)

# Save just 1/50 of the dataset train
np.savez('data/train.npz', X=X_train[:len(X_train)//50], y=y_train[:len(y_train)//50])

X_extratrain = X_train[len(X_train)//50:]
y_extratrain = y_train[len(y_train)//50:]

# Extract only 1 labelled from extratrain
people_with_mask = (y_extratrain == 0)

y_extrapeople_with_mask = y_extratrain[people_with_mask]
X_extrapeople_with_mask = X_extratrain[people_with_mask]

np.savez('data/extra.npz', X=X_extrapeople_with_mask, y=y_extrapeople_with_mask)



