import torch

# Specify the path to your .pth file
checkpoint_path = "../checkpoints/genomic_multitarget_informer/checkpoint.pth"

# Load the checkpoint; if you're not using a GPU, map it to CPU
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Print the keys to see what's inside the checkpoint
print("Keys in checkpoint:")
for key in checkpoint.keys():
    print(key)

# Optionally, print the entire checkpoint content (if it's not too large)
print("\nFull checkpoint content:")
print(checkpoint)