# Dataset
import tensorflow as tf
print(tf.__version__)
data_dir = r'D:\mini project\brain tumor\stylegan\Tfrecords(Pituitory tumor)/dataset.tfrecords'

# Training parameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10000

# Other parameters
network_type = 'stylegan2'  # Adjust based on available model types
resolution = 256  # Adjust based on desired image resolution
augment = True  # Enable data augmentation if needed
