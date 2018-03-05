# Hamiltonian adaptive moment estimation (Hadam)
new deep learning algorithms based Hamiltonian Monte Carlo and Adam (Hadam)

Prerequisites: 
* Pytorch
* CUDA

Usage Example:
* Run experiments of deep autoencoder on MNIST:
  -  python run_autoencoder_mnist_cuda.py --method "hadam" --lr 0.001 --gamma 10000 --fraction 0.5 --eta 0.1 
* Run experiments of ResNet18 on Cifar10:
  -  python run_cnn_cifar10_cuda.py --method "hadam" --lr 0.001 --gamma 100 --fraction 0.5 --eta 0.1 --bias_correction
