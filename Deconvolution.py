import numpy as np
import torch
from PADRO_optimizer import PADRO_optimizer, PADRO_oracle, convolve
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf

"""
Example: find deconvolution operator using PADRO-framework.
"""
def deconvolution_matrix(kernel, input_shape):
    """
    Finds deconvolution matrix for a given kernel with 0 padding and stride 1.
    :param kernel: Kernel to find deconvolution matrix for.
    :param input_shape: Shape of data the forward kernel convolves.
    :return: deconvolution matrix
    """
    kernel_size = kernel.shape[0]
    x_height, x_width = input_shape
    y_height = x_height + kernel_size - 1
    y_width = x_width + kernel_size - 1
    output_pixels = y_height * y_width
    input_pixels = x_height * x_width

    conv_matrix = torch.zeros((output_pixels, input_pixels))

    for in_i in range(x_height):
        for in_j in range(x_width):
            input_idx = in_i * x_width + in_j

            for di in range(kernel_size):
                for dj in range(kernel_size):
                    output_i, output_j = in_i + di, in_j + dj
                    output_idx = output_i * y_width + output_j
                    conv_matrix[output_idx, input_idx] = kernel[di, dj]

    return conv_matrix


def gaussian_kernel(size, sigma):
    """ Generate Gaussian kernel

    Parameters:
    :param size: Size of kernel
    :param sigma: Standard deviation of Gaussian
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)


class MyDataset(Dataset):
    def __init__(self, Feature_Tr_tensor, Target_Tr_mat_tensor):
        self.Data = TensorDataset(Feature_Tr_tensor, Target_Tr_mat_tensor)

    def __getitem__(self, index):
        data, target = self.Data[index]

        return data, target, index

    def __len__(self):
        return len(self.Data)


def plotting():
    """ Helper function to plot convergence. """
    plt.rcParams["figure.dpi"] = 300  # Change default DPI

    # Plot convergence of g
    plt.figure()
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Extract the data for each pair of indices
            data = [gi[i][j] for gi in g_hist]
            plt.plot(data, markersize=2, linestyle='-', label=f'{i}{j}')

    plt.title(r'Entries of g')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.grid(True)
    # plt.legend()
    plt.ylim(-1, 1)

    # Save file
    g_file = os.path.join(folder_name, 'g plot.png')
    plt.savefig(g_file)

    # Plot convergence of sigma
    if not np.isscalar(sig0):
        plt.figure()
        colors = ['r', 'b', 'g', 'k']

        for i in range(n1*n2):
            for j in range(n1*n2):
                # Extract data for each pair of indices
                data = [si[i][j] for si in sig_hist]

                # Differentiate in diagonal
                if i == j:
                    plt.plot(data, markersize=2, linestyle='-', color=colors[i], label=f'{i}{j}')
                else:
                    plt.plot(data, markersize=2, linestyle='--', color=colors[i], label=f'{i}{j}')

        plt.title(r'Entries of $\Sigma$')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()

    else:
        plt.figure()
        plt.plot(sig_hist, color='b', markersize=2)
        plt.title(r'Value of $\sigma$')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)

    # Save file
    sig_file = os.path.join(folder_name, 'sig plot.png')
    plt.savefig(sig_file)

    # plt.show()

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(45)
    np.random.seed(28)

    # Generate data
    a = 0  # Bounds for uniform data distribution
    b = 1
    N = 150  # Nr of data points
    batch = 10
    n1 = 28  # Rows in data
    n2 = n1  # Columns in data
    k1 = 26  # Convolve 28x28 by 3x3 kernel -> 26x26
    k2 = k1

    # Import MNIST test data
    mnist = tf.keras.datasets.mnist
    (x_train_np, _), (_, _) = mnist.load_data()
    x_train_np = x_train_np/x_train_np.max()  # normalize
    x = torch.tensor(x_train_np[:N, :, :], dtype=torch.float32)
    # x = torch.rand(size=(N, n1, n2), dtype=torch.float32)

    # Forward operator
    kernel_size = 3
    H = torch.tensor(gaussian_kernel(kernel_size, 0.5), dtype=torch.float32)

    # Forward output
    Hx = convolve(H, x)

    # Noisy measurement
    sd = 0  # If you want to consider a particular type of noise, choose  sd carefully so that randomness does not overpower the structure of Hx
    y = torch.randn([N, k1, k2]) * sd + Hx

    # Dataloader
    Dataset_tensor = MyDataset(x, y)
    feature_tensor1, target_tensor1 = Dataset_tensor.Data.tensors
    Dataloader_all = DataLoader(Dataset_tensor, batch_size=batch, shuffle=True)

    # Initialize parameters
    delta = 0.1  # Sinkhorn regularization parameter
    eps = 0.001  # Radius Wasserstein-ball
    lamb_bound_l = 0.1  # Lower-bound Lambda (dual variable)
    lamb_bound_u = 1  # Upper-bound Lambda
    sig_bound = 2  # Upper bound st. dev. / Cholesky parameter covariance matrix
    lr_sig = 0.01 # Learning rate sigma
    lr_g = 0.001  # Learning rate g (approximate inverse)
    epochs = 50  # Number of epochs
    Problem = 'deconvolution'  # Problem type, either 'matrix_inversion' or 'deconvolution'

    # Initial guesses
    g0 = np.random.uniform(-1, 1, size=(kernel_size, kernel_size))  # Initial guess g
    sig0 = 0.01  # Initial guess variance Gaussian

    # Create folder with correct name
    folder_name = os.path.join('Results', 'Deconvolution')
    folder_name = os.path.join(folder_name, 'eps={}'.format(eps))
    os.makedirs(folder_name, exist_ok=True)

    before_optimizing = time.time() # Log time
    Lambda, g, sig, opt, g_hist, loss_hist_g, sig_hist = PADRO_optimizer(Problem, lamb_bound_l, lamb_bound_u, g0, H, sig0, sig_bound, delta, Dataloader_all, eps, lr_sig, lr_g,
                                                                         maxiterout=100, tol=1e-3, epochs=epochs, silence=False, K_sample_max=4, test_iterations=1)

    # If optimal Lambda is saved, quickly re-solve for fixed Lambda for e.g. new plots
    # Lambda = 0.4409905507088448
    # opt, g, g_hist, loss_hist_g, sig, sig_hist = PADRO_oracle(Problem, Lambda, g0, H, sig0, sig_bound, delta, Dataloader_all, eps, lr_sig, lr_g,
    #              epochs=epochs, silence=False)

    after_optimizing = time.time() # Log time

    # Save results and parameters to .txt file
    Results = "Opt: {}, Lambda: {}, \n g: {}, \n sig:{}. \n Optimization time: {:.2f}s".format(opt, Lambda, g, sig, after_optimizing - before_optimizing)
    parameters = f"""
    a = {a}
    b = {b}
    N = {N}
    n1 = {n1}
    n2 = {n2}
    sd = {sd}
    H =
    {H}
    g0 =
    {g0}
    delta = {delta}
    eps = {eps}
    lamb_bound_l = {lamb_bound_l}
    lamb_bound_r = {lamb_bound_u}
    sig0 =
    {sig0}
    sig_bound = {sig_bound}
    lr_sig = {lr_sig}
    lr_g = {lr_g}
    epochs = {epochs}
    batch size = {batch}
    """
    print(Results)

    results_file = os.path.join(folder_name, 'Results.txt')
    params_file = os.path.join(folder_name, 'Parameters.txt')
    g_file = os.path.join(folder_name, 'g.csv')
    f = open(results_file, 'w')
    f.write(Results)
    h = open(params_file, 'w')
    h.write(parameters)
    np.savetxt(g_file, g, delimiter=',',fmt="%f")

    # Plot convergence
    plotting()

