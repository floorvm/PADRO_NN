import numpy as np
import torch
from torch.autograd import Variable
from scipy.linalg import cholesky, LinAlgError

""" 
Optimizer for Perturbation-Aware DRO problem. 

Can be applied to matrix inversion or deconvolution. 
Possible perturbations are isotropic and anisotropic Gaussian.
Use PADRO_optimizer to optimize full problem. 
Use PADRO_oracle to optimize objective for a given Lambda (dual variable).
Also includes Sinkhorn-DRO for comparison purposes. Structure is similar but works only for anisotropic matrix inversion.
"""

def adjust_lr(optimizer, lr0, epoch):
    """ Adjust learning rate according to epoch. """
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def isscalar(tensor):
    """ Determine if tensor is scalar or not.

    :param tensor: Tensor to evaluate
    :return: True or False
    """
    return tensor.dim() == 0


def is_positive_definite(matrix):
    """ Determine if matrix is positive definite.

    :param matrix: Matrix to evaluate
    :return: True or False
    """
    try:
        # Attempt to compute Cholesky decomposition
        cholesky(matrix)
        return True
    except LinAlgError:
        # If error, matrix not positive definite
        return False


def convolve(H, x):
    """ Convolve data x with kernel H.

    Parameters:
    :param H: Kernel to convolve with
    :param x: Data to convolve

    Returns:
    :return: Hx: Convolved data
    """
    Hx_exp = torch.nn.functional.conv2d(x.unsqueeze(1), H.unsqueeze(0).unsqueeze(0))
    Hx = Hx_exp.squeeze(1)
    return Hx


def deconvolve(G, y):
    """ Deconvolve data y with kernel G.

    Parameters:
    :param G: Kernel to deconvolve with
    :param y: Data to deconvolve

    Returns:
    :return: Gy: Deconvolved data
    """
    Gy_exp = torch.nn.functional.conv_transpose2d(y, G.unsqueeze(0).unsqueeze(0))
    Gy = Gy_exp.squeeze(1)
    return Gy


def generate_obj(problem, Lambda, g, H, sig, L, delta, K_sample, probabilities, data, target):
    """ Estimate objective function for a given Lambda, g, sig (RT-MLMC estimator)."""

    # Determine and remember if sig is scalar (standard deviation) or not (covar matrix)
    scalar_sig = isscalar(sig)

    # Initialization
    N, n1, n2 = data.shape
    _, k1, k2 = target.shape
    data = Variable(data)
    target = Variable(target)
    m = 2 ** K_sample

    # Sample from outer (x,y)-distribution (empirical data distribution)
    x_sample_out = data
    y_sample_out = target

    # Determine scalar that is re-used log(frac)
    if scalar_sig:
        frac = 1 / (torch.pow(2 * torch.pi * (sig ** 2), 1 / (n1 * n2)))
    else:
        frac = 1 / torch.sqrt((2 * torch.pi) ** (n1 * n2) * torch.linalg.det(sig))
    logfrac = torch.log(frac)

    def integrand(H, g, x_sample_out, y_sample_out):
        """Estimate full objective  without randomized truncation. """

        # Randomly sample from inner x-distribution (empirical data distribution)
        x_in_indices = torch.randint(0, N, (m * N,))
        x_sample_in = x_sample_out[x_in_indices]

        # Determine forward output
        if problem == 'matrix_inversion':
            Hx = torch.matmul(H, x_sample_in)
        else:
            Hx = convolve(H, x_sample_in)

        # Reshape for tensor calculations
        Hx_exp = Hx.reshape([N, m, k1, k2])
        x_in_exp = x_sample_in.reshape([N, m, n1, n2])

        # Randomly sample from inner y-distribution (Gaussian with mean=Hx, st.dev./covar=sig)
        if scalar_sig:
            y_sample_in = torch.randn([N, m, m, k1, k2], dtype=torch.float32) * sig + Hx_exp.reshape([N, m, 1, k1, k2])

            # Compute rest term (dependent on sig)
            rest_matrix = logfrac - torch.linalg.norm(Hx_exp.unsqueeze(2) - y_sample_in, ord=2, dim=(-2, -1)) ** 2 / (2 * sig ** 2)
        else:
            y_in_vec = torch.randn([N, m, m, k1 * k2], dtype=torch.float32)  # Random standard Gaussian
            Hx_exp_vec = Hx_exp.reshape([N, m, 1, k1 * k2])
            y_in_vec = torch.matmul(y_in_vec, L.T) + Hx_exp_vec
            y_sample_in = y_in_vec.view(N, m, m, k1, k2) # Reshape for tensor calculations

            # Compute rest term (dependent on sig)
            diff_vec = y_in_vec - Hx_exp_vec
            product_vec = torch.matmul(diff_vec, torch.linalg.inv(sig).T)
            inner_product_vec = torch.bmm(diff_vec.view(m * m * N, 1, k1 * k2),
                                          product_vec.view(m * m * N, k1 * k2, 1)).view(N, m, m)
            rest_matrix = logfrac - (1 / 2) * inner_product_vec

        # Determine backward output
        if problem == 'matrix_inversion':
            gy = torch.matmul(g.unsqueeze(0).unsqueeze(0).unsqueeze(0), y_sample_in)
        else:
            y_reshaped = y_sample_in.view(m * m * N, 1, k1, k2)
            gy = deconvolve(g, y_reshaped)

            # If working with convolution matrix instead of kernel
            # y_reshaped = y_sample_in.view(m*m*N, k1*k2)
            # gy = torch.matmul(y_reshaped, g.T)

            gy = gy.view(N, m, m, n1, n2)  # Reshape for tensor calculations

        # Compute loss term
        loss_matrix = (1 / 2) * torch.linalg.norm(x_in_exp.reshape([N, m, 1, n1, n2]) - gy, ord=2, dim=(-2, -1)) ** 2

        # Compute cost term (sum of x-cost and y-cost)
        xcost = (1 / 2) * torch.linalg.norm(x_sample_out.unsqueeze(1) - x_in_exp, ord=2, dim=(-2, -1)) ** 2
        ycost = (1 / 2) * torch.linalg.norm(y_sample_out.unsqueeze(1).unsqueeze(1) - y_sample_in, ord=2, dim=(-2, -1)) ** 2
        cost_matrix = xcost.unsqueeze(-1) + ycost

        # Compute inner integrand
        Obj_in_matrix = loss_matrix - Lambda * cost_matrix - Lambda * delta * rest_matrix

        # Compute outer integrand
        Obj_out_matrix = torch.sum(Obj_in_matrix, dim=2)

        # !!! Integral still needs to be calculated dependent on truncation !!!
        return Obj_out_matrix

    # Generated full (non-truncated) integrand
    Obj_in = integrand(H, g, x_sample_out, y_sample_out)

    # Calculate truncated integral
    if m == 1:
        Obj_out = torch.mean(Obj_in)
    else:
        m1 = int(2 ** (K_sample - 1))

        Obj_1 = torch.mean(Obj_in)
        Obj_2 = torch.mean(Obj_in[:, :m1])
        Obj_3 = torch.mean(Obj_in[:, m1:])

        Obj_out = Obj_1 - 0.5 * (Obj_2 + Obj_3)

    Obj = Obj_out / probabilities[K_sample]

    return Obj


def PADRO_solver_RTMLMC(problem, Lambda, g0, H, sig0, sig_bound, delta, dataloader, lr_sig, lr_g,
                        epochs=100, K_sample_max=4, test_iterations=1, silence=True):
    """ Solve inner optimization problem and finds optimal g and sigma for given Lambda. """
    iter = 0

    # Initialization
    sig = torch.tensor(sig0, dtype=torch.float, requires_grad=False)
    g = torch.tensor(g0, dtype=torch.float, requires_grad=True)
    g = UNet(2).cuda()
    if not torch.is_tensor(H):
        H = torch.tensor(H, dtype=torch.float, requires_grad=False)
    optimizer_g = torch.optim.SGD([g], lr=lr_g)

    sig_hist = [sig.clone().detach().numpy()]
    g_hist = [g.clone().detach().numpy()]
    loss_hist = []

    # Determine & remember if sig is scalar (standard deviation) or not (covar matrix)
    scalar_sig = isscalar(sig)

    if scalar_sig:
        L = 0  # To avoid errors
        sig = torch.tensor(sig0, dtype=torch.float, requires_grad=True)
        optimizer_sig = torch.optim.SGD([sig], lr=lr_sig, maximize=True)
    else:
        L = torch.linalg.cholesky(sig).requires_grad_(True)
        optimizer_sig = torch.optim.SGD([L], lr=lr_sig, maximize=True)

    # Generate distribution for sampling variable K_sample of RT-MLMC estimator
    elements = np.arange(0, K_sample_max)
    probabilities = 0.5 ** elements
    probabilities = probabilities / np.sum(probabilities)

    for epoch in range(epochs):
        for _, (data, target, idx) in enumerate(dataloader):
            iter += 1

            # First, optimize sig
            K_sample = int(np.random.choice(list(elements), 1, list(probabilities)))  # Determine sampling variable

            # Optimize sig
            optimizer_sig.zero_grad()
            Obj_sig = generate_obj(problem, Lambda, g, H, sig, L, delta, K_sample, probabilities, data, target)
            Obj_sig.backward()
            optimizer_sig.step()

            # Check if sig is within bounds / positive-definite & non-singular
            if scalar_sig:
                with torch.no_grad():
                    sig.data.clamp_(min=-sig_bound, max=sig_bound)
            else:
                with torch.no_grad():
                    L.data.clamp_(min=-sig_bound, max=sig_bound)
                    sig = torch.matmul(L, L.T)
                    pos_def = is_positive_definite(sig.numpy())
                    if not pos_def or torch.linalg.det(sig) == 0:
                        print('Sig is singular or not positive definite')
                        L.data = torch.tril(L.data)
                        sig = torch.matmul(L, L.T)

            sig_clone = sig.clone().detach().numpy()
            sig_hist.append(sig.clone().detach().numpy())
            loss_hist.append(Obj_sig.item())


            # Next, optimize g
            K_sample = int(np.random.choice(list(elements), 1, list(probabilities))) # Determine sampling variable

            # Optimize g
            optimizer_g.zero_grad()
            Obj_g = generate_obj(problem, Lambda, g, H, sig, L, delta, K_sample, probabilities, data, target)
            Obj_g.backward()
            optimizer_g.step()

            with torch.no_grad():
                g.data.clamp_(min=-2, max=2)

            g_clone = g.clone().detach().numpy()
            g_hist.append(g_clone)
            loss_hist.append(Obj_g.item())

            # Every X iterations, log intermediate results
            if (silence == False) and (iter % test_iterations == 0):
                print("Iter: {}, Loss: {:.2f}, sig={}, top left g={}".format(iter, Obj_g.item(), sig_clone,
                                                                             g_clone[0][0]))

        # Adjust learning rate every epoch
        adjust_lr(optimizer_g, lr_g, epoch + 1)
        adjust_lr(optimizer_sig, lr_sig, epoch + 1)

    return g.detach().to_dense().numpy(), g_hist, Obj_g.item(), loss_hist, sig, sig_hist


def PADRO_oracle(problem, Lambda, g0, H, sig0, sig_bound, delta, dataloader, eps, lr_sig, lr_g,
                 epochs=100, K_sample_max=4, test_iterations=10, silence=False):
    """ Solve inner optimization of Perturbation-Aware DRO problem

    For a given value of Lambda (dual variable) returns optimal values for Lambda (dual variable),
    g (parameter in loss function to be optimized, i.e. inverse or kernel of deconvolution)
    and sigma (standard deviation or covariance matrix).

    """

    g, g_hist, Loss, loss_hist, sig, sig_hist = PADRO_solver_RTMLMC(problem, Lambda, g0, H, sig0, sig_bound, delta, dataloader,
                                                                    lr_sig, lr_g, epochs, K_sample_max)

    full_loss = Loss + Lambda * eps

    return full_loss, g, g_hist, loss_hist, sig, sig_hist


def PADRO_bisection_search(problem, ll, lr, g0, H, sig0, sig_bound, delta, dataloader, eps, lr_sig, lr_g,
                           maxiterout=100, tol=1e-5, epochs=100,
                           K_sample_max=4, test_iterations=10, silence=False):
    """ Bisection search finds optimal Lambda in outer optimization.

    Every iteration, algorithm solves inner optimization three times; for the end-points of the interval and the "mid-point".

    """

    # Initialization
    iter = 0
    lamb = ll
    xl = ll
    xr = lr

    def oracle(lamb):
        """ Wrapper function for oracle to remove repeating parameters. """
        f, g, g_hist, loss_hist, sig, sig_hist = PADRO_oracle(problem, lamb, g0, H, sig0, sig_bound, delta, dataloader, eps,
                                                              lr_sig, lr_g,
                                                              epochs, K_sample_max=K_sample_max)
        return f, g, g_hist, loss_hist, sig, sig_hist

    while iter <= maxiterout and (xr - xl) > tol:
        # Re-initialize every iteration
        iter += 1
        bl = 1 / 3 * (2 * xl + xr)
        br = 1 / 3 * (xl + 2 * xr)

        # Determine objective value for end-points and "mid-point"
        f_left, gl, _, _, _, _ = oracle(bl)
        f_right, gr, _, _, _, _ = oracle(br)
        f_lamb, glamb, _, _, _, _ = oracle(lamb)

        # Refine interval based on objective values
        if f_left <= f_right:
            xl = xl
            xr = br
            if f_left <= f_lamb:
                lamb = bl
            else:
                lamb = lamb
        else:
            xl = bl
            xr = xr
            if f_right <= f_lamb:
                lamb = br
            else:
                lamb = lamb

        # Every X iterations, log intermediate results
        if (silence == False) and (iter % test_iterations == 0):
            print("Lamb Iter: {}, lambda: {}".format(iter, lamb))

    # Calculate final values
    final_loss, final_g, g_hist, loss_hist, sig, sig_hist = oracle(lamb)

    return lamb, final_g, final_loss, g_hist, loss_hist, sig, sig_hist


def PADRO_optimizer(problem, ll, lr, g0, H, sig0, sig_bound, delta, dataloader, eps, lr_sig, lr_g,
                    maxiterout=100, tol=1e-5, epochs=100,
                    K_sample_max=4, test_iterations=10, silence=False):
    """ Solve Perturbation-Aware DRO problem.

    Returns optimal values for Lambda (dual variable),
    g (parameter in loss function to be optimized, i.e. inverse or kernel of deconvolution) and
    sigma (standard deviation or covariance matrix).

    Parameters:
    :param problem: Describes problem to be solved, either 'matrix_inversion' or 'deconvolution'
    :param ll: Lower bound for Lambda (dual variable)
    :param lr: Upper bound for Lambda
    :param g0: Initial guess for g
    :param H: Parameter of forward operator (matrix inversion: forward matrix, deconvolution: kernel of convolution)
    :param sig0: Initial guess for sigma
    :param sig_bound: Bound for sigma
    :param delta: Sinkhorn regularization parameter (geq 0). The higher, the more regularization.
    :param dataloader: Dataloader for optimization
    :param eps: Radius of Wasserstein-ball (geq 0). The higher, the more 'worst'-case.
    :param lr_sig: Learning rate sigma
    :param lr_g: Learning rate g
    :param maxiterout: Maximum number of iterations of bisection search
    :param tol: Convergence tolerance of bisection search (terminates if length of interval < tol)
    :param epochs: Number of epochs used to train
    :param K_sample_max: Max value of K_sample (random parameter for sampling in RT-MLMC estimator)
    :param test_iterations: If silence=False, will print intermediate results every test_iterations
    :param silence: Set to True to suppress intermediate results

    Returns:
    :return: lamb: Optimal value of Lambda (dual variable)
    :return: g: Optimal value of g (parameter in loss function to be optimized, i.e. inverse or kernel of deconvolution)
    :return: sig: Optimal value of sigma (standard deviation or covariance matrix)
    :return: loss: Optimal value of objective
    :return: g_hist: Loss history g
    :return: loss_hist: Loss history objective
    :return: sig_hist: Loss history sigma
    """

    if problem not in ('matrix_inversion', 'deconvolution'):
        print('Wrong problem type, should be either matrix_inversion or deconvolution.')
        return

    lamb, g, loss, g_hist, loss_hist, sig, sig_hist = PADRO_bisection_search(problem, ll, lr, g0, H, sig0, sig_bound, delta, dataloader, eps, lr_sig, lr_g,
                                                                             maxiterout, tol, epochs, K_sample_max, test_iterations, silence)

    return lamb, g, sig, loss, g_hist, loss_hist, sig_hist