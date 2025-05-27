from torch.autograd import Variable
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PADRO_Loss(Lambda, g, H, delta, eps, K_sample, probabilities, data, target):
    """ Estimate objective function for a given Lambda, g, sig (RT-MLMC estimator)."""

    # Initialization
    data = data.squeeze(1)
    target = target.squeeze(1)
    # print(data.shape, target.shape, predict.shape)
    N, n1, n2 = data.shape
    _, k1, k2 = target.shape
    data = Variable(data)
    target = Variable(target)
    m = 2 ** K_sample

    # Sample from outer (x,y)-distribution (empirical data distribution)
    x_sample_out = data.to(device)
    y_sample_out = target.to(device)

    def integrand(H, g, x_sample_out, y_sample_out):
        """Estimate full objective  without randomized truncation. """
        # Randomly sample from inner x-distribution (empirical data distribution)
        x_in_indices = torch.randint(0, N, (N, m))
        x_sample_in = x_sample_out[x_in_indices]
        y_sample_in = y_sample_out.unsqueeze(1).repeat(1, m, 1, 1)

        y_in_flat = y_sample_in.view(N*m, 1, k1, k2)
        gy_flat = g(y_in_flat).squeeze(1)
        gy = gy_flat.view(N, m, k1, k2)

        loss = 0.5 * torch.linalg.norm(x_sample_in - gy, dim=(2, 3), ord=2) ** 2
        # print("Loss range:", loss.min().item(), loss.max().item())

        xcost = 0.5 * torch.linalg.norm(x_sample_out.unsqueeze(1) - x_sample_in, dim=(2, 3), ord=2) ** 2
        # ycost = 0.5 * torch.linalg.norm(y_sample_out.unsqueeze(1) - y_sample_in, dim=(2, 3), ord=2) ** 2
        ycost = 0
        cost = xcost + ycost
        # print("Cost range:", cost.min().item(), cost.max().item())

        exponent = (loss - Lambda * cost ) / (Lambda * delta)
        Obj_in_matrix = torch.exp(exponent)
        obj_out = torch.mean(Obj_in_matrix, dim=1)

        Obj_out_matrix = torch.log(obj_out)
        # print("Exponent range:", exponent.min().item(), exponent.max().item())
        # print("Obj_in_matrix range:", Obj_in_matrix.min().item(), Obj_in_matrix.max().item())
        # print("Obj_out_mtrx range:", Obj_out_matrix.min().item(), Obj_out_matrix.max().item())

        func_val = Lambda * delta * Obj_out_matrix
        # print("func_val range:", func_val.min().item(), func_val.max().item())

        # !!! Integral still needs to be calculated dependent on truncation !!!
        return func_val

    # Generated full (non-truncated) integrand
    Obj_in = integrand(H, g, x_sample_out, y_sample_out)
    # print("Obj_in range:", Obj_in.min().item(), Obj_in.max().item())
    # Calculate truncated integral
    if m == 1:
        Obj_out = torch.mean(Obj_in)
        # print("Obj_out(=full):", Obj_out)
    else:
        m1 = int(2 ** (K_sample - 1))

        Obj_1 = torch.mean(Obj_in)
        # print("Obj_full:", Obj_1)
        Obj_2 = torch.mean(Obj_in[:m1])
        Obj_3 = torch.mean(Obj_in[m1:])

        Obj_out = Obj_1 - 0.5 * (Obj_2 + Obj_3)
        # print("Obj_out:", Obj_out)

    Obj = Obj_out / probabilities[K_sample] + Lambda * eps
    # print("Obj:", Obj)

    return Obj