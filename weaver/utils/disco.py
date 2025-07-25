# Implementation of distance correlation (DisCo) between two variables


import torch


def distance_correlation(var_1, var_2, weight=None, power=1):
    # calculate distance correlation between var_1 and var_2.
    # implementation based on this one: https://github.com/gkasieczka/DisCo/tree/master,
    # corresponding to this paper: https://doi.org/10.1103/PhysRevD.103.035021
    # input arguments:
    # - var_1: first variable to decorrelate (e.g. mass).
    # - var_2: second variable to decorrelate (e.g. classifier output).
    # - weight: per-instance weight.
    #   note: sum of weights should add up to the number of instances.
    #   default: weight 1 for each instance.
    # - power: exponent used in calculating the distance correlation.
    # notes:
    # - var_1, var_2 and weight should all be 1D tensors
    #   with the same number of entries.
    
    # type conversion
    if not torch.is_tensor(var_1): var_1 = torch.from_numpy(var_1).float()
    if not torch.is_tensor(var_2): var_2 = torch.from_numpy(var_2).float()

    # initialize size and weights
    size = len(var_1)
    if weight is None:
        weight = torch.ones(size)
        # transfer weights to GPU
        if torch.cuda.is_available(): weight = weight.to('cuda')

    # make distance matrix for variable 1
    columns = var_1.view(-1, 1).repeat(1, size).view(size, size)
    columns_transpose = var_1.repeat(size, 1).view(size, size)
    dist_1 = (columns - columns_transpose).abs()
    # perform double centering
    row_avg = torch.mean(dist_1*weight, dim=1)
    rows = row_avg.repeat(size, 1).view(size, size)
    rows_transpose = row_avg.view(-1, 1).repeat(1, size).view(size, size)
    dist_1 = dist_1 - rows - rows_transpose + torch.mean(row_avg*weight)
    
    # make distance matrix for variable 2
    columns = var_2.view(-1, 1).repeat(1, size).view(size, size)
    columns_transpose = var_2.repeat(size, 1).view(size, size)
    dist_2 = (columns - columns_transpose).abs()
    # perform double centering
    row_avg = torch.mean(dist_2*weight, dim=1)
    rows = row_avg.repeat(size, 1).view(size, size)
    rows_transpose = row_avg.view(-1, 1).repeat(1, size).view(size, size)
    dist_2 = dist_2 - rows - rows_transpose + torch.mean(row_avg*weight)

    # calculate (squared) covariances
    cov_12 = torch.mean(torch.mean(dist_1 * dist_2 * weight, dim=1) * weight)
    cov_11 = torch.mean(torch.mean(dist_1 * dist_1 * weight, dim=1) * weight)
    cov_22 = torch.mean(torch.mean(dist_2 * dist_2 * weight, dim=1) * weight)

    # calculate correlation coefficient
    if power==1: dcorr = cov_12 / torch.sqrt(cov_11 * cov_22)
    elif power==2: dcorr = cov_12**2 / (cov_11 * cov_22)
    else: dcorr = (cov_12 / torch.sqrt(cov_11 * cov_22))**power
    
    return dcorr
