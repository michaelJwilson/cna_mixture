import torch
import torch.nn.functional as F


def negative_binomial_log_pmf(k, r, p):
    log_pmf = (
        torch.lgamma(k + r)
        - torch.lgamma(k + 1)
        - torch.lgamma(r)
        + r * torch.log(p)
        + k * torch.log(1 - p)
    )
    return log_pmf


k = torch.tensor(k, requires_grad=False)  # Number of successes
mean = torch.tensor(mu, requires_grad=True)  # Mean of the distribution
var = torch.tensor(var, requires_grad=True)  # Variance of the distribution

r = mean**2 / (var - mean)
p = mean / var

log_pmf = negative_binomial_log_pmf(k, r, p)
log_pmf.backward()

grad_mean = mean.grad
grad_var = var.grad

print("Log PMF:", log_pmf.item())
print("Gradient with respect to mean:", grad_mean.item())
print("Gradient with respect to variance:", grad_var.item())
