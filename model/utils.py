import torch
from torch.distributions.transforms import Transform
from torch.distributions import constraints
from torch.nn.functional import softplus
import math
LOGPI = math.log(math.pi)

class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = 1/b\log(1 + \exp(b x))`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __init__(self, beta: int = 1.5):
        super(SoftplusTransform, self).__init__()
        self.beta = beta
        self._cache_log_detJ = None

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        self._cache_log_detJ = -softplus(-x, self.beta)*self.beta
        return softplus(x, self.beta)

    def _inverse(self, y):
        return torch.log(torch.exp(y*self.beta)-1)/self.beta

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x, self.beta)*self.beta
    


def conditional(Xnew, X, kernel, f_loc, f_scale_tril=None, Lff=None, full_cov=False,
                whiten=False, jitter=1e-6):

    N = X.size(0)
    M = Xnew.size(0)
    latent_shape = f_loc.shape[:-1]

    if Lff is None:
        Kff = kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += jitter  # add jitter to diagonal
        Lff = torch.linalg.cholesky(Kff)
    Kfs = kernel(X, Xnew)

    # convert f_loc_shape from latent_shape x N to N x latent_shape
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    # convert f_loc to 2D tensor for packing
    f_loc_2D = f_loc.reshape(N, -1)
    if f_scale_tril is not None:
        # convert f_scale_tril_shape from latent_shape x N x N to N x N x latent_shape
        f_scale_tril = f_scale_tril.permute(-2, -1, *range(len(latent_shape)))
        # convert f_scale_tril to 2D tensor for packing
        f_scale_tril_2D = f_scale_tril.reshape(N, -1)

    if whiten:
        v_2D = f_loc_2D
        W = Kfs.triangular_solve(Lff, upper=False)[0].t()
        if f_scale_tril is not None:
            S_2D = f_scale_tril_2D
    else:
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril_2D), dim=1)

        Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
        # unpack
        v_2D = Lffinv_pack[:, :f_loc_2D.size(1)]
        W = Lffinv_pack[:, f_loc_2D.size(1):f_loc_2D.size(1) + M].t()
        if f_scale_tril is not None:
            S_2D = Lffinv_pack[:, -f_scale_tril_2D.size(1):]

    loc_shape = latent_shape + (M,)
    loc = W.matmul(v_2D).t().reshape(loc_shape)

    if full_cov:
        Kss = kernel(Xnew)
        Qss = W.matmul(W.t())
        cov = Kss - Qss
    else:
        Kssdiag = kernel(Xnew, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        # Theoretically, Kss - Qss is non-negative; but due to numerical
        # computation, that might not be the case in practice.
        var = (Kssdiag - Qssdiag).clamp(min=0)

    if f_scale_tril is not None:
        W_S_shape = (Xnew.size(0),) + f_scale_tril.shape[1:]
        W_S = W.matmul(S_2D).reshape(W_S_shape)
        # convert W_S_shape from M x N x latent_shape to latent_shape x M x N
        W_S = W_S.permute(list(range(2, W_S.dim())) + [0, 1])

        if full_cov:
            St_Wt = W_S.transpose(-2, -1)
            K = W_S.matmul(St_Wt)
            cov = cov + K
        else:
            Kdiag = W_S.pow(2).sum(dim=-1)
            var = var + Kdiag
    else:
        if full_cov:
            cov = cov.expand(latent_shape + (M, M))
        else:
            var = var.expand(latent_shape + (M,))

    return (loc, cov) if full_cov else (loc, var)

class ATanTransform(Transform):
    r"""
    Bijective transform via the mapping :math:`y = \text{ELU}(x)`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.x0 = 2.0
        self.scale = 2.0
        super().__init__()

    def __eq__(self, other):
        return isinstance(other, ATanTransform)

    def _call(self, x):
        return (torch.arctan((x-self.x0)/self.gamma)/torch.pi + 0.5)*self.scale

    def _inverse(self, y, eps=1e-8):
        return self.gamma*torch.tan(torch.pi*((y+ eps)/self.scale-0.5)) + self.x0

    def log_abs_det_jacobian(self, x, y):
        return math.log(self.scale)-LOGPI - torch.log(1+((x-self.x0)/self.gamma)**2)-math.log(self.gamma)
