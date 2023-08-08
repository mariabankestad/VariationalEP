import math
import torch
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, broadcast_all
from pyro.distributions import TorchDistribution
from torch.distributions.transforms import Transform
from torch.nn import Parameter

from pyro.distributions.torch import TransformedDistribution
from torch.nn.functional import softplus
from .spline import Spline
LOGPI = math.log(math.pi)
from pyro.distributions.torch_transform import TransformModule

class ATanTransform(Transform):
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


# Backport of https://github.com/pytorch/pytorch/pull/52300
class SoftplusTransform(TransformModule):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = 1/b\log(1 + \exp(b x))`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1
    def __init__(self, beta = 1.5):
        super().__init__()
        self._cache_log_detJ = None
        self.beta =  Parameter(torch.tensor(2.5), requires_grad=True)
        self.scale = beta

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        #self.beta = clamp_preserve_gradients(self.beta,  min=2.5, max = 5)
        self._cache_log_detJ = -softplus(-x*self.beta.clamp(min=2.5, max = 4.0))
        return softplus(x*self.beta.clamp(min=2.5, max = 4.0))/self.beta.clamp(min=2.5, max = 4.0)/self.scale

    def _inverse(self, y):   
        #self.beta = clamp_preserve_gradients(self.beta,  min=2.5, max = 5)
        return y + (y*self.beta.clamp(min=2.5, max = 4.0)*self.scale).neg().expm1().neg().log()/self.beta.clamp(min=2.5, max = 4.0)

    def log_abs_det_jacobian(self, x, y):
        #self.beta = clamp_preserve_gradients(self.beta,  min=2.5, max = 5)
        return -softplus(-x*self.beta.clamp(min=2.5, max = 4.0))/self.scale


from torch.distributions import constraints


class Elliptical(TorchDistribution):

    arg_constraints = {'loc': constraints.real, 'var': constraints.positive}
    support = constraints.positive
    has_rsample = False

    def __init__(self, validate_args=None, integration_steps = 256, beta = 1.5,  device = "cpu"):    
        self.loc = torch.zeros(1)
        self.var =  torch.ones(1)
        self.device = device

        self.loc, self.var = broadcast_all(self.loc, self.var)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        self.integration_steps= integration_steps
        
        self.out_trans = SoftplusTransform(beta = beta)
        self.base_dist =torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1).to(device), torch.ones(1).to(device)),1)
        self.spline_trans = Spline(1, count_bins=9, order="quadratic").to(device)
        PATH = "model/spline_state_dict_likelihood.pickle"
        self.spline_trans.load_state_dict(torch.load(PATH))

        super(Elliptical, self).__init__(batch_shape, event_shape, validate_args)


    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p_spline = self.get_spline_distribution()
        w = p_spline.sample(sample_shape)
        X = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + X* (torch.rsqrt(w   +  self.var))

    @torch.no_grad()
    def sample_pw(self, sample_shape=torch.Size()):

        p_spline = self.get_spline_distribution()
        return p_spline.sample(sample_shape)

    def log_prob(self, value):
        p_spline = self.get_spline_distribution()
        w = torch.linspace(1e-4, 3.0, self.integration_steps).to(self.device)
        w_log_prob = p_spline.log_prob(w.view(-1,1))


        w_log_prob_norm =  w_log_prob
        w_log_prob_norm[0] = w_log_prob_norm[0] -math.log(2)
        w_log_prob_norm[-1] = w_log_prob_norm[-1] -math.log(2)

        log_prob_norm = torch.logsumexp(w_log_prob_norm, dim = 0) + math.log((w[1]-w[0]))
        if self.var is None:
            u = (value.view(-1,1) - self.loc.view(-1,1))**2 / ( w.view(1,-1))
            log_prob = -0.5*u- 0.5*torch.log(  w.view(1,-1))-0.5*math.log(2*torch.pi) + w_log_prob.flatten() 
        else:
            u = (value.view(-1,1) - self.loc.view(-1,1))**2 / (self.var.view(-1,1) + w.view(1,-1))
            log_prob = -0.5*u- 0.5*torch.log(self.var.view(-1,1) +  w.view(1,-1))-0.5*math.log(2*torch.pi) + w_log_prob.flatten() 
        log_prob[:,0] = log_prob[:,0] - math.log(2.0)
        log_prob[:,-1] = log_prob[:,-1] - math.log(2.0)

        return torch.logsumexp(log_prob, dim = 1)+ torch.log((w[1]-w[0])) - log_prob_norm 
        

    def log_prob_eval(self, value):

        p_spline = self.get_spline_distribution()
        w = torch.linspace(1e-4, 3.0, self.integration_steps).to(self.device)
        w_log_prob = p_spline.log_prob(w.view(-1,1))
        w_log_prob_norm =  w_log_prob
        w_log_prob_norm[0] = w_log_prob_norm[0] -math.log(2)
        w_log_prob_norm[-1] = w_log_prob_norm[-1] -math.log(2)
        log_prob_norm = torch.logsumexp(w_log_prob_norm, dim = 0) + math.log((w[1]-w[0]))
        u = (value.view(-1,1) - self.loc.view(-1,1))**2 / (self.var.view(-1,1) + w.view(1,-1))
        log_prob = -0.5*u- 0.5*torch.log(self.var.view(-1,1) +  w.view(1,-1))-0.5*math.log(2*torch.pi) + w_log_prob.flatten() 
        log_prob[:,0] = log_prob[:,0] - math.log(2.0)
        log_prob[:,-1] = log_prob[:,-1] - math.log(2.0)

        return torch.logsumexp(log_prob, dim = 1)+ torch.log((w[1]-w[0])) - log_prob_norm

    def get_spline_distribution(self):
        return TransformedDistribution( self.base_dist, [self.spline_trans,  self.out_trans])



import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.gp.likelihoods.likelihood import Likelihood
from pyro.nn.module import PyroParam


class EllipticalLikelihood(Likelihood):

    def __init__(self, beta = 1.5, device = "cpu"):
        super().__init__()

        self.elliptical_distribution = Elliptical(beta = beta, device = device)

    def forward(self, f_loc, f_var  = None, y = None):
        pyro.module("spline_tran", self.elliptical_distribution.spline_trans)
        self.elliptical_distribution.loc = f_loc
        self.elliptical_distribution.var = f_var
        y_dist = self.elliptical_distribution.expand_by(y.shape[: -f_loc.dim()]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)

    def parameters(self):
        return self.elliptical_distribution.spline_trans.parameters()
    
    def reset(self):
        self.elliptical_distribution.spline_trans.clear_cache()

    def get_spline_state_dict(self):
        return self.elliptical_distribution.spline_trans.state_dict()

    def load_spline_state_dict(self, state_dict):
        self.elliptical_distribution.spline_trans.load_state_dict(state_dict)

    def get_parameters(self):
        return self.elliptical_distribution.spline_trans.parameters()