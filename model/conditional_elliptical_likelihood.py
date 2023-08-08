import math
import torch
from torch.nn import Parameter
from torch.distributions import constraints
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.nn.functional import softplus

import pyro.distributions as dist
from pyro.distributions.torch_transform import TransformModule
from pyro import sample, module
from .spline import ConditionalSpline
from .densenet import DenseNN

from pyro.contrib.gp.likelihoods.likelihood import Likelihood


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
        self.beta =  Parameter(torch.tensor(2.8), requires_grad=True)
        self.scale = beta

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        self._cache_log_detJ = -softplus(-x*self.beta.clamp(min=2.5, max = 4.0))
        return softplus(x*self.beta.clamp(min=2.5, max = 4.0))/self.beta.clamp(min=2.5, max = 4.0)/self.scale

    def _inverse(self, y):   
        return y + (y*self.beta.clamp(min=2.5, max = 4.0)*self.scale).neg().expm1().neg().log()/self.beta.clamp(min=2.5, max = 4.0)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x*self.beta.clamp(min=2.5, max = 4.0))/self.scale



class Elliptical(dist.TorchDistribution):

    arg_constraints = {'loc': constraints.real, 'var': constraints.positive}
    support = constraints.positive
    has_rsample = False

    def __init__(self, x_dim, validate_args=None, num_bins = 7, integration_steps = 512, beta = 1.5, device = "cpu"):    
        self.loc = torch.zeros(1)
        self.var =  torch.ones(1)
        self.X =  torch.ones(1)
        self.device = device
        num_bins = num_bins
        self.loc, self.var = broadcast_all(self.loc, self.var)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        self.integration_steps= integration_steps
        
        self.out_trans =SoftplusTransform(beta = beta)
        self.base_dist =torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1).to(device), torch.ones(1).to(device)),1)
        param_dims = [num_bins, num_bins, num_bins - 1]

        hypernet = DenseNN(x_dim , [128, 128], param_dims).to(device)

        self.spline_trans = ConditionalSpline(hypernet, 1, num_bins, bound = 3.0, order="quadratic").to(device)
        super(Elliptical, self).__init__(batch_shape, event_shape, validate_args)


    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        self.get_spline_distribution()
        w = self.distr.sample(sample_shape)
        X = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + X* (torch.rsqrt(w   +  self.var))

    @torch.no_grad()
    def sample_pw(self, sample_shape=torch.Size()):
        self.get_spline_distribution()
        return self.distr.sample(sample_shape)

    def log_prob(self, value):

        self.get_spline_distribution()
        w = torch.linspace(1e-4, 3.0, self.integration_steps).to(self.device)
        w_rep = w.view(-1,1).repeat(1,self.X.size(0) ).unsqueeze(-1)
        w_log_prob = self.distr.log_prob(w_rep).squeeze(-1)
        w_log_prob_norm =  w_log_prob
        w_log_prob_norm[0,:] = w_log_prob_norm[0,:] -math.log(2)
        w_log_prob_norm[-1,:] = w_log_prob_norm[-1,:] -math.log(2)

        log_prob_norm = torch.logsumexp(w_log_prob_norm, dim = 0) + math.log((w[1]-w[0]))
        u = (value.view(-1,1) - self.loc.view(-1,1))**2 / ( w.view(1,-1))
        log_prob = -0.5*u- 0.5*torch.log(  w.view(1,-1))-0.5*math.log(2*torch.pi) + w_log_prob.T 

        log_prob[:,0] = log_prob[:,0] - math.log(2.0)
        log_prob[:,-1] = log_prob[:,-1] - math.log(2.0)     

        return torch.logsumexp(log_prob, dim = 1)+ torch.log((w[1]-w[0])) - log_prob_norm 
    


    def log_prob_eval(self, value):

        self.get_spline_distribution()
        w = torch.linspace(1e-4, 3.0, self.integration_steps).to(self.device)
        w_rep = w.view(-1,1).repeat(1,self.X.size(0) ).unsqueeze(-1)
        w_log_prob = self.distr.log_prob(w_rep).squeeze(-1)
        w_log_prob_norm =  w_log_prob
        w_log_prob_norm[0,:] = w_log_prob_norm[0,:] -math.log(2)
        w_log_prob_norm[-1,:] = w_log_prob_norm[-1,:] -math.log(2)

        log_prob_norm = torch.logsumexp(w_log_prob_norm, dim = 0) + math.log((w[1]-w[0]))
        u = (value.view(-1,1) - self.loc.view(-1,1))**2 / ( w.view(1,-1))
        log_prob = -0.5*u- 0.5*torch.log(  w.view(1,-1))-0.5*math.log(2*torch.pi) + w_log_prob.T 

        log_prob[:,0] = log_prob[:,0] - math.log(2.0)
        log_prob[:,-1] = log_prob[:,-1] - math.log(2.0)     
        return torch.logsumexp(log_prob, dim = 1)+ torch.log((w[1]-w[0])) - log_prob_norm 
    

    def get_spline_distribution(self):
        self.distr = dist.ConditionalTransformedDistribution( self.base_dist, [ self.spline_trans,  self.out_trans,]).condition(self.X)





class ConditionalEllipticalLikelihood(Likelihood):
    """
    Implementation of Gaussian likelihood, which is used for regression problems.
    Gaussian likelihood uses :class:`~pyro.distributions.Normal` distribution.
    :param torch.Tensor variance: A variance parameter, which plays the role of
        ``noise`` in regression problems.
    """

    def __init__(self,x_dim, beta = 1.5, device = "cpu"):
        super().__init__()

        self.elliptical_distribution = Elliptical(x_dim, beta = beta, device = device)

    def forward(self, X, f_loc, f_var  = None, y = None):
        module("spline_tran", self.elliptical_distribution.spline_trans)
        self.elliptical_distribution.loc = f_loc
        self.elliptical_distribution.var = f_var
        self.elliptical_distribution.X = X

        y_dist = self.elliptical_distribution.expand_by(y.shape[: -f_loc.dim()]).to_event(y.dim())
        return sample(self._pyro_get_fullname("y"), y_dist, obs=y)

    def parameters(self):
        return self.elliptical_distribution.spline_trans.parameters()
    
    def reset(self):
        self.elliptical_distribution.distr.clear_cache()

    def get_spline_state_dict(self):
        return self.elliptical_distribution.spline_trans.state_dict()

    def load_spline_state_dict(self, state_dict):
        self.elliptical_distribution.spline_trans.load_state_dict(state_dict)

    def get_parameters(self):
        return self.elliptical_distribution.spline_trans.parameters()