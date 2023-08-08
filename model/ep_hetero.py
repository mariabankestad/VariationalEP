# This implementation is adapted in part from:
# * https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/gp/models/vsgp.py;

import torch
import pyro 
from torch.distributions import constraints
from torch.nn import Parameter

from pyro.distributions.util import eye_like
import pyro.distributions as dist
from pyro.nn.module import PyroParam, pyro_method
from pyro.contrib.gp.models.model import GPModel
from pyro.distributions.torch import TransformedDistribution
import pyro.poutine as poutine

from .conditional_elliptical_likelihood import ConditionalEllipticalLikelihood
from .utils import SoftplusTransform, conditional, ATanTransform


class SparseEPHetero(GPModel):
    def __init__(self,  X, y, Xu, kernel , learn_xu = True , jitter = 1e-5, 
                 variance = 0.1, mean_function = None, device = "cpu", beta = 1.5, learn_qxi= True, x_dim = None):
        super().__init__(X, y, kernel, mean_function, jitter)
        self.device = device
        self.X = X
        self.y = y
        self.kernel = kernel
        self.jitter = jitter
        self.variance = torch.tensor(1.0)
        self.Xu = Xu
        if learn_xu is True:
            self.Xu = Parameter(Xu)
        if x_dim is None:
            x_dim = X.size(1)
        M = self.Xu.size(0)
        f_loc_prior = self.Xu.new_zeros(M).to(device)
        self.f_loc = Parameter(f_loc_prior)
        self.out_trans = SoftplusTransform(beta = beta)
        f_scale_tril = eye_like(self.Xu, M).to(device)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
        f_scale_tril = torch.linalg.cholesky(Kuu.clone().detach())
        f_scale_tril = eye_like(self.Xu, M).to(device)
        self.f_scale_tril = PyroParam(f_scale_tril, constraints.lower_cholesky)
        self.num_data = X.size(0)
        self.likelihood = ConditionalEllipticalLikelihood(x_dim = x_dim, device = device).to(device)
        self.learn_qxi = learn_qxi
        self.define_splines(device=device)

        if learn_qxi:
            self.output_trans_xi = ATanTransform()

    def get_q_xi(self):
        return TransformedDistribution( self.base_dist_qxi, [self.spline_guide_xi, self.output_trans_xi])

    @pyro_method    
    def model(self):

        if self.learn_qxi is True:    
            p_xi = self.base_dist_pxi
            xi = pyro.sample("xi", p_xi)

        if self.het_in is None:
            het_in = self.X
        else:
            het_in = self.het_in
        M = self.Xu.size(0)
        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
        Luu = torch.linalg.cholesky(Kuu)

        zero_loc = self.Xu.new_zeros(self.f_loc.shape)
        identity = eye_like(self.Xu, M)
        if self.learn_qxi is True: 
            pyro.sample(self._pyro_get_fullname("u"), dist.MultivariateNormal(zero_loc, scale_tril=identity*(torch.sqrt(xi)).expand_as(identity))
                                .to_event(zero_loc.dim() - 1)).to(self.device)
        else:
            pyro.sample(self._pyro_get_fullname("u"), dist.MultivariateNormal(zero_loc, scale_tril=identity)
                                .to_event(zero_loc.dim() - 1)).to(self.device)

        f_loc, f_var = conditional(self.X, self.Xu, self.kernel, self.f_loc, self.f_scale_tril,
                                   Luu, full_cov=False, whiten=True, jitter=self.jitter)

        if self.learn_qxi is True:
            f_var = (f_var*xi).squeeze() 
            f = dist.Normal(f_loc, f_var.sqrt())()
        else:
            f = dist.Normal(f_loc, f_var.sqrt())()
        with poutine.scale(scale= self.X.size(0)/self.num_data):
            self.likelihood(het_in, f, None, y = self.y)

    @pyro_method
    def guide(self):
        if self.learn_qxi is True:
                    pyro.module("spline_guide_xi", self.spline_guide_xi)
        if self.learn_qxi is True:    
            q_xi = TransformedDistribution( self.base_dist_qxi, [self.spline_guide_xi, self.output_trans_xi])
            xi = pyro.sample("xi", q_xi)
            pyro.sample(self._pyro_get_fullname("u"), dist.MultivariateNormal( self.f_loc , scale_tril=(self.f_scale_tril*(torch.sqrt(xi)).expand_as(self.f_scale_tril)))
                            .to_event(self.f_loc.dim() - 1))
        else:
            pyro.sample(self._pyro_get_fullname("u"), dist.MultivariateNormal( self.f_loc , scale_tril=self.f_scale_tril)
                        .to_event(self.f_loc.dim() - 1))


    def forward(self, X_new):
        loc, var = conditional(X_new, self.Xu, self.kernel, self.f_loc, self.f_scale_tril,
                               full_cov=False, whiten=True, jitter=self.jitter) 
        return loc,  var, None

    def likelihood_log_prob(self, X, value, loc, var):
        self.likelihood.elliptical_distribution.loc = loc
        self.likelihood.elliptical_distribution.var = var
        self.likelihood.elliptical_distribution.X = X

        log_prob = self.likelihood.elliptical_distribution.log_prob(value)
        return log_prob

    def estimate_predictive_log_prob(self,X_new, y_new, het_in_new = None):
        if het_in_new == None:
            het_in_new = X_new
        if X_new is None:
            X_new = self.X
        if y_new is None:
            y_new = self.y
        mu_new, var_new = conditional(X_new, self.Xu, self.kernel, self.f_loc, self.f_scale_tril,
                               full_cov=False, whiten=True, jitter=self.jitter)

        y_new = y_new.flatten()
        return self.likelihood_log_prob(het_in_new, y_new, mu_new, var_new).detach().sum()
    
    def define_splines(self, device = "cpu"):
        
        if self.learn_qxi is True:
            self.base_dist_pxi = torch.distributions.Independent(dist.InverseGamma(torch.ones(1).to(device)*500 / 2,torch.ones(1).to(device)*250/2),1)
            self.base_dist_qxi = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1).to(device), torch.ones(1).to(device)), 1)
            self.spline_guide_xi = dist.transforms.Spline(1, count_bins=5, order="linear").to(device)
            self.spline_guide_xi.load_state_dict(torch.load("model/spline_state_dict.pickle"))


    def derive_confidence(self):
        if self.learn_qxi is True:
            q_xi = TransformedDistribution( self.base_dist_qxi, [self.spline_guide_xi, self.output_trans_xi])
            xi_ = q_xi((10000,))
            samples = xi_
        else:
            samples = torch.ones((10000,)).detach()
        confidence_dict = {}
        for k in torch.linspace(0.01,4,10000):
            prob = (torch.erf(k/torch.sqrt(samples*2)).sum()/10000)
            if prob <= 0.05:
                confidence_dict[5] = k.item()
            if prob <= 0.10:
                confidence_dict[10] = k.item()
            if prob <= 0.15:
                confidence_dict[15] = k.item()
            if prob <= 0.20:
                confidence_dict[20] = k.item()
            if prob <= 0.25:
                confidence_dict[25] = k.item()
            if prob <= 0.30:
                confidence_dict[30] = k.item()
            if prob <= 0.35:
                confidence_dict[35] = k.item()
            if prob <= 0.40:
                confidence_dict[40] = k.item()
            if prob <= 0.45:
                confidence_dict[45] = k.item()
            if prob <= 0.50:
                confidence_dict[50] = k.item()
            if prob <= 0.55:
                confidence_dict[55] = k.item()
            if prob <=  0.60:
                confidence_dict[60] = k.item()
            if prob <=  0.65:
                confidence_dict[65] = k.item()
            if prob <=  0.70:
                confidence_dict[70] = k.item()
            if prob <=  0.75:
                confidence_dict[75] = k.item()
            if prob <=  0.80:
                confidence_dict[80] = k.item()
            if prob <= 0.85:
                confidence_dict[85] = k.item()
            if prob <= 0.90:
                confidence_dict[90] = k.item()
            if prob <= 0.95:
                confidence_dict[95] = k.item()
            if prob <= 0.99:
                confidence_dict[99] = k.item()
        return confidence_dict