# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_dglm.ipynb (unless otherwise specified).

__all__ = ['dglm', 'bern_dglm', 'pois_dglm', 'dlm', 'bin_dglm']

# Internal Cell
from nbdev.showdoc import *


import numpy as np
import scipy as sc
import pandas as pd
from collections.abc import Iterable

from .latent_factor_fxns import update_lf_analytic, update_lf_sample, forecast_marginal_lf_analytic, \
    forecast_marginal_lf_sample, forecast_path_lf_copula, forecast_path_lf_sample, get_mean_and_var_lf, \
    get_mean_and_var_lf_dlm, update_lf_analytic_dlm
from .seasonal import seascomp, createFourierToSeasonalL
from .update import update, update_dlm, update_bindglm
from .forecast import forecast_marginal, forecast_path, forecast_path_copula,\
    forecast_marginal_bindglm, forecast_path_dlm, forecast_state_mean_and_var
from .conjugates import trigamma, bern_conjugate_params, bin_conjugate_params, pois_conjugate_params

# These are for the bernoulli and Poisson DGLMs
from scipy.special import digamma
from scipy.special import beta as beta_fxn
from scipy import stats

# Cell
class dglm:

    def __init__(self,
                 a0=None,
                 R0=None,
                 nregn=0,
                 ntrend=0,
                 nlf=0,
                 nhol=0,
                 seasPeriods=[],
                 seasHarmComponents=[],
                 deltrend=1, delregn=1,
                 dellf=1,
                 delhol=1, delseas=1,
                 rho=1,
                 interpolate=True,
                 adapt_discount=False,
                 adapt_factor=0.5,
                 discount_forecast=False):
        """
        A Dynamic Generalized Linear Model (DGLM). Generic Class. Children include Poisson, Bernoulli, and Binomial DGLMs, as well as the Normal DLM.

        :param a0: Prior mean vector
        :param R0: Prior covariance matrix
        :param nregn: Number of regression components
        :param ntrend: Number of trend components
        :param nhol: Number of holiday components
        :param seasPeriods: List of periods of seasonal components
        :param seasHarmComponents: List of harmonic components included for each period
        :param deltrend: Discount factor on trend components
        :param delregn: Discount factor on regression components
        :param delhol: Discount factor on holiday components (currently deprecated)
        :param delseas: Discount factor on seasonal components
        :param interpolate: Whether to use interpolation for conjugate parameters (provides a computational speedup)
        :param adapt_discount: What method of discount adaption. False = None, 'positive_regn' = only discount if regression information is available, 'info' = information based,\
        :param adapt_factor: If adapt_discount='info', then a higher value adapt_factor leads to a quicker adaptation (with less discounting) on overly uncertain parameters
        :param discount_forecast: Whether to use discounting when forecasting
        :return An object of class dglm
        """

        # Setting up trend F, G matrices

        i = 0
        self.itrend = list(range(i, ntrend))
        i += ntrend
        if ntrend == 0:
            Gtrend = np.empty([0, 0])
            Ftrend = np.zeros([ntrend]).reshape(-1, 1)
        # Local level
        elif ntrend == 1:
            Gtrend = np.identity(ntrend)
            Ftrend = np.array([1]).reshape(-1, 1)
        # Locally linear
        elif ntrend == 2:
            Gtrend = np.array([[1, 1], [0, 1]])
            Ftrend = np.array([1, 0]).reshape(-1, 1)

        # Setting up regression F, G matrices
        self.iregn = list(range(i, i + nregn))
        if nregn == 0:
            Fregn = np.empty([0]).reshape(-1, 1)
            Gregn = np.empty([0, 0])
        else:
            Gregn = np.identity(nregn)
            Fregn = np.ones([nregn]).reshape(-1, 1)
            i += nregn

        # Setting up holiday F, G matrices (additional regression indicators components)
        self.ihol = list(range(i, i + nhol))
        self.iregn.extend(self.ihol)  # Adding on to the self.iregn
        if nhol == 0:
            Fhol = np.empty([0]).reshape(-1, 1)
            Ghol = np.empty([0, 0])
        else:
            Ghol = np.identity(nhol)
            Fhol = np.ones([nhol]).reshape(-1, 1)
            i += nhol

        # Setting up seasonal F, G matrices
        self.iseas = []
        if len(seasPeriods) == 0:
            Fseas = np.empty([0]).reshape(-1, 1)
            Gseas = np.empty([0, 0])
            nseas = 0
        else:
            output = list(map(seascomp, seasPeriods, seasHarmComponents))
            Flist = [x[0] for x in output]
            Glist = [x[1] for x in output]
            self.L = list(map(createFourierToSeasonalL, seasPeriods, seasHarmComponents, Flist, Glist))
            nseas = 2 * sum(map(len, seasHarmComponents))
            Fseas = np.zeros([nseas]).reshape(-1, 1)
            Gseas = np.zeros([nseas, nseas])
            idx = 0
            for harmComponents in seasHarmComponents:
                self.iseas.append(list(range(i, i + 2 * len(harmComponents))))
                i += 2 * len(harmComponents)
            for Fs, Gs in output:
                idx2 = idx + Fs.shape[0]
                Fseas[idx:idx2, 0] = Fs.squeeze()
                Gseas[idx:idx2, idx:idx2] = Gs
                idx = idx2

        # Setting up the latent factor F, G matrices
        if nlf == 0:
            self.latent_factor = False
            Glf = np.empty([0, 0])
            Flf = np.zeros([0]).reshape(-1, 1)
        else:
            self.latent_factor = True
            Glf = np.identity(nlf)
            Flf = np.ones([nlf]).reshape(-1, 1)
            self.ilf = list(range(i, i + nlf))
            i += nlf

        # Combine the F and G components together
        F = np.vstack([Ftrend, Fregn, Fhol, Fseas, Flf])
        G = sc.linalg.block_diag(Gtrend, Gregn, Ghol, Gseas, Glf)

        # store the discount info
        self.deltrend = deltrend
        self.delregn = delregn
        self.delhol = delhol
        self.delseas = delseas
        self.dellf = dellf

        # Random effect to inflate variance (if rho < 1)
        self.rho = rho

        self.ntrend = ntrend
        self.nregn = nregn + nhol  # Adding on nhol
        self.nregn_exhol = nregn
        self.nhol = nhol
        self.nseas = nseas
        self.nlf = nlf

        self.adapt_discount = adapt_discount
        self.k = adapt_factor

        # Set up discount matrix
        self.discount_forecast = discount_forecast
        Discount = self.build_discount_matrix()
        self.Discount = Discount

        self.param1 = 2  # Random initial guess
        self.param2 = 2  # Random initial guess


        self.seasPeriods = seasPeriods
        self.seasHarmComponents = seasHarmComponents
        self.F = F
        self.G = G
        self.a = a0.reshape(-1, 1)
        self.R = R0
        self.t = 0
        self.interpolate = interpolate
        self.W = self.get_W()

    def build_discount_matrix(self, X=None, phi_mu=None):
        # build up discount factors while possibly taking special care to not discount when the "regn"
        # type factors are zero

        # do this all with matrix slicing which is much faster than the block diag
        p = np.sum([self.ntrend, self.nregn_exhol, self.nhol, self.nseas, self.nlf])
        # start with no discounting
        component_discounts = np.ones([p, p])
        i = 0 # this will be the offset of the current block
        for discount_pair, n in zip([('std', self.deltrend), ('regn', self.delregn), ('hol', self.delhol),
                                     ('std', self.delseas), ('lf', self.dellf)],
                                    [self.ntrend, self.nregn_exhol, self.nhol, self.nseas, self.nlf]):
            discount_type, discount = discount_pair
            if n > 0:
                if isinstance(discount, Iterable):
                    if len(discount) < n:
                        raise ValueError('Error: Length of discount factors must be 1 or match component length')
                    for j, disc in enumerate(discount[:n]):
                        # fill the diags one at a time
                        component_discounts[i+j, i+j] = disc
                else:
                    # fill the block with the constant
                    component_discounts[i:(i+n), i:(i+n)] = discount

                # overwrite with ones if doing the positive logic
                if X is not None and self.adapt_discount == 'positive_regn' and (discount_type == 'regn' or discount_type == 'hol'):
                    if not isinstance(X, Iterable):
                        X = [X]

                    if discount_type == 'regn':
                        # offset of the regression params
                        regn_i = 0
                    elif discount_type == 'hol':
                        regn_i = self.nregn_exhol
                    # look through the regression params and set that slice on the
                    # discount to 1 if 0
                    for j in range(n):
                        if X[regn_i] == 0:
                            # set all discounts to one (i offsets the block and j offsets the regn param)
                            component_discounts[i + j, :] = 1.
                            component_discounts[:, i + j] = 1.
                        regn_i += 1

                if phi_mu is not None and self.adapt_discount == 'positive_regn' and discount_type == 'lf':
                    # offset of the latent factor params
                    lf_i = 0
                    # look through the latent factor params and set that slice on the
                    # discount to 1 if 0
                    for j in range(n):
                        if phi_mu[lf_i] == 0:
                            # set all discounts to one (i offsets the block and j offsets the regn param)
                            component_discounts[i + j, :] = 1.
                            component_discounts[:, i + j] = 1.
                        lf_i += 1

                # move on to the next block
                i += n

        return component_discounts

    def update(self, y=None, X=None, phi_mu = None, phi_sigma = None, analytic=True, phi_samps=None, **kwargs):
        """
        Update the DGLM state vector mean and covariance after observing 'y', with covariates 'X'.
        """

        if self.latent_factor:
            if analytic:
                update_lf_analytic(self, y, X, phi_mu, phi_sigma)
            else:
                parallel = kwargs.get('parallel')
                if parallel is None: parallel = False
                update_lf_sample(self, y, X, phi_samps, parallel)
        else:
            update(self, y, X)

    def forecast_marginal(self, k, X=None, nsamps=1, mean_only=False,
                          phi_mu = None, phi_sigma=None, analytic=True, phi_samps=None,
                          state_mean_var=False, y=None):
        """
        Simulate from the forecast distribution at time *t+k*.
        """
        if self.latent_factor:
            if analytic:
                return forecast_marginal_lf_analytic(self, k, X, phi_mu, phi_sigma, nsamps, mean_only, state_mean_var)
            else:
                forecast_marginal_lf_sample(self, k, X, phi_samps, mean_only)
        else:
            return forecast_marginal(self, k, X, nsamps, mean_only, state_mean_var, y)

    def forecast_path(self, k, X=None, nsamps=1, copula=True,
                      phi_mu=None, phi_sigma=None, phi_psi=None, analytic=True, phi_samps=None,
                      **kwargs):
        """
        Simulate from the path (joint) forecast distribution from *1* to *k* steps ahead.
        """
        if self.latent_factor:
            if analytic:
                return forecast_path_lf_copula(self, k, X, phi_mu, phi_sigma, phi_psi, nsamps, **kwargs)
            else:
                return forecast_path_lf_sample(self, k, X, phi_samps)
        else:
            if copula:
                return forecast_path_copula(self, k, X, nsamps, **kwargs)
            else:
                return forecast_path(self, k, X, nsamps)

    def forecast_path_copula(self, k, X=None, nsamps=1, **kwargs):
        return forecast_path_copula(self, k, X, nsamps, **kwargs)

    # Define specific update and forecast functions, which advanced users can access manually
    def update_lf_sample(self, y=None, X=None, phi_samps=None, parallel=False):
        update_lf_sample(self, y, X, phi_samps, parallel)

    def update_lf_analytic(self, y=None, X=None, phi_mu=None, phi_sigma=None):
        update_lf_analytic(self, y, X, phi_mu, phi_sigma)

    def forecast_marginal_lf_analytic(self, k, X=None, phi_mu=None, phi_sigma=None, nsamps=1, mean_only=False, state_mean_var=False):
        return forecast_marginal_lf_analytic(self, k, X, phi_mu, phi_sigma, nsamps, mean_only, state_mean_var)

    def forecast_marginal_lf_sample(self, k, X=None, phi_samps=None, mean_only=False):
        return forecast_marginal_lf_sample(self, k, X, phi_samps, mean_only)

    def forecast_path_lf_copula(self, k, X=None, phi_mu=None, phi_sigma=None, phi_psi=None, nsamps=1, **kwargs):
        return forecast_path_lf_copula(self, k, X, phi_mu, phi_sigma, phi_psi, nsamps, **kwargs)

    def forecast_path_lf_sample(self, k, X=None, phi_samps=None, nsamps=1, **kwargs):
        return forecast_path_lf_sample(self, k, X, phi_samps)

    def forecast_state_mean_and_var(self, k, X = None):
        return forecast_state_mean_and_var(self, k, X)

    def get_mean_and_var(self, F, a, R):
        mean, var = F.T @ a, F.T @ R @ F / self.rho
        return np.ravel(mean)[0], np.ravel(var)[0]

    def get_mean_and_var_lf(self, F, a, R, phi_mu, phi_sigma, ilf):
        return get_mean_and_var_lf(self, F, a, R, phi_mu, phi_sigma, ilf)

    def get_W(self, X=None):
        if self.adapt_discount == 'info':
            info = np.abs(self.a.flatten() / np.sqrt(self.R.diagonal()))
            diag = self.Discount.diagonal()
            diag = np.round(diag + (1 - diag) * np.exp(-self.k * info), 5)
            Discount = np.ones(self.Discount.shape)
            np.fill_diagonal(Discount, diag)
        elif self.adapt_discount == 'positive_regn' and X is not None:
            Discount = self.build_discount_matrix(X)
        else:
            Discount = self.Discount
        return self.R / Discount - self.R

    def get_coef(self, component=None):
        """Return the coefficient (state vector) means and standard deviations.

        If component=None, then the full state vector is returned.

        Otherwise, specify a single component from 'trend', 'regn', 'seas', 'hol', and 'lf'.
        """


        trend_names = ['Intercept', 'Local Slope'][:self.ntrend]
        regn_names = ['Regn ' + str(i) for i in range(1, self.nregn_exhol+1)]
        seas_names = ['Seas ' + str(i) for i in range(1, self.nseas+1)]
        hol_names = ['Hol ' + str(i) for i in range(1, self.nhol+1)]
        lf_names = ['LF ' + str(i) for i in range(1, self.nlf+1)]

        if component is None:

            names = [*trend_names, *regn_names, *seas_names, *hol_names, *lf_names]

            return pd.DataFrame({'Mean':self.a.reshape(-1),
                                 'Standard Deviation': np.sqrt(self.R.diagonal())},
                                 index=names).round(2)
        elif component == 'trend':
            names = trend_names

            return pd.DataFrame({'Mean':self.a.reshape(-1)[self.itrend],
                                 'Standard Deviation': np.sqrt(self.R.diagonal())[self.itrend]},
                                 index=names).round(2)

        elif component == 'regn':
            names = regn_names

            return pd.DataFrame({'Mean':self.a.reshape(-1)[self.iregn],
                                 'Standard Deviation': np.sqrt(self.R.diagonal())[self.iregn]},
                                 index=names).round(2)

        elif component == 'seas':
            names = seas_names

            seas_idx = []
            for idx in self.iseas:
                seas_idx.extend(idx)

            return pd.DataFrame({'Mean':self.a.reshape(-1)[seas_idx],
                                 'Standard Deviation': np.sqrt(self.R.diagonal())[seas_idx]},
                                 index=names).round(2)

        elif component == 'hol':
            names = hol_names

            return pd.DataFrame({'Mean':self.a.reshape(-1)[self.ihol],
                                 'Standard Deviation': np.sqrt(self.R.diagonal())[self.ihol]},
                                 index=names).round(2)

        elif component == 'lf':
            names = lf_names

            return pd.DataFrame({'Mean':self.a.reshape(-1)[self.ilf],
                                 'Standard Deviation': np.sqrt(self.R.diagonal())[self.ilf]},
                                 index=names).round(2)

    # Added functions
    def get_k_median_forecast(self, forecast):
        k_steps = np.median(forecast, axis=0)
        return k_steps
    def get_k_mean_forecast(self, forecast):
        k_steps = np.mean(forecast, axis=0)
        return k_steps

# Cell
class bern_dglm(dglm):

    def get_conjugate_params(self, ft, qt, alpha_init, beta_init):
        # Choose conjugate prior, beta, and match mean & variance
        return bern_conjugate_params(ft, qt, alpha_init, beta_init, interp=self.interpolate)

    def update_conjugate_params(self, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + y
        beta = beta + 1 - y

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - digamma(beta)
        qt_star = trigamma(alpha) + trigamma(beta)

        # constrain this thing from going to crazy places
        ft_star = max(-8, min(ft_star, 8))
        qt_star = max(0.001 ** 2, min(qt_star, 4 ** 2))

        return alpha, beta, ft_star, qt_star

    def simulate(self, alpha, beta, nsamps):
        p = np.random.beta(alpha, beta, [nsamps])
        return np.random.binomial(1, p, size=[nsamps])

    def simulate_from_sampling_model(self, p, nsamps):
        return np.random.binomial(1, p, [nsamps])

    def simulate_from_prior(self, alpha, beta, nsamps):
        return stats.beta.rvs(a=alpha, b=beta, size=nsamps)

    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.beta.ppf(cdf, alpha, beta)

    def sampling_density(self, y, p):
        return stats.binom.pmf(n=1, p=p, k=y)

    def marginal_cdf(self, y, alpha, beta):
        if y == 1:
            return 1
        elif y == 0:
            return beta_fxn(y + alpha, 1 - y + beta) / beta_fxn(alpha, beta)

    def loglik(self, y, alpha, beta):
        return stats.bernoulli.logpmf(y, alpha / (alpha + beta))

    def get_mean(self, alpha, beta):
        return np.ravel(alpha / (alpha + beta))[0]

    def get_prior_var(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

# Cell
class pois_dglm(dglm):

    def get_conjugate_params(self, ft, qt, alpha_init, beta_init):
        # Choose conjugate prior, gamma, and match mean & variance
        return pois_conjugate_params(ft, qt, alpha_init, beta_init, interp=self.interpolate)

    def update_conjugate_params(self, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + float(y)
        beta = beta + 1

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - np.log(beta)
        qt_star = trigamma(alpha)

        # constrain this thing from going to crazy places?
        qt_star = max(0.001 ** 2, min(qt_star, 4 ** 2))

        return alpha, beta, ft_star, qt_star

    def simulate(self, alpha, beta, nsamps):
        return np.random.negative_binomial(alpha, beta / (1 + beta), [nsamps])

    def simulate_from_sampling_model(self, rate, nsamps):
        return np.random.poisson(rate, [nsamps])

    def simulate_from_prior(self, alpha, beta, nsamps):
        return stats.gamma.rvs(a=alpha, scale=1/beta, size=nsamps)

    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.gamma.ppf(cdf, a=alpha, scale=1 / beta)

    def sampling_density(self, y, mu):
        return stats.poisson.pmf(mu=mu, k=y)

    def marginal_cdf(self, y, alpha, beta):
        return stats.nbinom.cdf(y, alpha, beta / (1 + beta))

    def marginal_inverse_cdf(self, cdf, alpha, beta):
        return stats.nbinom.ppf(cdf, alpha, beta / (1 + beta))

    def loglik(self, y, alpha, beta):
        return stats.nbinom.logpmf(y, alpha, beta / (1 + beta))

    def get_mean(self, alpha, beta):
        return np.ravel(alpha/beta)[0]

    def get_prior_var(self, alpha, beta):
        return alpha / beta ** 2

# Cell
class dlm(dglm):

    def __init__(self, *args, n0=1, s0=1, delVar=1, **kwargs):
        self.delVar = delVar  # Discount factor for the variance - using a beta-gamma random walk
        self.n = n0  # Prior sample size for the variance
        self.s = s0  # Prior mean for the variance
        super().__init__(*args, **kwargs)

    def get_mean_and_var(self, F, a, R):
        return F.T @ a, F.T @ R @ F + self.s

    def get_mean_and_var_lf(self, F, a, R, phi_mu, phi_sigma, ilf):
        ct = self.n / (self.n - 2)
        ft, qt = get_mean_and_var_lf_dlm(F, a, R, phi_mu, phi_sigma, ilf, ct)
        qt = qt + self.s
        return ft, qt

    def get_mean(self, ft, qt):
        return np.ravel(ft)[0]

    def get_conjugate_params(self, ft, qt, mean, var):
        return ft, qt

    def simulate(self, mean, var, nsamps):
        return mean + np.sqrt(var) * np.random.standard_t(self.n, size=[nsamps])

    def simulate_from_sampling_model(self, mean, var, nsamps):
        return np.random.normal(mean, np.sqrt(var), nsamps)

    def update(self, y=None, X=None, phi_mu=None, phi_sigma=None, analytic=True, phi_samps=None, **kwargs):
        if self.latent_factor:
            if analytic:
                update_lf_analytic_dlm(self, y, X, phi_mu, phi_sigma)
            else:
                print('Sampled-based updating for the Latent Factor DLM is not yet implemented - please use analytic inference')
        else:
            update_dlm(self, y, X)

    def forecast_path(self, k, X=None, nsamps=1, **kwargs):
        if self.latent_factor:
            print('Path forecasting for latent factor DLMs is not yet implemented')
        else:
            return forecast_path_dlm(self, k, X, nsamps)

    def update_lf_analytic(self, y=None, X=None, phi_mu=None, phi_sigma=None):
        update_lf_analytic_dlm(self, y, X, phi_mu, phi_sigma)

    def loglik(self, y, mean, var):
        return stats.t.logpdf(y, df=self.n, loc=mean, scale=np.sqrt(var))

# Cell
class bin_dglm(dglm):

    def get_conjugate_params(self, ft, qt, alpha_init, beta_init):
        # Choose conjugate prior, beta, and match mean & variance
        return bin_conjugate_params(ft, qt, alpha_init, beta_init, interp=self.interpolate)

    def update_conjugate_params(self, n, y, alpha, beta):
        # Update alpha and beta to the conjugate posterior coefficients
        alpha = alpha + y
        beta = beta + n - y

        # Get updated ft* and qt*
        ft_star = digamma(alpha) - digamma(beta)
        qt_star = trigamma(alpha) + trigamma(beta)

        # constrain this thing from going to crazy places?
        ft_star = max(-8, min(ft_star, 8))
        qt_star = max(0.001 ** 2, min(qt_star, 4 ** 2))

        return alpha, beta, ft_star, qt_star

    def simulate(self, n, alpha, beta, nsamps):
        p = np.random.beta(alpha, beta, [nsamps])
        return np.random.binomial(n.astype(int), p, size=[nsamps])

    def simulate_from_sampling_model(self, n, p, nsamps):
        return np.random.binomial(n, p, [nsamps])

    def prior_inverse_cdf(self, cdf, alpha, beta):
        return stats.beta.ppf(cdf, alpha, beta)

    def marginal_cdf(self, y, n, alpha, beta):
        cdf = 0.0
        for i in range(y + 1):
            cdf += sc.misc.comb(n, y) * beta_fxn(y + alpha, n - y + beta) / beta_fxn(alpha, beta)
        return cdf

    def loglik(self, data, alpha, beta):
        n, y = data
        return stats.binom.logpmf(y, n, alpha / (alpha + beta))

    def get_mean(self, n, alpha, beta):
        return np.ravel(n * (alpha / (alpha + beta)))[0]

    def get_prior_var(self, alpha, beta):
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    def update(self, n=None, y=None, X=None):
        update_bindglm(self, n, y, X)

    def forecast_marginal(self, n, k, X=None, nsamps=1, mean_only=False):
        return forecast_marginal_bindglm(self, n, k, X, nsamps, mean_only)