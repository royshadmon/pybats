__all__ = ['analysis', 'analysis_dcmm', 'analysis_dbcm', 'analysis_dlmm']

# Internal Cell
#exporti
import numpy as np
import pandas as pd

from .define_models import define_dglm, define_dcmm, define_dbcm, define_dlmm
from .shared import define_holiday_regressors
from collections.abc import Iterable


# Cell
def analysis(Y, X=None, k=1, forecast_start=0, forecast_end=0,
             nsamps=500, family = 'normal', n = None,
             model_prior = None, prior_length=20, ntrend=1,
             dates = None, holidays = [],
             seasPeriods = [], seasHarmComponents = [],
             latent_factor = None, new_latent_factors = None,
             ret=['model', 'forecast'],
             mean_only = False, forecast_path = False,
             **kwargs):
    """
    This is a helpful function to run a standard analysis. The function will:
    1. Automatically initialize a DGLM
    2. Run sequential updating
    3. Forecast at each specified time step
    """

    # Add the holiday indicator variables to the regression matrix
    nhol = len(holidays)
    X = define_holiday_regressors(X, dates, holidays)

    # Check if it's a latent factor DGLM
    if latent_factor is not None:
        is_lf = True
        nlf = latent_factor.p
    else:
        is_lf = False
        nlf = 0

    if model_prior is None:
        mod = define_dglm(Y, X, family=family, n=n, prior_length=prior_length, ntrend=ntrend, nhol=nhol, nlf=nlf,
                          seasPeriods=seasPeriods, seasHarmComponents=seasHarmComponents,
                          **kwargs)
    else:
        mod = model_prior


    # Convert dates into row numbers
    if dates is not None:
        dates = pd.Series(dates)
        if type(forecast_start) == type(dates.iloc[0]):
            forecast_start = np.where(dates == forecast_start)[0][0]
        if type(forecast_end) == type(dates.iloc[0]):
            forecast_end = np.where(dates == forecast_end)[0][0]

    # Define the run length
    T = len(Y) + 1

    if ret.__contains__('model_coef'):
        m = np.zeros([T-1, mod.a.shape[0]])
        C = np.zeros([T-1, mod.a.shape[0], mod.a.shape[0]])
        if family == 'normal':
            n = np.zeros(T)
            s = np.zeros(T)

    if new_latent_factors is not None:
        if not ret.__contains__('new_latent_factors'):
            ret.append('new_latent_factors')

        if not isinstance(new_latent_factors, Iterable):
            new_latent_factors = [new_latent_factors]

        tmp = []
        for lf in new_latent_factors:
            tmp.append(lf.copy())
        new_latent_factors = tmp

    # Create dummy variable if there are no regression covariates
    if X is None:
        X = np.array([None]*(T+k)).reshape(-1,1)
    else:
        if len(X.shape) == 1:
            X = X.reshape(-1,1)

    # Initialize updating + forecasting
    horizons = np.arange(1, k + 1)

    if mean_only:
        forecast = np.zeros([1, forecast_end - forecast_start + 1, k])
    else:
        forecast = np.zeros([nsamps, forecast_end - forecast_start + 1, k])

    for t in range(prior_length, T):

        if forecast_start <= t <= forecast_end:
            if t == forecast_start:
                print('beginning forecasting')

            if ret.__contains__('forecast'):
                if is_lf:
                    if forecast_path:
                        pm, ps, pp = latent_factor.get_lf_forecast(dates.iloc[t])
                        forecast[:, t - forecast_start, :] = mod.forecast_path_lf_copula(k=k, X=X[t + horizons - 1, :],
                                                                                         nsamps=nsamps,
                                                                                         phi_mu=pm, phi_sigma=ps, phi_psi=pp)
                    else:
                        pm, ps = latent_factor.get_lf_forecast(dates.iloc[t])
                        pp = None  # Not including path dependency in latent factor

                        forecast[:, t - forecast_start, :] = np.array(list(map(
                            lambda k, x, pm, ps:
                            mod.forecast_marginal_lf_analytic(k=k, X=x, phi_mu=pm, phi_sigma=ps, nsamps=nsamps, mean_only=mean_only),
                            horizons, X[t + horizons - 1, :], pm, ps))).squeeze().T.reshape(-1, k)#.reshape(-1, 1)
                else:
                    if forecast_path:
                        forecast[:, t - forecast_start, :] = mod.forecast_path(k=k, X = X[t + horizons - 1, :], nsamps=nsamps)
                    else:
                        if family == "binomial":
                            forecast[:, t - forecast_start, :] = np.array(list(map(
                                lambda k, n, x:
                                mod.forecast_marginal(k=k, n=n, X=x, nsamps=nsamps, mean_only=mean_only),
                                horizons, n[t + horizons - 1], X[t + horizons - 1, :]))).squeeze().T.reshape(-1, k)  # .reshape(-1, 1)
                        else:
                            # Get the forecast samples for all the items over the 1:k step ahead marginal forecast distributions
                            forecast[:, t - forecast_start, :] = np.array(list(map(
                                lambda k, x:
                                mod.forecast_marginal(k=k, X=x, nsamps=nsamps, mean_only=mean_only),
                                horizons, X[t + horizons - 1, :]))).squeeze().T.reshape(-1, k)#.reshape(-1, 1)

            if ret.__contains__('new_latent_factors'):
                for lf in new_latent_factors:
                    lf.generate_lf_forecast(date=dates[t], mod=mod, X=X[t + horizons - 1],
                                            k=k, nsamps=nsamps, horizons=horizons)

        # Now observe the true y value, and update:
        if t < len(Y):
            if is_lf:
                pm, ps = latent_factor.get_lf(dates.iloc[t])
                mod.update_lf_analytic(y=Y[t], X=X[t],
                                       phi_mu=pm, phi_sigma=ps)
            else:
                if family == "binomial":
                    mod.update(y=Y[t], X=X[t], n=n[t])
                else:
                    mod.update(y=Y[t], X=X[t])

            if ret.__contains__('model_coef'):
                m[t,:] = mod.m.reshape(-1)
                C[t,:,:] = mod.C
                if family == 'normal':
                    n[t] = mod.n / mod.delVar
                    s[t] = mod.s

            if ret.__contains__('new_latent_factors'):
                for lf in new_latent_factors:
                    lf.generate_lf(date=dates[t], mod=mod, Y=Y[t], X=X[t], k=k, nsamps=nsamps)

    out = []
    for obj in ret:
        if obj == 'forecast': out.append(forecast)
        if obj == 'model': out.append(mod)
        if obj == 'model_coef':
            mod_coef = {'m':m, 'C':C}
            if family == 'normal':
                mod_coef.update({'n':n, 's':s})

            out.append(mod_coef)
        if obj == 'new_latent_factors':
            #for lf in new_latent_factors:
            #    lf.append_lf()
            #    lf.append_lf_forecast()
            if len(new_latent_factors) == 1:
                out.append(new_latent_factors[0])
            else:
                out.append(new_latent_factors)

    if len(out) == 1:
        return out[0]
    else:
        return out