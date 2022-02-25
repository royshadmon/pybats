# Import the necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import pmdarima as pmd
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
import math

from pybats.loss_functions import MAPE
from pybats.analysis import analysis
from pybats.point_forecast import median
from pybats.plot import plot_data_forecast

from pybats.forecast import forecast_path_dlm

R_d = 287.058 # gas constant for dry air
R_w = 461.495 # gas constant for wet air
A = 20 # The area swept by the rotor
a = 1/2 # assume axial induction factor is constant but this thesis talks about it https://www.iri.upc.edu/files/scidoc/2183-Data-Driven-Decentralized-Algorithm-forWind-Farm-Control-with-Population-Games-Assistance.pdf
k=15 # Forecast horizon (15 min ahead)


def csv_to_df(filepath):
    df = pd.read_csv(filepath ,parse_dates=True)
    df['Date'] = df[['DATE (MM/DD/YYYY)', 'MST']].agg(' '.join, axis=1)
    df = df.drop(['DATE (MM/DD/YYYY)', 'MST'], axis=1)
    df = df.rename(columns={'Avg Wind Speed @ 2m [m/s]': 'WindSpeed', 'Avg Wind Direction @ 2m [deg]': 'WindDir', 'Temperature @ 2m [deg C]': 'Temperature', 'Relative Humidity [%]': 'Humidity', 'Station Pressure [mBar]': 'Pressure'})

    df = df.reindex(columns=['Date', 'WindSpeed', 'WindDir', 'Temperature', 'Humidity', 'Pressure'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

def calc_density(temperature, humidity, pressure):
    water_vapor_pressure = humidity * pressure / 100 # divide 100 because humidity is in %
    pressure_dry_air = pressure - water_vapor_pressure
    rho = pressure_dry_air / (R_d * temperature) + water_vapor_pressure / (R_w * temperature)
    return rho

def calc_wind_diff(dataframe, adj):

    prev = dataframe.WindDir[0]
    windDiff = [0+adj]
    for row in dataframe.WindDir[1:].values:
        if np.abs(row - prev) <= 180:
            windDiff.append(prev-row+adj)
        else:
            change = 360 - np.abs(prev-row)
            if prev - row < 0:
                change *= -1
            windDiff.append(change + adj)
        prev = row

    windDiff = np.array(windDiff)
    dataframe['WindDiff'] = windDiff

def calc_power_coeffcient(a):
    C_p = 4 * a * np.power(1-a, 2)
    return C_p

C_p = calc_power_coeffcient(a)

# inputs: air density, Area swept by blade, power coefficient, observed wind speed
def calc_power_generated(air_density, wind_speed, A=20, C_p=C_p):
    power = (1/2) * np.abs(air_density) * A * C_p * np.power(wind_speed, 3)
    return power

def Forecast(dataframe, prediction_field: str, correlated_fields, forecast_start, forecast_end, model_prior=None):

    prior_length = 21 # Number of minutes of data used to set prior
    rho = 0.6           # Random effect discount factor to increase variance of forecast

    mod, samples, model_coef = analysis(dataframe[prediction_field].values,
                                        #                                     df[['WindSpeed', 'WindDiff']].values,
                                        #                                     df[['Power', 'WindSpeed', 'WindDiff']].values,
                                        dataframe[correlated_fields].values,
                                        k,
                                        forecast_start,
                                        forecast_end,
                                        model_prior=model_prior,
                                        nsamps=5000,
                                        family='normal',
                                        # seasPeriods=[1],
                                        # seasHarmComponents=[[1,2]],
                                        prior_length=prior_length,
                                        dates=dataframe.index,
                                        holidays=USFederalHolidayCalendar.rules,
                                        rho=rho,
                                        # forecast_path=True,
                                        ret = ['model', 'forecast', 'model_coef'])
    return mod, samples, model_coef

# Display WindDir forecast

def displayDirPrediction(dataframe, samples, label, forecast_start, forecast_end, h=15):

    h = k


    start = forecast_start + pd.DateOffset(minutes=h-1)
    end = forecast_end + pd.DateOffset(minutes=h-1)

    data_1step = dataframe.loc[start:end]
    # print(data_1step)
    samples_1step = samples[:,:,h-1]
    forecast = []
    for i in median(samples_1step):
        if i > 360:
            i -= 360
        elif i < 0:
            i += 360
        forecast.append(i)
    forecast = np.array(forecast
                        )
    fig, ax = plt.subplots(figsize=(10,5))
    plot_data_forecast(fig, ax,
                       data_1step[label],
                       forecast,
                       samples_1step,
                       data_1step.index,
                       credible_interval=95)

    plt.ylim(0, 400)
    ax.legend([f'Observed {label}', 'Forecast', 'Credible Interval'])
    plt.title(f'{label} {k}-min Ahead Forecast')
    plt.ylabel(f'Minute {label}')
    return forecast, start

def displayPrediction(dataframe, samples, label, forecast_start, forecast_end, h=15):
    h = k

    start = forecast_start + pd.DateOffset(minutes=h-1)
    end = forecast_end + pd.DateOffset(minutes=h-1)

    data_1step = dataframe.loc[start:end]
    # print(data_1step)
    samples_1step = samples[:,:,h-1]
    forecast = median(samples_1step)

    fig, ax = plt.subplots(figsize=(10,5))
    plot_data_forecast(fig, ax,
                       data_1step[label],
                       median(samples_1step),
                       samples_1step,
                       data_1step.index,
                       credible_interval=95)

    plt.ylim(np.min(samples_1step),np.max(samples_1step))
    ax.legend([f'Observed {label}', 'Forecast', 'Credible Interval'])
    plt.title(f'{label} {k}-min Ahead Forecast')
    plt.ylabel(f'Minute {label}')
    return forecast, start

def setupForecastDS(dataframe, forecast_start, h):
    start = forecast_start + pd.DateOffset(minutes=h-1)
    df_predictions = pd.DataFrame(columns=['Date'])
    df_predictions['Date'] = dataframe.index[dataframe.index.get_loc(start):]
    df_predictions = df_predictions.set_index('Date')
    return df_predictions

def add_prediction_to_df(dataframe, colName, forecast):
    dataframe[colName] = pd.to_numeric(forecast)





def main():
    df = csv_to_df('wind-data-small-features.csv')
    df = df.assign(Density = lambda x: (calc_density(x['Temperature'], x['Humidity'], x['Pressure'])))
    adj = 200
    calc_wind_diff(df, adj)
    df = df.assign(Power = lambda x: calc_power_generated(x['Density'], x['WindSpeed']))
    #### Modifiable ####
    k=15
    forecast_start = pd.to_datetime(df.index[-100])
    forecast_end = pd.to_datetime(df.index[-k])
    ####################
    # Init df_predictions
    df_predictions = setupForecastDS(df, forecast_start, k)
    models = {} # stores models



    # WindDir Forecast

    mod_wind, samples, model_coef = Forecast(df, 'WindDir', ['WindDiff'], forecast_start, forecast_end)
    forecast, start = displayDirPrediction(df, samples, "WindDir", forecast_start, forecast_end)
    models['WindDir'] = {"model": mod_wind, "model_coef": model_coef, "forecast": forecast}
    add_prediction_to_df(df_predictions, "WindDir", forecast)

    # Predict WindSpeed

    mod, samples, model_coef = Forecast(df, 'WindSpeed', ['Power', 'WindDir'], forecast_start, forecast_end)
    forecast, start = displayPrediction(df, samples, "WindSpeed", forecast_start, forecast_end)

    models['WindSpeed'] = {"model": mod, "model_coef": model_coef, "forecast": forecast}
    add_prediction_to_df(df_predictions, "WindSpeed", forecast)


    nsamps = 10
    X = None
    k=5

    # X = np.array([[0], [0.5]])
    X = np.array(mod_wind.nregn*[0])
    # X = models['WindSpeed']['forecast'][-12:,]
    # models['WindSpeed']['model'].forecast_path(k, X, nsamps)
    # forecast_samples = mod.forecast_marginal(k=50, X=X, nsamps=3)
    # forecast_samples = mod.forecast_path(k = 2, X=X, nsamps=3)
    # print(np.percentile(forecast_samples, [2.5, 97.5]))
    # print(f'RETURN IS {forecast_samples}')

    forecast_samples = forecast_path_dlm(mod_wind, k, X,nsamps=5)
    # forecast_path_dlm returns a numpy ndarray containing [k x nsamps]
    # Each column contains a 1-step prediction and each row is a random sample from the distribution
    # For example, column 2 is the k+2 prediction.

    # The best way to minimize the loss is to take the median sample from each k-step prediction.
    # print(type(forecast_samples))
    print(f'forecast_samples are \n{forecast_samples}')
    print(f'MEDIAN ISn{mod.get_k_median_forecast(forecast_samples)}')
    print(f'MEAN IS \n{mod.get_k_mean_forecast(forecast_samples)}')






if __name__ == "__main__":
    main()



