"""Forecasting module for the Solar Energy Optimization System."""

import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor

class EnergyDemandForecaster:
    """Forecasts energy demand using time series features."""
    
    def __init__(self, model_type='prophet'):
        self.model_type = model_type
        self.model = None
        self.features = ['hour', 'dayofweek', 'is_weekend', 'month']
        
    def create_features(self, df):
        """Create time-based features for forecasting."""
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['month'] = df.index.month
        return df
    
    def fit(self, historical_data):
        """Train the forecasting model."""
        if self.model_type == 'prophet':
            df = historical_data[['timestamp', 'demand_kwh']].copy()
            df.columns = ['ds', 'y']
            self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            self.model.fit(df)
        elif self.model_type == 'xgboost':
            df = historical_data.set_index('timestamp')
            df = self.create_features(df)
            X = df[self.features]
            y = df['demand_kwh']
            self.model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
            self.model.fit(X, y)
    
    def predict(self, future_timestamps):
        """Generate demand forecast."""
        if self.model_type == 'prophet':
            future = pd.DataFrame({'ds': future_timestamps})
            forecast = self.model.predict(future)
            return pd.Series(forecast['yhat'].values, index=future_timestamps)
        elif self.model_type == 'xgboost':
            future_df = pd.DataFrame(index=future_timestamps)
            future_df = self.create_features(future_df)
            return pd.Series(self.model.predict(future_df[self.features]), index=future_timestamps)

class SolarProductionForecaster:
    """Forecasts solar energy production using weather data."""
    
    def __init__(self, model_type='prophet'):
        self.model_type = model_type
        self.model = None
        
    def create_features(self, df):
        """Create features for solar production forecast."""
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofyear'] = df.index.dayofyear
        df['solar_irradiance'] = np.maximum(0, np.sin(np.pi * (df['hour'] - 6) / 12) * 1000)
        if 'cloud_cover' in df.columns:
            df['solar_irradiance'] *= (1 - df['cloud_cover']/100)
        return df
    
    def fit(self, historical_data, weather_data):
        """Train the solar production forecasting model."""
        df = historical_data.set_index('timestamp')
        weather = weather_data.set_index('timestamp')
        df = df.join(weather, how='inner')
        
        if self.model_type == 'prophet':
            df = self.create_features(df).reset_index()
            df_prophet = df[['timestamp', 'solar_production_kwh']].copy()
            df_prophet.columns = ['ds', 'y']
            
            self.model = Prophet(yearly_seasonality=True, daily_seasonality=True)
            self.model.add_regressor('solar_irradiance')
            df_prophet['solar_irradiance'] = df['solar_irradiance']
            
            self.model.fit(df_prophet)
            
        elif self.model_type == 'xgboost':
            df = self.create_features(df)
            X = df[['hour', 'dayofyear', 'solar_irradiance', 'temperature', 'cloud_cover']]
            y = df['solar_production_kwh']
            self.model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
            self.model.fit(X, y)
    
    def predict(self, future_timestamps, weather_forecast):
        """Generate solar production forecast."""
        if self.model_type == 'prophet':
            future = pd.DataFrame({'ds': future_timestamps})
            future['solar_irradiance'] = self.create_features(
                pd.DataFrame(index=future_timestamps).join(weather_forecast.set_index('timestamp'))
            )['solar_irradiance'].values
            forecast = self.model.predict(future)
            return pd.Series(forecast['yhat'].values, index=future_timestamps)
        elif self.model_type == 'xgboost':
            future_df = self.create_features(
                pd.DataFrame(index=future_timestamps).join(weather_forecast.set_index('timestamp'))
            )
            X = future_df[['hour', 'dayofyear', 'solar_irradiance', 'temperature', 'cloud_cover']]
            return pd.Series(self.model.predict(X), index=future_timestamps)
