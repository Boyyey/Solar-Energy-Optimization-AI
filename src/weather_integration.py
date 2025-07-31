"""
Weather Integration Module for Solar Energy Optimization System

This module provides functionality to fetch real-time weather data
from the OpenWeatherMap API.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import Optional, Dict, List

# Constants
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5"
CACHE_DURATION = 3600  # 1 hour in seconds

@dataclass
class WeatherData:
    """Container for weather data."""
    timestamp: datetime
    temperature: float  # Celsius
    cloud_cover: float  # %
    humidity: float     # %
    wind_speed: float   # m/s
    solar_irradiance: Optional[float] = None  # W/m²
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'temperature': self.temperature,
            'cloud_cover': self.cloud_cover,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'solar_irradiance': self.solar_irradiance
        }

class WeatherAPI:
    """Handles weather data fetching and caching."""
    
    def __init__(self, api_key: str, cache_dir: str = 'data/weather_cache'):
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, lat: float, lon: float) -> str:
        """Get cache file path for location."""
        return os.path.join(self.cache_dir, f"{lat:.4f}_{lon:.4f}.json")
    
    def _load_cached_weather(self, lat: float, lon: float) -> Optional[dict]:
        """Load cached weather if recent."""
        cache_path = self._get_cache_path(lat, lon)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            cache_time = datetime.fromisoformat(data['timestamp'])
            if (datetime.now() - cache_time).total_seconds() < CACHE_DURATION:
                return data
                
        except (json.JSONDecodeError, KeyError):
            pass
            
        return None
    
    def _save_weather_to_cache(self, lat: float, lon: float, data: dict) -> None:
        """Save weather data to cache."""
        cache_path = self._get_cache_path(lat, lon)
        data['timestamp'] = datetime.now().isoformat()
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _extract_weather_data(self, data: dict) -> WeatherData:
        """Extract weather data from API response."""
        current = data.get('current', {})
        clouds = current.get('clouds', {}).get('all', 50)
        
        # Simple solar irradiance model
        solar = 0
        if 'dt' in current and 'sunrise' in current and 'sunset' in current:
            now = current['dt']
            if current['sunrise'] < now < current['sunset']:
                # Basic model: max at solar noon, reduced by clouds
                solar = 1000 * (1 - (clouds/100) * 0.8)
        
        return WeatherData(
            timestamp=datetime.fromtimestamp(current.get('dt', 0)),
            temperature=current.get('temp', 293.15) - 273.15,  # K to C
            cloud_cover=clouds,
            humidity=current.get('humidity', 60),
            wind_speed=current.get('wind_speed', 0),
            solar_irradiance=max(0, solar)
        )
    
    def get_weather_forecast(self, lat: float, lon: float) -> List[WeatherData]:
        """Get weather forecast for a location."""
        # Check cache first
        cache_key = f"forecast_{lat:.4f}_{lon:.4f}"
        cached = self._load_cached_weather(lat, lon)
        if cached and cache_key in cached:
            return [WeatherData(**d) for d in cached[cache_key]]
        
        # Fetch from API
        url = f"{OPENWEATHER_API_BASE}/onecall"
        params = {
            'lat': lat,
            'lon': lon,
            'exclude': 'minutely,daily,alerts',
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process hourly forecast
            forecast = []
            for hour in data.get('hourly', [])[:24]:  # Next 24 hours
                weather = WeatherData(
                    timestamp=datetime.fromtimestamp(hour['dt']),
                    temperature=hour.get('temp', 20),
                    cloud_cover=hour.get('clouds', 50),
                    humidity=hour.get('humidity', 60),
                    wind_speed=hour.get('wind_speed', 0)
                )
                forecast.append(weather)
            
            # Cache the result
            cache_data = {
                cache_key: [w.to_dict() for w in forecast],
                'timestamp': datetime.now().isoformat()
            }
            self._save_weather_to_cache(lat, lon, cache_data)
            
            return forecast
            
        except (requests.RequestException, KeyError) as e:
            print(f"Error fetching weather data: {e}")
            return []

def get_weather_api_key() -> Optional[str]:
    """Get OpenWeatherMap API key from environment or config."""
    # Check environment variable first
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    # Then check config file
    if not api_key and os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                api_key = config.get('api_keys', {}).get('openweathermap')
        except (json.JSONDecodeError, KeyError):
            pass
    
    return api_key

def test_weather_integration():
    """Test the weather integration."""
    api_key = get_weather_api_key()
    if not api_key:
        print("Error: No OpenWeatherMap API key found.")
        print("Please set the OPENWEATHER_API_KEY environment variable or add it to config.json")
        return
    
    # Test with New York coordinates
    nyc_lat, nyc_lon = 40.7128, -74.0060
    
    weather_api = WeatherAPI(api_key)
    forecast = weather_api.get_weather_forecast(nyc_lat, nyc_lon)
    
    if forecast:
        print(f"Weather forecast for New York:")
        for hour in forecast[:6]:  # Show next 6 hours
            print(f"{hour.timestamp}: {hour.temperature:.1f}°C, "
                  f"Clouds: {hour.cloud_cover}%, "
                  f"Solar: {hour.solar_irradiance or 0:.0f} W/m²")
    else:
        print("Failed to fetch weather forecast")

if __name__ == "__main__":
    test_weather_integration()
