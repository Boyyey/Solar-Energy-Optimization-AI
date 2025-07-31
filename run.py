#!/usr/bin/env python3
"""
Solar Energy Optimization System - Main Entry Point

This script provides a command-line interface to run the Solar Energy Optimization System.
It can be used to generate forecasts, optimize energy usage, and start the web dashboard.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('solar_optimizer.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from src.forecast import EnergyDemandForecaster, SolarProductionForecaster
from src.optimizer import EnergyOptimizer
from src.weather_integration import WeatherAPI, get_weather_api_key
from src.utils import load_config, save_config, ensure_data_directories

class SolarEnergyOptimizer:
    """Main class for the Solar Energy Optimization System."""
    
    def __init__(self, config_path='config.json'):
        """Initialize the Solar Energy Optimizer."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Ensure required directories exist
        ensure_data_directories()
        
        # Initialize components
        self.weather_api = WeatherAPI(
            api_key=get_weather_api_key() or self.config.get('api_keys', {}).get('openweathermap')
        )
        
        self.demand_forecaster = EnergyDemandForecaster(
            model_type=self.config.get('forecasting', {}).get('demand_model_type', 'prophet')
        )
        
        self.solar_forecaster = SolarProductionForecaster(
            model_type=self.config.get('forecasting', {}).get('solar_model_type', 'prophet')
        )
        
        self.optimizer = EnergyOptimizer(
            battery_capacity_kwh=self.config.get('system', {}).get('battery_capacity_kwh', 10.0),
            max_charge_rate_kw=self.config.get('system', {}).get('max_charge_rate_kw', 5.0),
            battery_efficiency=self.config.get('system', {}).get('battery_efficiency', 0.95),
            initial_soc=self.config.get('system', {}).get('initial_soc', 0.5)
        )
        
        # Initialize data structures
        self.forecast_horizon = self.config.get('system', {}).get('forecast_horizon_hours', 24)
        self.location = self.config.get('location', {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'timezone': 'America/New_York'
        })
    
    def update_weather_forecast(self) -> pd.DataFrame:
        """Fetch and update weather forecast data."""
        logger.info("Updating weather forecast...")
        
        forecast = self.weather_api.get_weather_forecast(
            self.location['latitude'],
            self.location['longitude']
        )
        
        if not forecast:
            logger.error("Failed to fetch weather forecast")
            return None
        
        # Convert to DataFrame
        weather_data = []
        for entry in forecast:
            data = entry.to_dict()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            weather_data.append(data)
        
        weather_df = pd.DataFrame(weather_data)
        weather_df.set_index('timestamp', inplace=True)
        
        logger.info(f"Updated weather forecast with {len(weather_df)} hours of data")
        return weather_df
    
    def generate_forecasts(self, weather_df: pd.DataFrame) -> tuple:
        """Generate demand and solar production forecasts."""
        logger.info("Generating energy forecasts...")
        
        # Prepare data for forecasting
        timestamps = pd.date_range(
            start=datetime.now(),
            periods=self.forecast_horizon,
            freq='H'
        )
        
        # Generate demand forecast
        demand_forecast = self.demand_forecaster.predict(timestamps)
        
        # Generate solar production forecast
        solar_forecast = self.solar_forecaster.predict(
            timestamps,
            weather_df[['temperature', 'cloud_cover']].reset_index()
        )
        
        logger.info("Energy forecasts generated successfully")
        return demand_forecast, solar_forecast
    
    def optimize_schedule(self, demand_forecast: pd.Series, solar_forecast: pd.Series) -> dict:
        """Optimize energy usage schedule."""
        logger.info("Optimizing energy schedule...")
        
        # Get electricity prices (simple time-of-use pricing for now)
        # In a real application, this would come from a utility API or configuration
        prices = pd.Series(
            index=demand_forecast.index,
            data=self.config.get('electricity', {}).get('buy_price', 0.15)
        )
        
        # Optimize schedule
        schedule = self.optimizer.optimize_schedule(
            demand_forecast=demand_forecast,
            production_forecast=solar_forecast,
            electricity_prices=prices
        )
        
        # Calculate savings
        savings = self.optimizer.calculate_savings(schedule, prices)
        
        logger.info(f"Optimization complete. Estimated savings: ${savings['savings']:.2f} ({savings['savings_percent']:.1f}%)")
        return schedule, savings
    
    def run_optimization_cycle(self):
        """Run a complete optimization cycle."""
        try:
            # Update weather data
            weather_df = self.update_weather_forecast()
            if weather_df is None:
                logger.error("Weather update failed, using last known data")
                return None
            
            # Generate forecasts
            demand_forecast, solar_forecast = self.generate_forecasts(weather_df)
            
            # Optimize schedule
            schedule, savings = self.optimize_schedule(demand_forecast, solar_forecast)
            
            # Prepare results
            results = {
                'timestamp': datetime.now().isoformat(),
                'demand_forecast': demand_forecast.to_dict(),
                'solar_forecast': solar_forecast.to_dict(),
                'schedule': {k: v.to_dict() for k, v in schedule.items()},
                'savings': savings
            }
            
            # Save results
            self.save_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during optimization cycle: {e}", exc_info=True)
            return None
    
    def save_results(self, results: dict):
        """Save optimization results to disk."""
        try:
            # Ensure results directory exists
            results_dir = os.path.join('data', 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save to JSON file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(results_dir, f'optimization_{timestamp}.json')
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solar Energy Optimization System')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run optimization cycle
    run_parser = subparsers.add_parser('run', help='Run a single optimization cycle')
    run_parser.add_argument('--config', default='config.json', help='Path to configuration file')
    
    # Start dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Start the web dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    dashboard_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    dashboard_parser.add_argument('--config', default='config.json', help='Path to configuration file')
    
    # Generate sample data
    sample_data_parser = subparsers.add_parser('generate-sample-data', 
                                              help='Generate sample data for testing')
    sample_data_parser.add_argument('--days', type=int, default=30, 
                                   help='Number of days of sample data to generate')
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    if args.command == 'run':
        # Run a single optimization cycle
        optimizer = SolarEnergyOptimizer(args.config)
        results = optimizer.run_optimization_cycle()
        
        if results:
            print("Optimization completed successfully!")
            print(f"Estimated savings: ${results['savings']['savings']:.2f} ({results['savings']['savings_percent']:.1f}%)")
        else:
            print("Optimization failed. Check the logs for details.")
            sys.exit(1)
    
    elif args.command == 'dashboard':
        # Start the web dashboard
        print(f"Starting dashboard on http://localhost:{args.port}")
        from src.dashboard import app
        app.run_server(debug=args.debug, port=args.port)
    
    elif args.command == 'generate-sample-data':
        # Generate sample data
        print(f"Generating {args.days} days of sample data...")
        from notebooks.generate_sample_data import generate_sample_data, save_sample_data
        data = generate_sample_data(days=args.days)
        save_sample_data(data)
        print("Sample data generated successfully!")
    
    else:
        print("Please specify a command. Use --help for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
