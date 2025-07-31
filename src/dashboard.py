"""Dashboard module for the Solar Energy Optimization System."""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc

# Import forecasting and optimization modules
try:
    # When running as a module
    from .forecast import EnergyDemandForecaster, SolarProductionForecaster
    from .optimizer import EnergyOptimizer
except ImportError:
    # When running directly
    from forecast import EnergyDemandForecaster, SolarProductionForecaster
    from optimizer import EnergyOptimizer

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Solar Energy Optimizer"

# Sample data generation function (to be replaced with real data)
def generate_sample_data():
    """Generate sample data for demonstration."""
    # Create a date range for the next 24 hours
    now = datetime.now()
    hours = pd.date_range(now, now + timedelta(hours=23), freq='H')
    
    # Generate sample demand (higher during evening)
    demand = 1.5 + np.sin(np.linspace(0, 4*np.pi, 24)) * 0.8
    demand = np.maximum(0.5, demand)  # Minimum 0.5 kW
    
    # Generate sample solar production (0 at night, peaks at noon)
    solar = 3 * np.sin(np.linspace(0, np.pi, 24)) ** 2
    
    # Generate sample electricity prices (higher during peak hours)
    prices = 0.15 + 0.1 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 24))
    
    return pd.DataFrame({
        'timestamp': hours,
        'demand_kwh': demand,
        'solar_production_kwh': solar,
        'electricity_price': prices,
        'temperature': 20 + 10 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 24)),
        'cloud_cover': 30 + 40 * np.sin(np.linspace(0, 2*np.pi, 24)) ** 2
    })

# Generate sample data
df = generate_sample_data()

# Initialize forecasters and optimizer
demand_forecaster = EnergyDemandForecaster(model_type='prophet')
solar_forecaster = SolarProductionForecaster(model_type='prophet')
optimizer = EnergyOptimizer(battery_capacity_kwh=10.0, max_charge_rate_kw=5.0)

# Train models (in a real app, this would use historical data)
# For now, we'll use the sample data as "historical"
demand_forecaster.fit(df[['timestamp', 'demand_kwh']])
solar_forecaster.fit(
    df[['timestamp', 'solar_production_kwh']],
    df[['timestamp', 'temperature', 'cloud_cover']]
)

# Generate forecasts
forecast_hours = pd.date_range(
    datetime.now().replace(minute=0, second=0, microsecond=0),
    periods=24,
    freq='H'
)

demand_forecast = demand_forecaster.predict(forecast_hours)
weather_forecast = df[['timestamp', 'temperature', 'cloud_cover']].copy()
weather_forecast['timestamp'] = forecast_hours[:len(weather_forecast)]
solar_forecast = solar_forecaster.predict(forecast_hours, weather_forecast)

# Optimize schedule
schedule = optimizer.optimize_schedule(
    demand_forecast=demand_forecast,
    production_forecast=solar_forecast,
    electricity_prices=pd.Series(0.15, index=forecast_hours)  # Flat rate for demo
)

# Calculate savings metrics
savings = optimizer.calculate_savings(
    schedule=schedule,
    electricity_prices=pd.Series(0.15, index=forecast_hours)
)

# Create the app layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("☀️ Solar Energy Optimizer", className="text-center my-4"), width=12)
    ]),
    
    # Summary Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Solar Production", className="card-title"),
                    html.H3(f"{solar_forecast.sum():.1f} kWh", className="text-success"),
                    html.Small("Next 24 hours", className="text-muted")
                ])
            ], className="mb-4")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Energy Demand", className="card-title"),
                    html.H3(f"{demand_forecast.sum():.1f} kWh", className="text-primary"),
                    html.Small("Next 24 hours", className="text-muted")
                ])
            ], className="mb-4")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Cost Savings", className="card-title"),
                    html.H3(f"${savings['savings']:.2f}", className="text-success"),
                    html.Small(f"{savings['savings_percent']:.1f}% vs no optimization", className="text-muted")
                ])
            ], className="mb-4")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Self-Consumption", className="card-title"),
                    html.H3(f"{savings['self_consumption_ratio']*100:.1f}%", className="text-info"),
                    html.Small("of solar energy used on-site", className="text-muted")
                ])
            ], className="mb-4")
        ], md=3)
    ]),
    
    # Main charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Energy Production vs Demand"),
                dbc.CardBody([
                    dcc.Graph(
                        id='energy-chart',
                        figure={
                            'data': [
                                {'x': forecast_hours, 'y': demand_forecast, 'type': 'line', 'name': 'Demand Forecast', 'line': {'color': '#1f77b4'}},
                                {'x': forecast_hours, 'y': solar_forecast, 'type': 'line', 'name': 'Solar Forecast', 'line': {'color': '#ff7f0e'}},
                            ],
                            'layout': {
                                'xaxis': {'title': 'Time'},
                                'yaxis': {'title': 'Energy (kWh)'},
                                'hovermode': 'x unified'
                            }
                        }
                    )
                ])
            ], className="mb-4")
        ], md=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Battery Status"),
                dbc.CardBody([
                    dcc.Graph(
                        id='battery-chart',
                        figure={
                            'data': [
                                {
                                    'values': [schedule['battery_charge'].sum(), 
                                              schedule['battery_discharge'].sum(), 
                                              max(0, 10 - schedule['battery_charge'].sum() + schedule['battery_discharge'].sum())],
                                    'labels': ['Charged', 'Discharged', 'Remaining'],
                                    'type': 'pie',
                                    'hole': 0.6,
                                    'marker': {'colors': ['#2ca02c', '#d62728', '#e0e0e0']}
                                }
                            ],
                            'layout': {
                                'showlegend': True,
                                'margin': {'t': 0, 'b': 0, 'l': 0, 'r': 0}
                            }
                        }
                    ),
                    html.Div([
                        html.H5(f"Current SOC: {schedule['battery_soc'].iloc[0]*100:.1f}%", className="text-center mt-3"),
                        dbc.Progress(
                            value=schedule['battery_soc'].iloc[0]*100,
                            max=100,
                            style={"height": "30px"},
                            className="my-2"
                        )
                    ])
                ])
            ], className="mb-4")
        ], md=4)
    ]),
    
    # Schedule and optimization details
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Optimized Schedule"),
                dbc.CardBody([
                    dcc.Graph(
                        id='schedule-chart',
                        figure={
                            'data': [
                                {'x': forecast_hours, 'y': schedule['battery_charge'], 'type': 'bar', 'name': 'Battery Charge', 'marker': {'color': '#2ca02c'}},
                                {'x': forecast_hours, 'y': schedule['battery_discharge'], 'type': 'bar', 'name': 'Battery Discharge', 'marker': {'color': '#d62728'}},
                                {'x': forecast_hours, 'y': schedule['grid_import'], 'type': 'bar', 'name': 'Grid Import', 'marker': {'color': '#7f7f7f'}},
                                {'x': forecast_hours, 'y': -schedule['grid_export'], 'type': 'bar', 'name': 'Grid Export', 'marker': {'color': '#17becf'}},
                            ],
                            'layout': {
                                'barmode': 'relative',
                                'xaxis': {'title': 'Time'},
                                'yaxis': {'title': 'Energy (kWh)'},
                                'legend': {'orientation': 'h', 'y': -0.2}
                            }
                        }
                    )
                ])
            ], className="mb-4")
        ], md=12)
    ]),
    
    # Raw data table (collapsible)
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "Show Raw Data",
                id="collapse-button",
                className="mb-3",
                color="primary",
                n_clicks=0,
            ),
            dbc.Collapse(
                dbc.Card(dbc.CardBody([
                    html.Div(
                        id='table-container',
                        children=[
                            dash.dash_table.DataTable(
                                id='data-table',
                                columns=[{"name": i, "id": i} for i in df.columns],
                                data=df.to_dict('records'),
                                page_size=10,
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                },
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    }
                                ]
                            )
                        ]
                    )
                ])),
                id="collapse",
                is_open=False,
            )
        ], md=12)
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col(html.Div([
            html.Hr(),
            html.P(
                "© 2023 Solar Energy Optimizer | "
                "Data is simulated for demonstration purposes",
                className="text-center text-muted"
            )
        ]), width=12)
    ])
], fluid=True)

# Callback for the collapsible raw data section
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
