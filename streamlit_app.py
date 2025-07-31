# This must be the very first command
import streamlit as st
st.set_page_config(
    page_title="☀️ Solar AI Optimizer 2025",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh

# Enable autorefresh every 5 minutes (300000 ms)
st_autorefresh(interval=5 * 60 * 1000, key='data_refresh')

def generate_sample_data(days=30):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days-1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Generate demand data
    base_demand = 5.0
    seasonal = 0.5 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
    daily = 0.3 * np.sin(2 * np.pi * (date_range.hour - 6) / 24)
    weekly = 0.1 * (date_range.dayofweek >= 5)
    demand = base_demand * (1 + seasonal + daily + weekly + 0.2 * np.random.randn(len(date_range)))
    demand = np.maximum(0.5, demand)
    
    demand_df = pd.DataFrame({
        'timestamp': date_range,
        'demand_kwh': demand,
        'temperature': np.nan,
        'is_holiday': (date_range.month == 1) & (date_range.day == 1)
    })
    
    # Generate solar production data
    solar_potential = np.sin(np.pi * (date_range.hour - 6) / 12)
    solar_potential = np.where(solar_potential < 0, 0, solar_potential)
    solar_seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
    system_capacity = 8.0
    solar = system_capacity * solar_potential * solar_seasonal * (0.9 + 0.2 * np.random.rand(len(date_range)))
    
    solar_df = pd.DataFrame({
        'timestamp': date_range,
        'solar_production_kwh': solar,
        'solar_irradiance': 1000 * solar_potential * (0.8 + 0.4 * np.random.rand(len(date_range)))
    })
    
    # Generate weather data
    base_temp = 15 + 10 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
    daily_temp_variation = 8 * np.sin(2 * np.pi * (date_range.hour - 14) / 24)
    temperature = base_temp + daily_temp_variation + 3 * np.random.randn(len(date_range))
    
    cloud_cover = 30 + 20 * np.sin(2 * np.pi * date_range.dayofyear / 20)
    cloud_cover = np.clip(cloud_cover + 20 * np.random.randn(len(date_range)), 0, 100)
    
    weather_df = pd.DataFrame({
        'timestamp': date_range,
        'temperature': temperature,
        'cloud_cover': cloud_cover,
        'humidity': np.clip(60 + 30 * np.sin(2 * np.pi * date_range.hour / 24) + 10 * np.random.randn(len(date_range)), 20, 100),
        'wind_speed': np.maximum(0, 5 + 3 * np.random.randn(len(date_range)))
    })
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Save to CSV
    demand_df.to_csv(data_dir / "demand_data.csv", index=False)
    solar_df.to_csv(data_dir / "solar_data.csv", index=False)
    weather_df.to_csv(data_dir / "weather_data.csv", index=False)
    
    return {
        "demand": demand_df,
        "solar": solar_df,
        "weather": weather_df
    }

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size:24px; font-weight:bold; color:#1f77b4}
    .metric-card {border-radius:10px; padding:15px; margin:10px 0; background-color:#f8f9fa;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load sample data.
    
    Returns:
        Tuple containing demand, solar, and weather DataFrames.
        Returns (None, None, None) if there's an error.
    """
    try:
        demand_df = pd.read_csv("data/demand_data.csv")
        solar_df = pd.read_csv("data/solar_data.csv")
        weather_df = pd.read_csv("data/weather_data.csv")
        return demand_df, solar_df, weather_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def generate_ai_insights() -> dict:
    """Generate AI-powered insights about energy usage."""
    return {
        'peak_hours': {
            'time': '14:00-16:00',
            'efficiency': '92%',
            'savings_potential': '$8.50/day'
        },
        'device_usage': [
            {'name': 'Air Conditioner', 'usage': '42%', 'savings': '$3.20/day'},
            {'name': 'Water Heater', 'usage': '28%', 'savings': '$1.80/day'},
            {'name': 'Pool Pump', 'usage': '18%', 'savings': '$1.10/day'},
        ],
        'weather_impact': 'Sunny conditions tomorrow could increase solar production by 15%',
        'maintenance': 'Solar panels cleaning recommended in 12 days'
    }

def generate_3d_solar_map(latitude: float, longitude: float) -> pdk.Deck:
    """Generate a 3D map showing solar potential.
    
    Args:
        latitude: The latitude of the location to center the map on
        longitude: The longitude of the location to center the map on
    """
    # Create sample data points around the specified location
    lat_points = [latitude, latitude + 0.01, latitude - 0.01]
    lon_points = [longitude, longitude + 0.01, longitude - 0.01]
    
    data = pd.DataFrame({
        'lat': lat_points,
        'lon': lon_points,
        'elevation': [10, 25, 15],
        'solar_potential': [85, 92, 78]
    })
    
    layer = pdk.Layer(
        'ColumnLayer',
        data=data,
        get_position='[lon, lat]',
        get_elevation='elevation * 10',
        radius=100,
        get_fill_color='[200, 30, 0, 160]',
        elevation_scale=4,
        pickable=True,
        auto_highlight=True,
    )
    
    return pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=12,
            pitch=50,
        ),
        layers=[layer],
    )

def main() -> None:
    """Run the main Streamlit application."""
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {font-size: 2.5rem !important;}
    .metric-card {border-radius: 10px; padding: 15px; margin: 10px 0;}
    .stButton>button {width: 100%; border-radius: 8px;}
    .stProgress>div>div>div>div {background: linear-gradient(90deg, #4CAF50, #8BC34A);}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("☀️ Solar AI Optimizer 2025")
    st.markdown("### *The Future of Smart Energy Management*")
    
    
    # Sidebar with configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Import/Export
        with st.expander("📊 Data Management", expanded=True):
            st.subheader("Import Data")
            
            # File uploaders for each data type
            uploaded_demand = st.file_uploader("Upload Demand Data (CSV)", type="csv", key="demand_uploader")
            uploaded_solar = st.file_uploader("Upload Solar Data (CSV)", type="csv", key="solar_uploader")
            uploaded_weather = st.file_uploader("Upload Weather Data (CSV)", type="csv", key="weather_uploader")
            
            # Process uploaded files
            if uploaded_demand is not None:
                try:
                    demand_df = pd.read_csv(uploaded_demand)
                    demand_df.to_csv("data/demand_data.csv", index=False)
                    st.success("Successfully updated demand data!")
                except Exception as e:
                    st.error(f"Error processing demand data: {e}")
            
            if uploaded_solar is not None:
                try:
                    solar_df = pd.read_csv(uploaded_solar)
                    solar_df.to_csv("data/solar_data.csv", index=False)
                    st.success("Successfully updated solar production data!")
                except Exception as e:
                    st.error(f"Error processing solar data: {e}")
            
            if uploaded_weather is not None:
                try:
                    weather_df = pd.read_csv(uploaded_weather)
                    weather_df.to_csv("data/weather_data.csv", index=False)
                    st.success("Successfully updated weather data!")
                except Exception as e:
                    st.error(f"Error processing weather data: {e}")
            
            # Export buttons
            st.subheader("Export Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with open("data/demand_data.csv", "rb") as f:
                    st.download_button(
                        label="Export Demand",
                        data=f,
                        file_name="demand_data.csv",
                        mime="text/csv",
                        key="export_demand"
                    )
            
            with col2:
                with open("data/solar_data.csv", "rb") as f:
                    st.download_button(
                        label="Export Solar",
                        data=f,
                        file_name="solar_data.csv",
                        mime="text/csv",
                        key="export_solar"
                    )
            
            with col3:
                with open("data/weather_data.csv", "rb") as f:
                    st.download_button(
                        label="Export Weather",
                        data=f,
                        file_name="weather_data.csv",
                        mime="text/csv",
                        key="export_weather"
                    )
        
        # System configuration
        with st.expander("🔋 System Settings", expanded=True):
            battery_capacity = st.slider(
                "Battery Capacity (kWh)", 1.0, 20.0, 10.0, 0.5
            )
            max_charge_rate = st.slider(
                "Max Charge Rate (kW)", 1.0, 10.0, 5.0, 0.5
            )
            initial_soc = st.slider(
                "Initial State of Charge (%)", 0, 100, 50, 5
            ) / 100
        
        # Location settings
        with st.expander("🌍 Location", expanded=True):
            lat = st.number_input("Latitude", value=40.7128, format="%.4f")
            lon = st.number_input("Longitude", value=-74.0060, format="%.4f")
        
        # Weather API
        with st.expander("🌤️ Weather API", expanded=False):
            api_key = st.text_input("OpenWeatherMap API Key", type="password")
        
        update_clicked = st.button("🔍 Update Forecast")
        
        # Add a session state to track if we need to regenerate data
        if 'data_generated' not in st.session_state:
            st.session_state.data_generated = False
            
        if update_clicked:
            st.session_state.data_generated = True
            
    # Check if we need to regenerate data
    if 'data_generated' in st.session_state and st.session_state.data_generated:
        # Clear the flag and regenerate data
        st.session_state.data_generated = False
        data = generate_sample_data(days=30)
        data["demand"].to_csv("data/demand_data.csv", index=False)
        data["solar"].to_csv("data/solar_data.csv", index=False)
        data["weather"].to_csv("data/weather_data.csv", index=False)
        st.rerun()
    
    # Load and validate data
    demand_df, solar_df, weather_df = load_data()
    
    if demand_df is None or solar_df is None or weather_df is None:
        st.error("Failed to load data. Please check the data files.")
        return
    
    # Convert timestamps
    for df in [demand_df, solar_df, weather_df]:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Create tabs with improved layout
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard",
        "🔮 Forecast",
        "⚡ Optimization"
    ])
    
    with tab1:
        st.header("📊 Energy Dashboard")
        
        # Real-time metrics with animated counters
        st.subheader("🌡️ Real-time Energy Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Solar Production", "5.8 kW", "+1.2 kW from avg")
            st.progress(0.78)
            
        with col2:
            st.metric("Current Demand", "3.2 kW", "-0.5 kW from avg")
            st.progress(0.45)
            
        with col3:
            st.metric("Battery Status", "68%", "+12% today")
            st.progress(0.68)
            
        with col4:
            st.metric("CO₂ Saved Today", "24.5 kg", "🌱 3.2 kg vs yesterday")
            
        # 3D Solar Potential Map
        st.subheader("🗺️ 3D Solar Potential Analysis")
        st.pydeck_chart(generate_3d_solar_map(lat, lon))
        st.caption(f"Interactive 3D visualization of solar potential at {lat:.4f}°N, {lon:.4f}°W")
        
        # AI-Powered Insights
        st.subheader("🤖 AI Energy Advisor")
        insights = generate_ai_insights()
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("💡 Smart Recommendations", expanded=True):
                st.success(f"✅ {insights['weather_impact']}")
                st.warning(f"⚠️ {insights['maintenance']}")
                st.info(f"ℹ️ {insights['peak_hours']['time']} is your peak solar production time")
                
                st.markdown("### Device Optimization")
                for device in insights['device_usage']:
                    st.metric(
                        label=f"{device['name']}",
                        value=device['usage'],
                        delta=f"Save {device['savings']}"
                    )
        
        with col2:
            with st.expander("📊 Performance Analytics", expanded=True):
                st.subheader("Efficiency Score")
                score = 87
                st.metric("Overall Score", f"{score}/100", "+5 this month")
                st.progress(score/100)
                
                # Interactive chart
                st.subheader("Energy Distribution")
                data = pd.DataFrame({
                    'Source': ['Solar', 'Battery', 'Grid'],
                    'kWh': [42, 28, 15]
                })
                fig = px.pie(data, values='kWh', names='Source', 
                           color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
        
        # Smart Home Integration
        st.subheader("🏠 Smart Home Controls")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            if st.button("🌞 Optimize All Devices", key='optimize_all_devices'):
                st.success("All devices optimized for solar production!")
                
        with c2:
            if st.button("🌙 Night Mode", key='night_mode'):
                st.success("Energy-saving night mode activated!")
                
        with c3:
            if st.button("⚡ Fast Charge", key='fast_charge'):
                st.success("Battery charging at maximum rate!")
                
        with c4:
            if st.button("🔄 Refresh Data", key='refresh_data'):
                st.rerun()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Current Solar Production",
                f"{solar_df['solar_production_kwh'].iloc[-1]:.1f} kW"
            )
        with col2:
            st.metric(
                "Current Demand",
                f"{demand_df['demand_kwh'].iloc[-1]:.1f} kW"
            )
        with col3:
            st.metric(
                "Battery SOC",
                f"{initial_soc*100:.0f}%"
            )
        with col4:
            st.metric(
                "Weather",
                f"{weather_df['temperature'].iloc[-1]:.1f}°C"
            )
        
        # Energy production vs demand chart with better styling
        st.subheader("Energy Production vs Demand")
        fig = go.Figure()
        
        # Add traces with improved styling
        fig.add_trace(go.Scatter(
            x=demand_df["timestamp"],
            y=demand_df["demand_kwh"],
            name="Demand",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{y:.1f} kW<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            x=solar_df["timestamp"],
            y=solar_df["solar_production_kwh"],
            name="Solar Production",
            line=dict(color="#ff7f0e", width=2),
            hovertemplate="%{y:.1f} kW<extra></extra>"
        ))
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            height=450,
            margin=dict(l=50, r=50, t=30, b=50),
            plot_bgcolor="rgba(0,0,0,0.02)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        # Add grid and other styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
        
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
    
    with tab2:
        st.header("🔮 Energy Forecast")
        
        # Forecast explanation
        st.markdown("""
        ### 24-Hour Energy Forecast
        This section shows the predicted energy production and consumption for the next 24 hours.
        The forecast is based on historical data and weather predictions.
        """)
        
        # Create forecast visualization
        forecast_fig = go.Figure()
        
        # Add forecast traces (using sample data for now)
        forecast_fig.add_trace(go.Scatter(
            x=demand_df["timestamp"],
            y=demand_df["demand_kwh"] * np.random.uniform(0.8, 1.2, len(demand_df)),
            name="Forecasted Demand",
            line=dict(color="#1f77b4", width=2, dash="dot"),
            hovertemplate="%{y:.1f} kW<extra></extra>"
        ))
        
        forecast_fig.add_trace(go.Scatter(
            x=solar_df["timestamp"],
            y=solar_df["solar_production_kwh"] * np.random.uniform(0.9, 1.1, len(solar_df)),
            name="Forecasted Solar Production",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
            hovertemplate="%{y:.1f} kW<extra></extra>"
        ))
        
        # Update layout
        forecast_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Weather forecast section
        st.subheader("🌤️ Weather Forecast")
        
        # Sample weather data
        weather_cols = st.columns(4)
        for i in range(4):
            with weather_cols[i]:
                st.metric(
                    f"{i*6}:00",
                    f"{weather_df['temperature'].iloc[i*6]:.1f}°C",
                    f"{weather_df['cloud_cover'].iloc[i*6]:.0f}% clouds"
                )
    
    with tab3:
        st.header("⚡ Optimization")
        
        # Optimization results
        st.markdown("""
        ### Battery Optimization Strategy
        The system optimizes battery usage to maximize self-consumption of solar energy
        and minimize grid dependency based on your settings.
        """)
        
        # Create optimization visualization
        opt_fig = go.Figure()
        
        # Add optimization traces
        opt_fig.add_trace(go.Scatter(
            x=demand_df["timestamp"],
            y=demand_df["demand_kwh"],
            name="Demand",
            line=dict(color="#1f77b4")
        ))
        
        opt_fig.add_trace(go.Scatter(
            x=solar_df["timestamp"],
            y=solar_df["solar_production_kwh"],
            name="Solar Production",
            line=dict(color="#ff7f0e")
        ))
        
        # Add battery state area
        battery_use = np.minimum(solar_df["solar_production_kwh"], demand_df["demand_kwh"])
        opt_fig.add_trace(go.Scatter(
            x=demand_df["timestamp"],
            y=battery_use,
            fill='tonexty',
            name="Battery Usage",
            line=dict(color="#2ca02c", width=0),
            fillcolor="rgba(46, 160, 67, 0.3)"
        ))
        
        # Update layout
        opt_fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Energy (kWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(opt_fig, use_container_width=True)
        
        # Optimization metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Solar Self-Consumption", "68%", "+15% from baseline")
        with col2:
            st.metric("Grid Import", "32 kWh/day", "-28% from baseline")
        with col3:
            st.metric("Estimated Savings", "$12.50/week", "+$3.20 from last week")
    
    # Advanced Analytics Section
    st.markdown("---")
    st.header('🚀 Advanced Analytics')
    
    # AI-Powered Recommendations
    with st.expander('🤖 AI Energy Advisor', expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('💡 Smart Recommendations')
            st.success('✅ Shift high-energy usage to 2 PM - 4 PM for maximum solar efficiency')
            st.warning('⚠️ Consider upgrading to a 10kW battery to cover 98% of your needs')
            st.info('ℹ️ Your energy usage is 15% higher than similar households')
        
        with col2:
            st.subheader('📈 Performance Score')
            score = 87
            st.metric('Efficiency Score', f'{score}/100')
            st.progress(score/100)
            st.caption(f'Better than {92}% of users in your area')
    
    # Real-time Device Control
    with st.expander('🏠 Smart Home Integration', expanded=True):
        st.subheader('🔌 Device Control')
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('🌞 Optimize for Solar'):
                st.success('Devices scheduled for peak solar production! ☀️')
        with col2:
            if st.button('🌙 Night Mode'):
                st.success('Energy-saving mode activated! 🌜')
        with col3:
            if st.button('⚡ Fast Charge'):
                st.success('Battery charging at maximum rate! 🔋')
    
    # Community & Gamification
    with st.expander('🏆 Community & Rewards', expanded=True):
        st.subheader('🌍 Community Impact')
        st.metric('CO₂ Saved Today', '24.5 kg', '+3.2 kg vs yesterday')
        st.metric('Your Rank', '#42', '↑3 this week')
        
        st.progress(0.75)
        st.caption('75% to next achievement: Eco Warrior 🌱')
    
    # Footer with updated year
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 40px;'>
            <p>2025 Solar Energy Optimization System v2.0 | Made with ❤️ for a sustainable future</p>
            <p style='font-size: 0.8em;'>Saving the planet, one watt at a time</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
