import streamlit as st
import pandas as pd
import datetime
import folium
import joblib
import os
import numpy as np
from streamlit_folium import st_folium

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Inject minimal custom CSS to tighten up the default Streamlit sidebar widget gaps
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div.stElementContainer {
        margin-bottom: -10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Safely load and cache the baseline station data
@st.cache_data
def load_data():
    df = pd.read_csv("data/selected_stations_60.csv")
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)
    df = df.sort_values(by="station_name").reset_index(drop=True)
    return df

# Cache model loading so joblib doesn't re-read disk on every widget shift
@st.cache_resource
def load_ml_model(model_filename):
    model_path = os.path.join("models", model_filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"⚠️ Model file not found at: `{model_path}`")
        return None

try:
    df_stations = load_data()
    
    # 1. SIDEBAR SECTION
    st.sidebar.header("Navigation & Configurations")

    view_mode = st.sidebar.radio(
        label="Select View Mode:",
        options=["Map View", "Chart View"],
        index=0
    )
    
    st.sidebar.markdown("---")

    today = datetime.date.today()
    selected_date = st.sidebar.date_input(
        label="Select a Forecast Date",
        value=today,
        min_value=today
    )
    
    st.sidebar.markdown("---")

    station_options = ["Show All Stations"] + list(df_stations['station_name'])
    selected_station = st.sidebar.selectbox(
        label="Choose a Station:",
        options=station_options,
        index=0
    )

    st.sidebar.markdown("---")

    # Dynamic Model Selector Configuration
    model_mapping = {
        "LightGBM": "bicycle_lightgbm.pkl",
        "XGBoost": "bicycle_xgboost.pkl",
        "GRU": "bicycle_gru.pkl"
    }
    
    selected_model_name = st.sidebar.selectbox(
        label="Select Prediction Model:",
        options=list(model_mapping.keys())
    )
    
    target_filename = model_mapping[selected_model_name]
    model = load_ml_model(target_filename)

    # 2. HELPER FUNCTION TO GENERATE STANDARDIZED FEATURE MATRIX
    def build_feature_dataframe(base_df, date_obj, target_hour=None):
        rows_count = len(base_df)
        
        # Determine hourly distribution context
        if target_hour is None:
            hours_list = base_df["hour"].tolist() if "hour" in base_df.columns else [0] * rows_count
        elif isinstance(target_hour, list):
            hours_list = target_hour
        else:
            hours_list = [target_hour] * rows_count

        # Build DataFrame with the exact features your model requires
        features = pd.DataFrame()
        features["hour"] = hours_list
        features["day_of_week"] = [date_obj.weekday()] * rows_count
        features["month"] = [date_obj.month] * rows_count
        
        is_wknd = 1 if date_obj.weekday() >= 5 else 0
        features["is_weekend"] = [is_wknd] * rows_count
        
        # Map dynamic hourly commute string tags matching your training segments
        features["commute_period"] = [
            "morning" if (7 <= h <= 9) else ("evening" if (17 <= h <= 19) else "none") for h in hours_list
        ]
        
        # Map spatial parameters directly from selected_stations_60.csv columns
        features["latitude"] = base_df["lat"]
        features["longitude"] = base_df["lon"]
        features["region"] = base_df["gu"]
        features["station_group"] = base_df["group"]
        features["subway_dist_m"] = base_df["subway_dist_m_2021"] if "subway_dist_m_2021" in base_df.columns else 100.0
        features["is_station_area"] = base_df["became_station_area_at"].notnull().astype(int) if "became_station_area_at" in base_df.columns else 1
        
        # Add derived date attributes matching training rules
        features["is_holiday"] = [0] * rows_count
        features["holiday_type"] = ["none"] * rows_count  
        features["is_offday"] = [is_wknd] * rows_count
        features["is_day_before_offday"] = [1 if date_obj.weekday() in [4, 5] else 0] * rows_count
        features["is_day_after_offday"] = [1 if date_obj.weekday() in [0, 6] else 0] * rows_count
        features["offday_block_len"] = [2 if is_wknd else 0] * rows_count
        features["is_labor_day"] = [0] * rows_count
        features["dock_count"] = [20.0] * rows_count

        # Exact 39 training feature sequence verified by your diagnostic script
        expected_features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'commute_period', 'inventory_lag_1h', 
            'latitude', 'longitude', 'region', 'dock_count', 'rentals_lag_1h', 'rentals_lag_24h', 
            'rentals_lag_168h', 'rentals_rolling_mean_24h', 'returns_lag_1h', 'returns_lag_24h', 
            'returns_lag_168h', 'returns_rolling_mean_24h', 'station_group', 'subway_dist_m', 
            'is_station_area', 'wx_obs_present_lag1h', 'temp_c_lag1h', 'humidity_pct_lag1h', 
            'wind_ms_lag1h', 'precip_mm_lag1h', 'snow_depth_cm_lag1h', 'precip_imputed_lag1h', 
            'precip_sum_3h', 'rain_flag_lag1h', 'rain_any_3h', 'temp_c_change_3h', 'is_holiday', 
            'holiday_type', 'is_offday', 'is_day_before_offday', 'is_day_after_offday', 
            'offday_block_len', 'is_labor_day'
        ]

        # Populate any remaining fields (lags/weather constants) with default 0.0 values
        for col in expected_features:
            if col not in features.columns:
                features[col] = 0.0

        # Return correctly sorted feature matrix
        return features[expected_features], expected_features


    # 3. HELPER FUNCTION TO RUN COMPATIBLE ROUTED MODEL INFERENCE
    def execute_inference(X_input_df, expected_cols):
        cat_cols = ['station_group', 'region', 'commute_period', 'holiday_type']
        
        if "GRU" in selected_model_name:
            gru_dict = {}
            for col in cat_cols:
                # GRU Fix: Factorize categories to numeric codes and flatten to a 1D array per station
                codes, _ = pd.factorize(X_input_df[col].astype(str))
                gru_dict[f"input_{col}"] = codes.astype(np.float64)
            
            # GRU Fix: Explicitly filter down the numeric block to exactly 35 columns to match the layer shape
            gru_num_cols = [c for c in expected_cols if c not in cat_cols]
            gru_dict["input_numeric"] = X_input_df[gru_num_cols].values.astype(np.float64)
            
            # Execute Keras predict on the input dictionary
            preds = model.predict(gru_dict, verbose=0).flatten()
            return np.clip(preds, 0, None)
            
        elif "XGBoost" in selected_model_name or "LightGBM" in selected_model_name:
            # Tree-based pipeline: Cast the four core categorical columns to strict category type
            xgb_lgb_features = X_input_df.copy()
            for col in cat_cols:
                xgb_lgb_features[col] = xgb_lgb_features[col].astype('category')
                
            preds = model.predict(xgb_lgb_features)
            return np.clip(preds, 0, None)
            
        else:
            return np.zeros(len(X_input_df))

    # 4. MAIN CONTAINER AND RENDERING SECTION
    st.title("Seoul Station Interactive Prediction Hub")
    st.write(f"Active Model: **{selected_model_name}** | Target Date: **{selected_date}**")

    if view_mode == "Map View":
        st.subheader("🗺️ Geographic Forecast Layout")

        # Compute map preview predictions (using morning peak hour 8 AM as a baseline focus)
        if model is not None:
            try:
                X_map, expected_cols = build_feature_dataframe(df_stations, selected_date, target_hour=8)
                df_stations['predicted_commutes'] = execute_inference(X_map, expected_cols).astype(int)
            except Exception as e:
                st.sidebar.error(f"Prediction Pipeline Error: {e}")
                df_stations['predicted_commutes'] = df_stations['commute_total'].fillna(0).astype(int)
        else:
            df_stations['predicted_commutes'] = 0

        # Set map center point based on selections
        if selected_station == "Show All Stations":
            map_center = [37.5665, 126.9780]
            zoom_level = 11
        else:
            station_info = df_stations[df_stations['station_name'] == selected_station].iloc[0]
            map_center = [station_info['lat'], station_info['lon']]
            zoom_level = 15

        # Initialize OpenStreetMap
        m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="OpenStreetMap")

        # Render markers onto map canvas
        for idx, row in df_stations.iterrows():
            commutes = f"{int(row['commute_total']):,}" if pd.notnull(row['commute_total']) else "N/A"
            pred_val = row['predicted_commutes']
            predicted_commutes = f"{int(pred_val):,}" if pd.notnull(pred_val) else "N/A"
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 13px; width: 240px; line-height: 1.4;">
                <h4 style="margin: 0 0 5px 0; color: #1E3A8A;">{row['station_name']}</h4>
                <b>Gu:</b> {row['gu']}<br>
                <b>Nearest Subway:</b> {row['nearest_subway_2021']}<br>
                <hr style="border: 0; border-top: 1px solid #CCC; margin: 8px 0;">
                <span style="color: #2563EB;"><b>Historical Volume:</b> {commutes}</span><br>
                <span style="color: #DC2626;"><b>{selected_model_name} Prediction:</b> {predicted_commutes}</span>
            </div>
            """
            is_active = (row['station_name'] == selected_station)
            marker_color = "#DC2626" if is_active else "#2563EB"
            
            marker = folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=14 if is_active else 8,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=row['station_name'],
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.6,
                weight=2
            )
            if is_active:
                marker.add_child(folium.Popup(popup_html, max_width=300).add_to(marker))
            marker.add_to(m)

        st_folium(m, width="100%", height=600, returned_objects=[])

    else:
        st.subheader("📈 Chronological Forecast Timeline")
        
        if selected_station == "Show All Stations":
            st.warning("⚠️ **Please select a specific station** in the sidebar to view its chronological forecast timeline.")
            
            hours = [f"{str(h).zfill(2)}:00" for h in range(24)]
            empty_df = pd.DataFrame({
                "Time of Day": hours,
                f"{selected_model_name} Expected Demand": [None] * 24
            }).set_index("Time of Day")
            
            st.line_chart(empty_df, height=450)
            
        else:
            st.write(f"Showing predicted bicycle demand at **{selected_station}** for the full day of **{selected_date}**.")
            station_info = df_stations[df_stations['station_name'] == selected_station].iloc[0]

            hours = [f"{str(h).zfill(2)}:00" for h in range(24)]
            
            # Replicate row parameters into a 24-hour block for timeline evaluation
            station_repeated_df = pd.DataFrame([station_info] * 24).reset_index(drop=True)
            station_repeated_df["hour"] = list(range(24))

            if model is not None:
                try:
                    X_chart, expected_cols = build_feature_dataframe(station_repeated_df, selected_date)
                    predicted_timeline = execute_inference(X_chart, expected_cols).astype(int).tolist()
                except Exception as e:
                    st.warning(f"Could not calculate model timeline predictions: {e}")
                    predicted_timeline = [0] * 24
            else:
                predicted_timeline = [0] * 24

            timeline_df = pd.DataFrame({
                "Time of Day": hours,
                f"{selected_model_name} Expected Demand": predicted_timeline
            }).set_index("Time of Day")

            st.line_chart(timeline_df, height=450)
            
            st.markdown("### Forecast Schedule Summary")
            st.dataframe(timeline_df.T, use_container_width=True)

except FileNotFoundError:
    st.error("⚠️ Could not find your CSV file at `data/selected_stations_60.csv`.")
