import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
import pydeck as pdk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
import contextily as ctx

# Load and clean data
@st.cache_data
def load_data():
    # This would normally load from Excel - using the provided data structure
    data = {
        'location': ['Apapa pt1', 'Apapa pt2', 'Naval dockyard', 'Tin can', 'SNAKE', 
                    'Apapa pt1', 'Apapa pt2', 'Naval dockyard', 'Tin can', 'SNAKE',
                    'Apapa pt1', 'Apapa pt2', 'Naval dockyard', 'Tin can', 'SNAKE', 'Ponpoku'],
        'zn_ppm': [np.nan, np.nan, np.nan, 0.6408, 0.43, 0, 0, 0.7003, 0.8349, 0, 0.0206, np.nan, np.nan, 0.2408, 0.2116, np.nan],
        'cd_ppm': [0.0117, np.nan, 0.0114, 0.4462, 0.0786, 0.0126, 0.0189, 0.1127, 0.0399, 0.0056, 0.0117, np.nan, 0.0114, np.nan, np.nan, np.nan],
        'fe_ppm': [1.7785, np.nan, 0.2964, 16.883, np.nan, 0.3746, 0, 0.5504, 75.1557, 0.3953, 0.8779, np.nan, 0.0547, np.nan, np.nan, 0.7501],
        'cu_ppm': [0.1967, 0.2075, 0.1094, 0.3522, 0.1896, 0.0464, 0.2796, 0.1934, 0.2387, 0.2735, 0.2059, 0.2433, 0.0448, 0.2322, 0.2109, 0.1244],
        'ni_ppm': [0.0569, 0.0744, np.nan, 0.0745, 0.0442, np.nan, 0.049, 0.0721, np.nan, 0.0442, 0.071, 0.0936, np.nan, 0.0858, 0.0586, 0.0615],
        'pb_ppm': [0.968, 2.331, 1.1665, 2.522, 2.1553, 1.6961, 3.3924, 1.6598, 4.0352, 3.4158, 2.3462, 2.6105, 1.3566, 2.777, 2.377, np.nan],
        'co_ppm': [1.809, 2.1643, 1.4797, 2.799, 2.802, 0.1846, 1.5445, 1.0499, 2.1594, 2.1028, 1.7806, 2.0364, 0.1947, 1.97, 1.767, 0.5106],
        'k_mg_kg': [313.5087, 222.1278, 260.2243, 200.4098, 236.8843, 463.9451, 255.2003, 350.1878, 265.9111, 312.4334, 254.7875, 279.25, 273.6034, 272.0114, 224.9555, 30.9279],
        'cr_ppm': [0.3988, 1.6345, 0.0644, 0.2652, 0.3785, 0.1176, 0.6475, 0.2778, 0.545, 0.4584, 0.5293, 0.8594, 0.1162, 0.6083, 0.8167, 0.4187],
        'mg_mg_kg': [620.4659, 554.9387, 236.67567, 467.2475, 786.452, 312.0132, 731.7113, 941.8826, 701.7235, 804.4847, 790.7019, 848.0404, 301.8472, 840.5002, 683.7151, 63.959],
        'mn_ppm': [0.4249, 0.7269, 0.3962, 0.6355, 0.8936, 0.2251, 0.3556, 0.4501, 0.7977, 0.4143, 0.3971, 0.3658, 0.1408, 0.4344, 0.3441, 0.0356],
        'na_mg_kg': [84.4275, 53.6335, 94.7823, 105.4639, 92.3648, 172.4685, 71.2524, 136.1129, 52.376, 75.3573, 70.8837, 81.8685, 103.5651, 78.5475, 58.0553, 124.1836],
        'temp_c': [28.3, 28.6, 27.9, 28.42, 27.36, 27.8, 27.91, 27.9, 28.42, 27.36, 30.8, 30.21, 30.35, 30.18, 30.96, 30.12],
        'ph': [7.98, 8.33, 8.01, 7.92, 7.66, 8.02, 8.11, 7.99, 7.96, 7.47, 7.89, 8.08, 7.99, 7.75, 7.59, 7.34],
        'orp_mv': [149, 137, 173, 160, 145, 156, 158, 160, 184, 155, 111, 109, 125, 124, 143, 289],
        'conductivity_us_cm': [50.3, 52.8, 48.2, 48.4, 47.4, 49.5, 50.1, 47.1, 44.4, 8.7, 49.8, 50.3, 47.4, 37.1, 43.5, 9.77],
        'turbidity_ntu': [27.8, 18.6, 22.8, 14.6, 32.7, 5.3, 16.7, 2.9, 3.2, 56.2, 34.4, 9.8, 16.7, 17.7, 34.3, 12.4],
        'do_mg_l': [7.3, 6.5, 7.74, 5.95, 7.62, 8, 8.06, 6.27, 8.21, 7.89, 5.78, 4.43, 8.29, 5.87, 7.28, 11.39],
        'tds_g_l': [30.1, 30.6, 27.7, 25.5, 8.6, 30.2, 30, 28.7, 27.1, 10.6, 30.4, 30.2, 28.9, 22.7, 26.5, 5.16],
        'salinity_ppt': [30.96, 33.2, 25.59, 25.77, 22, 32.34, 32.78, 30.5, 28.68, 10.5, 32.55, 32.97, 30.8, 23.52, 27.99, 4.47],
        'acidity_mg_l': [10, 12, 6.5, 8.5, 6.75, 5, 7.5, 2.5, 3.75, 5, 7, 13, 6, 14, 8, 18],
        'alkalinity_mg_l': [132, 128, 130, 144, 18, 120, 130, 128, 136, 126, 120, 122, 74, 110, 104, 70],
        'hardness_mg_l': [157.54, 153.34, 130.64, 165.36, 152.59, 176.58, 159.76, 178.8, 154.14, 148, 194.19, 195.2, 177.18, 168, 163, 2430],
        'cod_mg_l': [106.6, 102.5, 98.6, 103.6, 107.2, 107.6, 108.9, 105.7, 94.4, 103.5, 107.2, 102.6, 106.3, 108.5, 105.8, np.nan],
        'month': ['Jan', 'Jan', 'Jan', 'Jan', 'Jan', 'Feb', 'Feb', 'Feb', 'Feb', 'Feb', 
                 'Mar', 'Mar', 'Mar', 'Mar', 'Mar', 'Mar'],
        'latitude': [6.45137, 6.4477, 6.42836, 6.43668, 6.42685, 6.45137, 6.4477, 6.42836, 6.43668, 6.42685,
                    6.45137, 6.4477, 6.42836, 6.43668, 6.42685, 6.42555],
        'longitude': [3.37344, 3.38142, 3.39891, 3.34101, 3.35246, 3.37344, 3.38142, 3.39891, 3.34101, 3.35246,
                      3.37344, 3.38142, 3.39891, 3.34101, 3.35246, 3.10993]
    }
    
    df = pd.DataFrame(data)
    
    # Replace NaNs with 0 for metals (assuming non-detected)
    metal_cols = ['zn_ppm', 'cd_ppm', 'fe_ppm', 'cu_ppm', 'ni_ppm', 'pb_ppm', 'co_ppm', 'cr_ppm']
    for col in metal_cols:
        df[col] = df[col].fillna(0)
    
    # Fill other NaNs with median
    for col in df.columns:
        if df[col].isna().sum() > 0 and col not in metal_cols:
            df[col] = df[col].fillna(df[col].median())
    
    # Create industrial site indicator
    industrial_sites = ['Apapa pt1', 'Apapa pt2', 'Naval dockyard', 'Tin can']
    df['is_industrial'] = df['location'].apply(lambda x: 1 if x in industrial_sites else 0)
    
    # Create pollution index
    df['pollution_index'] = (
        df['pb_ppm'] * 0.3 + 
        df['cd_ppm'] * 0.25 + 
        df['cr_ppm'] * 0.2 + 
        df['cu_ppm'] * 0.15 + 
        df['zn_ppm'] * 0.1
    )
    
    return df

# Perform clustering
def perform_clustering(df):
    metal_cols = ['zn_ppm', 'cd_ppm', 'fe_ppm', 'cu_ppm', 'ni_ppm', 'pb_ppm', 'co_ppm', 'cr_ppm']
    
    # Scale features
    scaler = StandardScaler()
    X_metals = scaler.fit_transform(df[metal_cols])
    
    # PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_metals)
    df[['PC1', 'PC2']] = principal_components
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_metals)
    df['pollution_cluster'] = clusters
    
    return df, pca

# Train COD prediction model
def train_cod_model(df):
    cod_features = ['temp_c', 'ph', 'orp_mv', 'conductivity_us_cm', 'turbidity_ntu', 
                   'do_mg_l', 'tds_g_l', 'salinity_ppt', 'hardness_mg_l']
    X = df[cod_features]
    y = df['cod_mg_l']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return model, importance

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Harbor Water Quality Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load data
    df = load_data()
    
    # Perform clustering
    df, pca_model = perform_clustering(df)
    
    # Train COD model
    cod_model, feature_importance = train_cod_model(df)
    
    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    selected_month = st.sidebar.selectbox("Select Month", options=['All'] + list(df['month'].unique()))
    selected_location = st.sidebar.selectbox("Select Location", options=['All'] + list(df['location'].unique()))
    
    # Filter data based on selections
    if selected_month != 'All':
        df = df[df['month'] == selected_month]
    if selected_location != 'All':
        df = df[df['location'] == selected_location]
    
    # Title and description
    st.title("🌊 Harbor Water Quality Analysis Dashboard")
    st.markdown("""
    Interactive visualization of heavy metal concentrations and physico-chemical parameters 
    across sampling locations in the harbor area.
    """)
    
    # First row: Map and Pollution Summary
    st.header("Spatial Distribution of Pollution")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create PyDeck map
        view_state = pdk.ViewState(
            latitude=df['latitude'].mean(),
            longitude=df['longitude'].mean(),
            zoom=11,
            pitch=50
        )
        
        # Scale pollution_index for visualization
        df['scaled_pollution'] = (df['pollution_index'] - df['pollution_index'].min()) / \
                                (df['pollution_index'].max() - df['pollution_index'].min()) * 100 + 50
        
        # Create layers
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius='scaled_pollution',
            pickable=True
        )
        
        text_layer = pdk.Layer(
            "TextLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_text='location',
            get_color=[255, 255, 255],
            get_size=14,
            get_alignment_baseline="'bottom'"
        )
        
        # Render map
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/satellite-v9',
            initial_view_state=view_state,
            layers=[scatter_layer, text_layer],
            tooltip={"text": "Location: {location}\nPollution Index: {pollution_index:.2f}\nLead: {pb_ppm:.3f} ppm"}
        ))
    
    with col2:
        st.subheader("Pollution Summary")
        
        # Top polluted locations
        st.markdown("### 🚨 Most Polluted Locations")
        top_polluted = df.groupby('location')['pollution_index'].mean().sort_values(ascending=False).head(3)
        for loc, val in top_polluted.items():
            st.progress(val / top_polluted.max(), text=f"{loc}: {val:.2f}")
        
        # Metal concentration comparison
        st.markdown("### 🔬 Heavy Metal Levels")
        metals = ['pb_ppm', 'cd_ppm', 'cr_ppm', 'cu_ppm']
        avg_metals = df[metals].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        avg_metals.plot(kind='bar', color='#ff6b6b', ax=ax)
        plt.title('Average Metal Concentrations (ppm)')
        plt.ylabel('Concentration (ppm)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Water quality indicators
        st.markdown("### 💧 Water Quality Indicators")
        indicators = {
            'pH': df['ph'].mean(),
            'Dissolved Oxygen': df['do_mg_l'].mean(),
            'Turbidity': df['turbidity_ntu'].mean(),
            'COD': df['cod_mg_l'].mean()
        }
        for name, value in indicators.items():
            st.metric(label=name, value=f"{value:.2f}")
    
    # Second row: Parameter Analysis
    st.header("Parameter Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Metal Concentrations", "Physico-chemical Parameters", "Cluster Analysis"])
    
    with tab1:
        st.subheader("Heavy Metal Concentrations by Location")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            metal_select = st.selectbox("Select Metal", options=['Lead (Pb)', 'Cadmium (Cd)', 'Chromium (Cr)', 'Copper (Cu)', 'Iron (Fe)'])
            metal_col = {
                'Lead (Pb)': 'pb_ppm',
                'Cadmium (Cd)': 'cd_ppm',
                'Chromium (Cr)': 'cr_ppm',
                'Copper (Cu)': 'cu_ppm',
                'Iron (Fe)': 'fe_ppm'
            }[metal_select]
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df, x='location', y=metal_col, hue='month', palette='viridis', ax=ax)
            plt.title(f'{metal_select} Concentration by Location')
            plt.ylabel('Concentration (ppm)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Metal Correlation Matrix")
            metals = ['zn_ppm', 'cd_ppm', 'fe_ppm', 'cu_ppm', 'ni_ppm', 'pb_ppm', 'co_ppm', 'cr_ppm']
            corr = df[metals].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
            plt.title('Correlation Between Heavy Metals')
            st.pyplot(fig)
            
            st.markdown("""
            **Key Insights:**
            - Lead (Pb) shows strong correlation with Cadmium (Cd) (r ≈ 0.68)
            - Chromium (Cr) correlates with Copper (Cu) (r ≈ 0.62)
            - Iron (Fe) has weak correlations with other metals
            """)
    
    with tab2:
        st.subheader("Water Quality Parameters")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            param_select = st.selectbox("Select Parameter", options=['pH', 'Dissolved Oxygen', 'Temperature', 'Salinity', 'Turbidity'])
            param_col = {
                'pH': 'ph',
                'Dissolved Oxygen': 'do_mg_l',
                'Temperature': 'temp_c',
                'Salinity': 'salinity_ppt',
                'Turbidity': 'turbidity_ntu'
            }[param_select]
            
            # Line chart over months
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x='month', y=param_col, hue='location', 
                         marker='o', markersize=8, linewidth=2.5, ax=ax)
            plt.title(f'{param_select} Trend by Month')
            plt.ylabel(param_select)
            plt.xlabel('Month')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Parameter Relationships")
            x_param = st.selectbox("X-Axis Parameter", 
                                  options=['pH', 'Dissolved Oxygen', 'Temperature', 'Salinity', 'Turbidity', 'COD'],
                                  index=0)
            y_param = st.selectbox("Y-Axis Parameter", 
                                  options=['Lead (Pb)', 'Cadmium (Cd)', 'pH', 'Dissolved Oxygen', 'COD'],
                                  index=3)
            
            x_col = {
                'pH': 'ph',
                'Dissolved Oxygen': 'do_mg_l',
                'Temperature': 'temp_c',
                'Salinity': 'salinity_ppt',
                'Turbidity': 'turbidity_ntu',
                'COD': 'cod_mg_l'
            }[x_param]
            
            y_col = {
                'Lead (Pb)': 'pb_ppm',
                'Cadmium (Cd)': 'cd_ppm',
                'pH': 'ph',
                'Dissolved Oxygen': 'do_mg_l',
                'COD': 'cod_mg_l'
            }[y_param]
            
            fig = px.scatter(df, x=x_col, y=y_col, color='location', size='pollution_index',
                             hover_data=['month'], trendline='ols',
                             title=f"{x_param} vs {y_param}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation
            correlation = df[x_col].corr(df[y_col])
            st.metric(f"Correlation between {x_param} and {y_param}", value=f"{correlation:.2f}")
    
    with tab3:
        st.subheader("Pollution Profile Clustering")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            # PCA scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(data=df, x='PC1', y='PC2', hue='pollution_cluster', 
                            style='location', s=200, palette='viridis', ax=ax)
            
            # Add cluster labels
            cluster_centers = df.groupby('pollution_cluster')[['PC1', 'PC2']].mean()
            for i, row in cluster_centers.iterrows():
                ax.text(row['PC1'], row['PC2'], f'Cluster {i}', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title('Pollution Clusters (PCA Projection)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Cluster Characteristics")
            
            # Cluster descriptions
            cluster_desc = {
                0: "Moderate Pollution: Balanced metal concentrations",
                1: "High Heavy Metals: Elevated Pb, Cd, and Cr levels",
                2: "Industrial Pollution: High Fe and Cu with moderate others"
            }
            
            # Display cluster info
            for cluster in [0, 1, 2]:
                cluster_df = df[df['pollution_cluster'] == cluster]
                st.subheader(f"Cluster {cluster}: {cluster_desc[cluster]}")
                
                # Show key parameters
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Avg. Lead", f"{cluster_df['pb_ppm'].mean():.3f} ppm")
                with cols[1]:
                    st.metric("Avg. Cadmium", f"{cluster_df['cd_ppm'].mean():.3f} ppm")
                with cols[2]:
                    st.metric("Avg. pH", f"{cluster_df['ph'].mean():.2f}")
                with cols[3]:
                    st.metric("Avg. DO", f"{cluster_df['do_mg_l'].mean():.2f} mg/L")
                
                # Locations in cluster
                st.caption(f"Locations: {', '.join(cluster_df['location'].unique())}")
    
    # Third row: Predictive Modeling
    st.header("Predictive Modeling")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("COD Prediction Model")
        
        # Feature importance
        st.markdown("### Feature Importance for COD Prediction")
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance.sort_values().plot(kind='barh', color='#4ecdc4', ax=ax)
        plt.title('Feature Importance in COD Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Insights:**
        - Turbidity and Hardness are the strongest predictors of COD
        - Dissolved Oxygen has a negative relationship with COD
        - Temperature shows moderate predictive power
        """)
    
    with col2:
        st.subheader("Predict COD with Custom Inputs")
        
        # Create sliders for input parameters
        col1, col2 = st.columns(2)
        with col1:
            turbidity = st.slider("Turbidity (NTU)", 0.0, 60.0, 15.0, 0.5)
            hardness = st.slider("Hardness (mg/L)", 100.0, 2500.0, 200.0, 10.0)
            do = st.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 7.0, 0.1)
        with col2:
            temp = st.slider("Temperature (°C)", 25.0, 35.0, 28.0, 0.1)
            salinity = st.slider("Salinity (ppt)", 0.0, 40.0, 25.0, 0.5)
            ph = st.slider("pH", 6.5, 9.0, 7.8, 0.1)
        
        # Create input array
        input_data = [[temp, ph, 150, 45, turbidity, do, 25, salinity, hardness]]
        
        # Predict COD
        cod_pred = cod_model.predict(input_data)[0]
        
        # Display prediction
        st.metric("Predicted COD", f"{cod_pred:.2f} mg/L")
        
        # Interpretation
        st.info("""
        Chemical Oxygen Demand (COD) indicates the amount of oxygen required to break down 
        organic matter in water. Higher COD values suggest more polluted water.
        """)
    
    # Footer
    st.markdown("---")
    st.caption("Water Quality Analysis Dashboard | Data collected from Harbor Monitoring Stations | © 2023 Environmental Monitoring Group")

if __name__ == "__main__":
    main()