# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Harbor Sediment Analysis")
st.title("🌊 Harbor Sediment Analysis Dashboard")
st.subheader("Comprehensive Analysis of Heavy Metals and Physiochemical Parameters")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])

# Load data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            return None, None
    else:
        # Create sample data if no file is uploaded
        data = {
            'LOCATION': ['Naval Dockyard', 'Tin Can', 'Snake Island', 'Naval Dockyard', 
                         'Tin Can', 'Snake Island', 'Naval Dockyard', 'Tin Can', 'Snake Island'],
            'Zn (mg/kg)': [17.6849, 689.6424, 58.3347, 40.775, 481.59, 2.01, 5.53, 716.755, 190.92],
            'Cd (mg/kg)': [0, 4.1097, 0, 0, 2.845, 0, np.nan, np.nan, np.nan],
            'Fe (mg/kg)': [768.7284, 8583.1664, 568.4728, 2403.56, 12583.77, 367.535, 978.5, 10508.95, 8196.505],
            'Cu (mg/kg)': [7.8719, 28.6727, 14.438, 4.5, 34.065, 3.52, 3.48, 35.755, 17.56],
            'Pb (mg/kg)': [np.nan, 23.3699, 5.2768, 0, 94.74, 3.695, 11.61, 102.325, 55.625],
            'Co (mg/kg)': [8.8373, 25.6488, 16.8649, 24.895, 25.515, 2.245, 3.91, 81.67, 38.305],
            'K (mg/kg)': [232.275, 1908.23, 2128.87, 276.05, 1865.435, 114.25, 232.275, 1908.23, 1148.87],
            'Cr (mg/kg)': [13.6438, 28.8376, 44.3589, 19.615, 46.98, 1.3, 2.62, 49.555, 20.29],
            'Mg (mg/kg)': [409.3256, 2756.6264, 2965.635, 706.955, 3485.64, 370.105, 673.905, 3846.775, 2068.95],
            'Mn (mg/kg)': [37.3478, 96.7278, 105.3769, 68.965, 107.78, 7.82, 24.91, 120.8, 97.51],
            'Na (mg/kg)': [147.3879, 1578.8276, 900.3625, 2175.51, 2063.95, 421.575, 618.89, 5479.025, 4738.835],
            'Conductivity µs/cm': [0.16, 0.12, 0.55, 0.39, 0.31, 1.25, 0.23, 0.65, 0.37],
            'pH': [6.3, 6.35, 6.05, 7.3, 6.8, 6.3, 6.7, 6.45, 7.1],
            '%Total Organic Carbon': [1.31, 0.76, 7.76, 2.07, 0.86, 3.24, 3.72, 1.14, 5.38],
            'Cation Exchange Capacity meq/g': [10, 15, 52.5, 25.7, 22.8, 37.3, 32.5, 28.7, 24.3],
            '%Total Organic Matter': [2.27, 1.31, 13.43, 3.58, 1.49, 5.61, 6.45, 1.97, 9.31],
            'Months': ['Jan', 'Feb', 'Mar', 'Jan', 'Feb', 'Mar', 'Jan', 'Feb', 'Mar'],
            'Latitude': [6.42836, 6.43668, 6.42685, 6.42836, 6.43668, 6.42685, 6.42836, 6.43668, 6.42685],
            'Longitude': [3.39891, 3.34101, 3.35246, 3.39891, 3.34101, 3.35246, 3.39891, 3.34101, 3.35246]
        }
        df = pd.DataFrame(data)
        st.sidebar.info("Using sample data")
    
    # Clean column names by removing extra spaces
    df.columns = [re.sub(r'\s+', ' ', col.strip()) for col in df.columns]
    
    # Replace non-numeric values
    df.replace(['ND', 'nd', 'n.d.', 'N/A', '-'], np.nan, inplace=True)
    
    # Convert numeric columns
    non_numeric_cols = ['LOCATION', 'Months']
    numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Calculate total heavy metal concentration
    metal_cols = [col for col in df.columns if '(mg/kg)' in col]
    if metal_cols:
        df['Total Heavy Metals (mg/kg)'] = df[metal_cols].sum(axis=1, skipna=True)
    else:
        st.error("No metal concentration columns found in the data!")
        df['Total Heavy Metals (mg/kg)'] = 0
    
    return df, metal_cols

if uploaded_file is not None:
    df, metal_cols = load_data(uploaded_file)
else:
    df, metal_cols = load_data(None)

# Only proceed if data loaded successfully
if df is not None:
    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    analysis_type = st.sidebar.selectbox("Analysis Type", [
        "Data Overview", 
        "Physiochemical Relationships",
        "Pollution Mapping",
        "Heavy Metal Analysis",
        "Clustering Characteristics"
    ])

    # Data Overview Section
    if analysis_type == "Data Overview":
        st.header("Data Overview")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Basic Statistics")
            st.dataframe(df.describe())
            
            st.subheader("Missing Values")
            missing_df = df.isna().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            st.dataframe(missing_df)
            
        with col2:
            st.subheader("Data Preview")
            st.dataframe(df)
            
            st.subheader("Parameter Distribution")
            param = st.selectbox("Select Parameter", df.select_dtypes(include=np.number).columns)
            fig = px.histogram(df, x=param, color='LOCATION', marginal='box')
            st.plotly_chart(fig, use_container_width=True)

    # Physiochemical Relationships
    elif analysis_type == "Physiochemical Relationships":
        st.header("Physiochemical Parameter Relationships")
        
        st.subheader("Parameter Correlation Matrix")
        physio_cols = ['pH', 'Conductivity µs/cm', '%Total Organic Carbon', 
                      'Cation Exchange Capacity meq/g', '%Total Organic Matter']
        
        # Filter only existing columns
        existing_physio = [col for col in physio_cols if col in df.columns]
        existing_metal = [col for col in metal_cols if col in df.columns]
        
        if existing_physio and existing_metal:
            corr_matrix = df[existing_physio + existing_metal].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for correlation matrix")
        
        st.subheader("Relationship Explorer")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-Axis Parameter", existing_physio, index=0)
        with col2:
            y_axis = st.selectbox("Y-Axis Parameter", existing_metal, index=0)
        with col3:
            color_by = st.selectbox("Color By", ['LOCATION', 'Months'], index=0)
        
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, 
                         hover_name='LOCATION', size='Total Heavy Metals (mg/kg)',
                         trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("3D Parameter Relationships")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_3d = st.selectbox("X-Axis", existing_physio, index=0)
        with col2:
            y_3d = st.selectbox("Y-Axis", existing_physio, index=1)
        with col3:
            z_3d = st.selectbox("Z-Axis", existing_metal, index=0)
        
        fig = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color='LOCATION',
                            hover_name='Months', size='Total Heavy Metals (mg/kg)')
        st.plotly_chart(fig, use_container_width=True)

    # Pollution Mapping
    elif analysis_type == "Pollution Mapping":
        st.header("Pollution Concentration Mapping")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Map Settings")
            map_options = ['Total Heavy Metals (mg/kg)'] + metal_cols + ['Custom Score']
            map_metric = st.selectbox("Visualization Metric", map_options)
            
            if map_metric == 'Custom Score':
                st.info("Create custom pollution index:")
                weight_zn = st.slider("Zinc Weight", 0.0, 2.0, 1.0)
                weight_cd = st.slider("Cadmium Weight", 0.0, 2.0, 1.5)
                weight_pb = st.slider("Lead Weight", 0.0, 2.0, 1.2)
                weight_org = st.slider("Organic Matter Weight", 0.0, 2.0, 0.8)
                
                zn_col = next((col for col in metal_cols if 'Zn' in col), None)
                cd_col = next((col for col in metal_cols if 'Cd' in col), None)
                pb_col = next((col for col in metal_cols if 'Pb' in col), None)
                
                df['Custom Score'] = 0
                if zn_col: 
                    df['Custom Score'] += weight_zn * df[zn_col].fillna(0)
                if cd_col: 
                    df['Custom Score'] += weight_cd * df[cd_col].fillna(0)
                if pb_col: 
                    df['Custom Score'] += weight_pb * df[pb_col].fillna(0)
                if '%Total Organic Matter' in df.columns:
                    df['Custom Score'] += weight_org * df['%Total Organic Matter'].fillna(0)
                
                map_metric = 'Custom Score'
            
            size_multiplier = st.slider("Marker Size Multiplier", 0.1, 5.0, 1.0)
            if map_metric in df.columns:
                st.metric("Average Pollution Level", 
                         f"{df[map_metric].mean():.1f}",
                         delta=f"Min: {df[map_metric].min():.1f}, Max: {df[map_metric].max():.1f}")
        
        with col2:
            st.subheader("Pollution Map")
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df_map = df[['Latitude', 'Longitude', 'LOCATION', map_metric]].copy()
                df_map['size'] = df_map[map_metric].abs() * size_multiplier
                
                fig = px.scatter_mapbox(
                    df_map, 
                    lat="Latitude", 
                    lon="Longitude", 
                    color=map_metric,
                    size='size',
                    hover_name="LOCATION", 
                    hover_data=[map_metric],
                    color_continuous_scale=px.colors.sequential.Viridis,
                    zoom=10
                )
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Latitude/Longitude columns not found in data")
        
        st.subheader("Temporal Pollution Trends")
        if 'Months' in df.columns and map_metric in df.columns:
            trend_df = df.groupby(['Months', 'LOCATION'])[map_metric].mean().reset_index()
            fig = px.line(trend_df, x='Months', y=map_metric, color='LOCATION', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Months column not found in data")

    # Heavy Metal Analysis
    elif analysis_type == "Heavy Metal Analysis":
        st.header("Heavy Metal Concentration Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Metal Comparison")
            selected_metals = st.multiselect("Select Metals", metal_cols, default=metal_cols[:3])
            
            st.subheader("Quality Guidelines")
            st.info("""
            Common sediment quality guidelines (mg/kg):
            - Zinc (Zn): 200-400
            - Cadmium (Cd): 0.6-1.2
            - Lead (Pb): 40-60
            - Copper (Cu): 25-50
            """)
            
            st.subheader("Location Comparison")
            if 'LOCATION' in df.columns:
                location = st.selectbox("Select Location", df['LOCATION'].unique())
                loc_data = df[df['LOCATION'] == location][metal_cols].mean().reset_index()
                loc_data.columns = ['Metal', 'Concentration']
                fig = px.bar(loc_data, x='Metal', y='Concentration', title=f"Metal Concentrations at {location}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Metal Distribution by Location")
            if selected_metals and 'LOCATION' in df.columns:
                fig = px.box(df, x='LOCATION', y=selected_metals, 
                             points="all", hover_data=['Months'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Metal Correlation Network")
            if metal_cols:
                metal_corr = df[metal_cols].corr().abs()
                fig = px.imshow(metal_corr, text_auto=".2f", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

    # Clustering Characteristics
    elif analysis_type == "Clustering Characteristics":
        st.header("Sediment Sample Clustering")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Clustering Settings")
            n_clusters = st.slider("Number of Clusters", 2, 6, 3)
            numeric_features = df.select_dtypes(include=np.number).columns.tolist()
            features = st.multiselect("Features for Clustering", 
                                     numeric_features,
                                     default=['Total Heavy Metals (mg/kg)', 
                                              '%Total Organic Matter' if '%Total Organic Matter' in numeric_features else numeric_features[0], 
                                              'pH' if 'pH' in numeric_features else numeric_features[1],
                                              'Zn (mg/kg)' if 'Zn (mg/kg)' in numeric_features else numeric_features[2]])
            
            if st.button("Run Clustering"):
                # Prepare data
                cluster_df = df[features].dropna()
                if not cluster_df.empty:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_df)
                    
                    # Apply KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    # Add to dataframe
                    df_cluster = df.copy().loc[cluster_df.index]
                    df_cluster['Cluster'] = clusters
                    
                    # PCA for visualization
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    df_cluster['PCA1'] = pca_result[:, 0]
                    df_cluster['PCA2'] = pca_result[:, 1]
                    
                    st.session_state.cluster_results = df_cluster
                    st.session_state.features = features
                else:
                    st.error("No valid data for clustering after removing missing values")
        
        with col2:
            st.subheader("Cluster Visualization")
            
            if 'cluster_results' in st.session_state:
                df_cluster = st.session_state.cluster_results
                
                fig = px.scatter(df_cluster, x='PCA1', y='PCA2', color='Cluster',
                                hover_name='LOCATION', hover_data=st.session_state.features,
                                title="PCA Projection of Clusters")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Cluster Characteristics")
                cluster_means = df_cluster.groupby('Cluster')[st.session_state.features].mean()
                st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
                
                st.subheader("Cluster Distribution")
                fig = px.pie(df_cluster, names='Cluster', 
                             title='Cluster Membership Distribution')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Click 'Run Clustering' to see results")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Data Sources**:  
    Harbor Sediment Samples  
    *Parameters: Heavy metals and physiochemical properties*  
    """)
else:
    st.error("Failed to load data. Please check your file format.")