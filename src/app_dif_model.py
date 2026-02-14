import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import sys

# Import custom NP classes
try:
    from numpy_models import (
        StandardScalerNumPy, 
        SimpleImputerNumPy, 
        DecisionTreeRegressorNumPy, 
        RandomForestRegressorNumPy
    )
except ImportError:
    pass

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Model Performance Comparison", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("📊 ML Model Performance Comparison")
st.markdown("Analyse comparative des modèles: CatBoost, Random Forest (sklearn), et Random Forest (Pure NumPy)")
st.markdown("---")

# Load and preprocess data (matching notebook exactly)
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data exactly as in the notebook"""
    # Update these paths to match your data location
    notebook_dir = os.getcwd()
    project_root = os.path.dirname(notebook_dir)
    csv_path = os.path.join(project_root, "Data", "xp.csv")
    csv_path1 = os.path.join(project_root, "Data", "metro.csv")
    
    df = pd.read_csv(csv_path)
    
    metro_df = pd.read_csv(csv_path1)
    metro_df.columns = metro_df.columns.str.strip()
    
    # Date processing
    df['date_mutation'] = pd.to_datetime(df['date_mutation'])
    df['year'] = df['date_mutation'].dt.year
    df['month'] = df['date_mutation'].dt.month
    df['day_of_week'] = df['date_mutation'].dt.dayofweek
    df['days_since_start'] = (df['date_mutation'] - df['date_mutation'].min()).dt.days
    
    # Kmeans
    geo_features = df[['longitude', 'latitude']].copy()
    n_clusters = 20
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['geo_cluster'] = km.fit_predict(geo_features)
    
    # Distance to center
    center_lon, center_lat = 2.3384444444444446, 48.86152777777778
    df['dist_center'] = np.sqrt(
        (df['longitude'] - center_lon)**2 + 
        (df['latitude'] - center_lat)**2
    ) * 111
    
    # Balltree metro
    metro_coords = np.radians(metro_df[['Latitude', 'Longitude']].values)
    tree = BallTree(metro_coords, metric='haversine')
    
    appart_coords = np.radians(df[['latitude', 'longitude']].values)
    
    distances, indices = tree.query(appart_coords, k=1)
    df['nearest_metro_dist_km'] = distances.flatten() * 6371
    df['nearest_metro_station'] = metro_df.iloc[indices.flatten()]['Libelle station'].values
    df['nearest_metro_line'] = metro_df.iloc[indices.flatten()]['Libelle Line'].values
    
    indices_300m = tree.query_radius(appart_coords, r=0.3/6371)
    indices_500m = tree.query_radius(appart_coords, r=0.5/6371)
    
    df['metro_count_300m'] = [len(idx) for idx in indices_300m]
    df['metro_count_500m'] = [len(idx) for idx in indices_500m]
    df['very_close_to_metro'] = (df['nearest_metro_dist_km'] < 0.1).astype(int)
    
    # Appartment features
    df['surface_per_piece'] = df['surface_reelle_bati'] / df['nombre_pieces_principales'].replace(0, 1)
    df['is_studio'] = (df['nombre_pieces_principales'] == 1).astype(int)
    df['is_large'] = (df['nombre_pieces_principales'] >= 4).astype(int)
    df['surface_category'] = pd.cut(df['surface_reelle_bati'], 
                                     bins=[9, 40, 80, float('inf')],
                                     labels=['small', 'medium', 'large'])
    
    # Train/Test Split
    df_sorted = df.sort_values('date_mutation').reset_index(drop=True)
    split_index = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()
    
    # Metro features
    station_stats_train = train_df.groupby('nearest_metro_station').agg({
        'surface_reelle_bati': ['mean', 'std', 'median', 'count'],
        'nombre_pieces_principales': ['mean', 'std', 'median'],
    }).round(2)
    
    station_stats_train.columns = [
        'station_avg_surface',
        'station_surface_std',
        'station_median_surface',
        'station_tx_count',
        'station_avg_rooms',
        'station_rooms_std',
        'station_median_rooms'
    ]
    
    station_stats_train['station_surface_range'] = (
        station_stats_train['station_avg_surface'] + 2*station_stats_train['station_surface_std'] -
        (station_stats_train['station_avg_surface'] - 2*station_stats_train['station_surface_std'])
    )
    
    station_stats_train['station_surface_std'] = station_stats_train['station_surface_std'].replace(0, 1)
    station_stats_train['station_rooms_std'] = station_stats_train['station_rooms_std'].replace(0, 1)
    
    # Merge station features
    train_df = train_df.merge(
        station_stats_train,
        left_on='nearest_metro_station',
        right_index=True,
        how='left'
    )
    
    test_df = test_df.merge(
        station_stats_train,
        left_on='nearest_metro_station',
        right_index=True,
        how='left'
    )
    
    # Fill missing values in station features
    station_cols = [c for c in train_df.columns if c.startswith('station_')]
    for col in station_cols:
        train_mean = train_df[col].mean()
        train_df[col] = train_df[col].fillna(train_mean)
        test_df[col] = test_df[col].fillna(train_mean)
    
    # Appartment/Metro features
    for dataset in [train_df, test_df]:
        dataset['surface_vs_station_avg'] = (
            (dataset['surface_reelle_bati'] - dataset['station_avg_surface']) / 
            dataset['station_surface_std']
        )
        
        dataset['rooms_vs_station_avg'] = (
            (dataset['nombre_pieces_principales'] - dataset['station_avg_rooms']) / 
            dataset['station_rooms_std']
        )
        
        dataset['larger_than_station_median'] = (
            dataset['surface_reelle_bati'] > dataset['station_median_surface']
        ).astype(int)
    
    # Groupby arr, time
    train_agg = train_df.groupby(['code_postal', 'year', 'month']).agg({
        'price_per_sqrtm': 'median',
        'valeur_fonciere': 'median',
        'nombre_pieces_principales': 'median',
        'surface_reelle_bati': 'median',
        'surface_per_piece': 'median',
        'longitude': 'mean',
        'latitude': 'mean',
        'days_since_start': 'median',
        'geo_cluster': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'dist_center': 'median',
        'nearest_metro_dist_km': 'median',
        'station_tx_count': 'median',
        'station_avg_surface': 'median',
        'station_surface_std': 'median',
        'station_median_surface': 'median',
        'station_avg_rooms': 'median',
        'station_rooms_std': 'median',
        'station_median_rooms': 'median',
        'station_surface_range': 'median',
        'surface_vs_station_avg': 'median',
        'rooms_vs_station_avg': 'median',
        'larger_than_station_median': 'median',
        'metro_count_300m': 'median',
        'metro_count_500m': 'median',
        'very_close_to_metro': 'median',
        'is_studio': 'median',
        'is_large': 'median',
        'date_mutation': 'first'
    }).reset_index()
    
    test_agg = test_df.groupby(['code_postal', 'year', 'month']).agg({
        'price_per_sqrtm': 'median',
        'valeur_fonciere': 'median',
        'nombre_pieces_principales': 'median',
        'surface_reelle_bati': 'median',
        'surface_per_piece': 'median',
        'longitude': 'mean',
        'latitude': 'mean',
        'days_since_start': 'median',
        'geo_cluster': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'dist_center': 'median',
        'nearest_metro_dist_km': 'median',
        'station_tx_count': 'median',
        'station_avg_surface': 'median',
        'station_surface_std': 'median',
        'station_median_surface': 'median',
        'station_avg_rooms': 'median',
        'station_rooms_std': 'median',
        'station_median_rooms': 'median',
        'station_surface_range': 'median',
        'surface_vs_station_avg': 'median',
        'rooms_vs_station_avg': 'median',
        'larger_than_station_median': 'median',
        'metro_count_300m': 'median',
        'metro_count_500m': 'median',
        'very_close_to_metro': 'median',
        'is_studio': 'median',
        'is_large': 'median',
        'date_mutation': 'first'
    }).reset_index()
    
    # Transaction features
    train_postal_counts = train_df.groupby('code_postal').size().to_dict()
    
    # Apply to both train and test aggregated data
    train_agg['historical_tx_count'] = train_agg['code_postal'].map(train_postal_counts)
    test_agg['historical_tx_count'] = test_agg['code_postal'].map(train_postal_counts).fillna(0)
    
    # Feature Rdf
    train_tx_counts = train_df.groupby(['code_postal', 'year', 'month']).size().reset_index(name='transaction_count')
    test_tx_counts = test_df.groupby(['code_postal', 'year', 'month']).size().reset_index(name='transaction_count')
    
    train_agg = train_agg.merge(train_tx_counts, on=['code_postal', 'year', 'month'], how='left')
    test_agg = test_agg.merge(test_tx_counts, on=['code_postal', 'year', 'month'], how='left')
    
    # Fill missing
    train_agg['transaction_count'] = train_agg['transaction_count'].fillna(0)
    test_agg['transaction_count'] = test_agg['transaction_count'].fillna(0)
    
    # Transaction activity ONLY on train
    historical_activity_train = train_df.groupby('code_postal').size().reset_index(name='total_transactions_train')
    
    # Merge train and test
    train_agg = train_agg.merge(historical_activity_train, on='code_postal', how='left')
    test_agg = test_agg.merge(historical_activity_train, on='code_postal', how='left')
    
    # Arr for test
    overall_avg_transactions = historical_activity_train['total_transactions_train'].mean()
    train_agg['total_transactions_train'] = train_agg['total_transactions_train'].fillna(overall_avg_transactions)
    test_agg['total_transactions_train'] = test_agg['total_transactions_train'].fillna(overall_avg_transactions)
    
    # Transaction ratio
    train_agg['market_activity_ratio'] = train_agg['transaction_count'] / train_agg['total_transactions_train'].replace(0, 1)
    test_agg['market_activity_ratio'] = test_agg['transaction_count'] / test_agg['total_transactions_train'].replace(0, 1)
    
    return df, train_df, test_df, train_agg, test_agg, km

# Load models
@st.cache_resource
def load_models():
    """Load all available models: CatBoost, Random Forest, and NumPy"""
    models = {}
    
    # Load model
    try:
        with open("model_cat.pkl", "rb") as f:
            models['catboost'] = {
                'model': pickle.load(f),
                'name': 'CatBoost Regressor',
                'available': True
            }
        with open("preprocessor_cat.pkl", "rb") as f:
            models['catboost']['preprocessor'] = pickle.load(f)
    except FileNotFoundError:
        models['catboost'] = {'available': False, 'name': 'CatBoost Regressor'}
    
    try:
        with open("model_random_forest.pkl", "rb") as f:
            models['random_forest'] = {
                'model': pickle.load(f),
                'name': 'Random Forest Regressor (sklearn)',
                'available': True
            }
        with open("preprocessor_random_forest.pkl", "rb") as f:
            models['random_forest']['preprocessor'] = pickle.load(f)
    except FileNotFoundError:
        models['random_forest'] = {'available': False, 'name': 'Random Forest Regressor (sklearn)'}
    
    try:
        with open("model_numpy.pkl", "rb") as f:
            models['numpy'] = {
                'model': pickle.load(f),
                'name': 'Random Forest Regressor (Pure NumPy)',
                'available': True
            }
        with open("preprocessor_numpy.pkl", "rb") as f:
            models['numpy']['preprocessor'] = pickle.load(f)
    except FileNotFoundError:
        models['numpy'] = {'available': False, 'name': 'Random Forest Regressor (Pure NumPy)'}
    
    return models

# Main app
try:
    df, train_df, test_df, train_agg, test_agg, kmeans = load_and_preprocess_data()
    models = load_models()
    
    # Model selection
    st.sidebar.header("🎯 Model Selection")
    
    # Check available
    available_models = {k: v for k, v in models.items() if v['available']}
    
    if not available_models:
        st.error("❌ No models found! Please ensure model files are in the 'src' folder.")
        st.stop()
    
    # Model selector
    model_options = {k: v['name'] for k, v in available_models.items()}
    selected_model_key = st.sidebar.selectbox(
        "Choose Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    # Get selected model
    selected_model_info = models[selected_model_key]
    model = selected_model_info['model']
    preprocessor = selected_model_info['preprocessor']
    model_name = selected_model_info['name']
    
    # Model info
    st.sidebar.success(f"✅ **Selected Model:** {model_name}")
    
    # Data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Information")
    st.sidebar.info(f"""
    **Total observations:** {len(df):,}  
    **Training transactions:** {len(train_df):,}  
    **Test transactions:** {len(test_df):,}  
    **Training aggregated:** {len(train_agg):,}  
    **Test aggregated:** {len(test_agg):,}
    
    **Date range:**  
    Train: {train_df['date_mutation'].min().strftime('%Y-%m-%d')} → {train_df['date_mutation'].max().strftime('%Y-%m-%d')}  
    Test: {test_df['date_mutation'].min().strftime('%Y-%m-%d')} → {test_df['date_mutation'].max().strftime('%Y-%m-%d')}
    """)
    
    # Features for pred
    if isinstance(preprocessor, dict):
        if 'drop_cols' in preprocessor:
            # CB - preproces is a dic with drop_cols and feature_columns
            drop_cols = preprocessor['drop_cols']
            feature_columns = preprocessor['feature_columns']
            
            # Features agg data
            X_train = train_agg.drop(columns=drop_cols, errors='ignore')[feature_columns]
            X_test = test_agg.drop(columns=drop_cols, errors='ignore')[feature_columns]
        else:
            # NP preproces is a dic with imputer, scaler, feature_names
            drop_cols = ['price_per_sqrtm', 'valeur_fonciere', 'date_mutation']
            X_train_raw = train_agg.drop(columns=drop_cols, errors='ignore')
            X_test_raw = test_agg.drop(columns=drop_cols, errors='ignore')
            
            # Num features
            numeric_features = preprocessor['feature_names']
            X_train_numeric = X_train_raw[numeric_features].values
            X_test_numeric = X_test_raw[numeric_features].values
            
            # Impute Scale
            imputer = preprocessor['imputer']
            scaler = preprocessor['scaler']
            
            X_train = scaler.transform(imputer.transform(X_train_numeric))
            X_test = scaler.transform(imputer.transform(X_test_numeric))
    else:
        # RDF preprocess is a ColumnTransformer
        # Need to drop cols
        drop_cols = ['price_per_sqrtm', 'valeur_fonciere', 'date_mutation']
        X_train = train_agg.drop(columns=drop_cols, errors='ignore')
        X_test = test_agg.drop(columns=drop_cols, errors='ignore')
        
        # Apply preproces
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
    
    y_train = train_agg['price_per_sqrtm']
    y_test = test_agg['price_per_sqrtm']
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Results df
    test_results = test_agg.copy()
    test_results['predicted_price_m2'] = y_test_pred
    test_results['actual_price_m2'] = y_test
    test_results['error'] = y_test_pred - y_test
    test_results['abs_error'] = np.abs(test_results['error'])
    test_results['pct_error'] = (test_results['error'] / y_test) * 100
    test_results['abs_pct_error'] = np.abs(test_results['pct_error'])
    test_results['year_month'] = test_results['year'].astype(str) + '-' + test_results['month'].astype(str).str.zfill(2)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🎯 Predictions Analysis", 
        "📍 Geographic Analysis",
        "📊 Feature Importance",
        "📋 Detailed Data"
    ])
    
    with tab1:
        st.subheader(f"Model Performance Metrics - {model_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test R² Score", f"{test_r2:.4f}")
            st.metric("Train R² Score", f"{train_r2:.4f}")
        
        with col2:
            st.metric("Test MAE", f"{test_mae:.2f} €/m²")
            st.metric("Train MAE", f"{train_mae:.2f} €/m²")
        
        with col3:
            st.metric("Test RMSE", f"{test_rmse:.2f} €/m²")
            st.metric("Train RMSE", f"{train_rmse:.2f} €/m²")
        
        with col4:
            st.metric("Test MAPE", f"{test_results['abs_pct_error'].mean():.2f}%")
            st.metric("Test Median Error", f"{test_results['abs_error'].median():.2f} €/m²")
        
        st.markdown("---")
        
        # Actual vs Predicted
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Actual vs Predicted Prices (Test Set)")
            
            # Sample for plotting
            sample_size = min(5000, len(test_results))
            plot_data = test_results.sample(n=sample_size, random_state=42) if len(test_results) > sample_size else test_results
            
            fig_scatter = px.scatter(
                plot_data,
                x='actual_price_m2',
                y='predicted_price_m2',
                color='abs_pct_error',
                color_continuous_scale='RdYlGn_r',
                labels={
                    'actual_price_m2': 'Prix Réel (€/m²)',
                    'predicted_price_m2': 'Prix Prédit (€/m²)',
                    'abs_pct_error': 'Erreur % Abs'
                },
                hover_data=['code_postal', 'surface_reelle_bati', 'nombre_pieces_principales']
            )
            
            # Perfect line
            min_val = min(plot_data['actual_price_m2'].min(), plot_data['predicted_price_m2'].min())
            max_val = max(plot_data['actual_price_m2'].max(), plot_data['predicted_price_m2'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.subheader("Error Distribution")
            
            fig_error_dist = px.histogram(
                test_results,
                x='error',
                nbins=50,
                labels={'error': 'Erreur de Prédiction (€/m²)', 'count': 'Fréquence'},
                color_discrete_sequence=['#4ECDC4']
            )
            fig_error_dist.add_vline(x=0, line_dash="dash", line_color="red", 
                                    annotation_text="Zero Error")
            fig_error_dist.update_layout(height=500)
            st.plotly_chart(fig_error_dist, use_container_width=True)
        
        # Residual plot
        st.subheader("Residual Plot")
        sample_residual = test_results.sample(n=min(3000, len(test_results)), random_state=42) if len(test_results) > 3000 else test_results
        fig_residual = px.scatter(
            sample_residual,
            x='predicted_price_m2',
            y='error',
            color='abs_pct_error',
            color_continuous_scale='RdYlGn_r',
            labels={
                'predicted_price_m2': 'Prix Prédit (€/m²)',
                'error': 'Erreur (€/m²)',
                'abs_pct_error': 'Erreur % Abs'
            }
        )
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residual.update_layout(height=400)
        st.plotly_chart(fig_residual, use_container_width=True)
    
    with tab2:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Best Predictions (Top 10)")
            best_predictions = test_results.nsmallest(10, 'abs_error')[[
                'code_postal', 'year_month', 'surface_reelle_bati', 'nombre_pieces_principales',
                'actual_price_m2', 'predicted_price_m2', 'abs_error', 'abs_pct_error'
            ]].round(2)
            best_predictions.columns = [
                'Code Postal', 'Période', 'Surface (m²)', 'Pièces', 
                'Prix Réel (€/m²)', 'Prix Prédit (€/m²)', 'Erreur Abs (€/m²)', 'Erreur %'
            ]
            st.dataframe(best_predictions, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Worst Predictions (Top 10)")
            worst_predictions = test_results.nlargest(10, 'abs_error')[[
                'code_postal', 'year_month', 'surface_reelle_bati', 'nombre_pieces_principales',
                'actual_price_m2', 'predicted_price_m2', 'abs_error', 'abs_pct_error'
            ]].round(2)
            worst_predictions.columns = [
                'Code Postal', 'Période', 'Surface (m²)', 'Pièces', 
                'Prix Réel (€/m²)', 'Prix Prédit (€/m²)', 'Erreur Abs (€/m²)', 'Erreur %'
            ]
            st.dataframe(worst_predictions, use_container_width=True, hide_index=True)
        
        # Statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Observations", f"{len(test_results):,}")
            st.metric("% of Total", f"{len(test_results)/len(test_results)*100:.1f}%")
        
        with col2:
            st.metric("Filtered MAE", f"{test_results['abs_error'].mean():.2f} €/m²")
            st.metric("Filtered RMSE", f"{np.sqrt((test_results['error']**2).mean()):.2f} €/m²")
        
        with col3:
            st.metric("Median Abs Error", f"{test_results['abs_error'].median():.2f} €/m²")
            st.metric("90th Percentile Error", f"{test_results['abs_error'].quantile(0.9):.2f} €/m²")
        
        with col4:
            st.metric("Mean % Error", f"{test_results['abs_pct_error'].mean():.2f}%")
            st.metric("Median % Error", f"{test_results['abs_pct_error'].median():.2f}%")
    
    with tab3:
        st.subheader("Geographic Analysis")
        
        # By postal code
        postal_analysis = test_results.groupby('code_postal').agg({
            'actual_price_m2': 'mean',
            'predicted_price_m2': 'mean',
            'abs_error': 'mean',
            'abs_pct_error': 'mean',
            'code_postal': 'count'
        }).rename(columns={'code_postal': 'count'}).reset_index()
        postal_analysis['count'] = postal_analysis['count'].astype(int)
        postal_analysis = postal_analysis.sort_values('abs_error', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 20 Postal Codes by Error")
            fig_postal_error = px.bar(
                postal_analysis.head(20),
                x='code_postal',
                y='abs_error',
                color='abs_error',
                color_continuous_scale='Reds',
                labels={
                    'code_postal': 'Code Postal',
                    'abs_error': 'Erreur Absolue Moyenne (€/m²)'
                }
            )
            fig_postal_error.update_layout(height=400)
            st.plotly_chart(fig_postal_error, use_container_width=True)
        
        with col2:
            st.subheader("Actual vs Predicted by Postal Code")
            top_postal = postal_analysis.nlargest(15, 'count')
            fig_postal_compare = go.Figure()
            fig_postal_compare.add_trace(go.Bar(
                x=top_postal['code_postal'].astype(str),
                y=top_postal['actual_price_m2'],
                name='Prix Réel',
                marker_color='#FF6B6B'
            ))
            fig_postal_compare.add_trace(go.Bar(
                x=top_postal['code_postal'].astype(str),
                y=top_postal['predicted_price_m2'],
                name='Prix Prédit',
                marker_color='#4ECDC4'
            ))
            fig_postal_compare.update_layout(
                barmode='group',
                xaxis_title='Code Postal',
                yaxis_title='Prix Moyen (€/m²)',
                height=400
            )
            st.plotly_chart(fig_postal_compare, use_container_width=True)
        
        # Detailed table
        st.subheader("Statistics by Postal Code")
        postal_display = postal_analysis.copy()
        postal_display.columns = [
            'Code Postal', 'Prix Réel Moyen (€/m²)', 'Prix Prédit Moyen (€/m²)',
            'Erreur Abs Moyenne (€/m²)', 'Erreur % Moyenne', 'Nombre Observations'
        ]
        postal_display = postal_display.round(2)
        st.dataframe(postal_display, use_container_width=True, hide_index=True, height=400)
    
    with tab4:
        st.subheader("Feature Importance Analysis")
        
        # Get feature importance
        try:
            if hasattr(model, 'feature_importances_'):
                # Get feature names
                if isinstance(preprocessor, dict):
                    if 'drop_cols' in preprocessor:
                        # CB
                        feature_names = X_train.columns if hasattr(X_train, 'columns') else preprocessor['feature_columns']
                    else:
                        # NP
                        feature_names = preprocessor['feature_names']
                else:
                    # RDF
                    try:
                        feature_names = preprocessor.get_feature_names_out()
                    except:
                        # Fallback
                        feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]
                
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Top 20 Most Important Features")
                    fig_importance = px.bar(
                        feature_importance.head(20),
                        x='importance',
                        y='feature',
                        orientation='h',
                        color='importance',
                        color_continuous_scale='Viridis',
                        labels={
                            'importance': 'Importance Score',
                            'feature': 'Feature'
                        }
                    )
                    fig_importance.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=600
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                with col2:
                    st.subheader("Top Features Table")
                    importance_table = feature_importance.head(20).copy()
                    importance_table['importance'] = importance_table['importance'].round(4)
                    importance_table.index = range(1, 21)
                    st.dataframe(importance_table, use_container_width=True, height=600)
                
                # Feature importance table
                st.subheader("All Features")
                full_importance = feature_importance.copy()
                full_importance['importance'] = full_importance['importance'].round(6)
                full_importance.index = range(1, len(full_importance) + 1)
                st.dataframe(full_importance, use_container_width=True, height=400)
            else:
                st.info("Feature importance not available for this model.")
        except Exception as e:
            st.error(f"Error displaying feature importance: {str(e)}")
            st.exception(e)
    
    with tab5:
        st.subheader("Detailed Prediction Results")
        
        # Select columns
        display_columns = st.multiselect(
            "Select columns to display",
            options=['code_postal', 'year_month', 'surface_reelle_bati', 
                    'nombre_pieces_principales', 'latitude', 'longitude',
                    'nearest_metro_dist_km', 'dist_center',
                    'actual_price_m2', 'predicted_price_m2', 'error', 'abs_error', 
                    'pct_error', 'abs_pct_error'],
            default=['code_postal', 'year_month', 'surface_reelle_bati', 'nombre_pieces_principales',
                    'actual_price_m2', 'predicted_price_m2', 'abs_error', 'abs_pct_error']
        )
        
        if display_columns:
            display_df = test_results[display_columns].copy()
            display_df = display_df.round(2)
            
            # Select options
            sort_by = st.selectbox(
                "Sort by",
                options=[col for col in ['abs_error', 'abs_pct_error', 'actual_price_m2', 'predicted_price_m2', 'year_month'] if col in display_columns],
                index=0
            )
            
            sort_order = st.radio("Sort order", ['Descending', 'Ascending'], horizontal=True)
            ascending = (sort_order == 'Ascending')
            
            display_df_sorted = display_df.sort_values(sort_by, ascending=ascending)
            
            st.dataframe(
                display_df_sorted,
                use_container_width=True,
                hide_index=True,
                height=500
            )
            
            # Download
            csv = display_df_sorted.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"{selected_model_key}_predictions.csv",
                mime="text/csv"
            )
        else:
            st.warning("Please select at least one column to display.")
        
    st.sidebar.markdown("---")
    
    # Features count
    if isinstance(preprocessor, dict):
        if 'drop_cols' in preprocessor:
            n_features = len(preprocessor['feature_columns'])
        else:
            n_features = len(preprocessor['feature_names'])
    else:
        try:
            n_features = len(preprocessor.get_feature_names_out())
        except:
            n_features = X_train.shape[1] if hasattr(X_train, 'shape') else "Unknown"
    
    st.sidebar.markdown(f"""
    **ℹ️ {model_name}**
    - **Target:** Price per m² (price_per_sqrtm)
    - **Aggregation:** By postal code, year, month
    - **Training:** Temporal split (80/20)
    
    **Available Models:**
    """)
    
    for key, info in models.items():
        if info['available']:
            st.sidebar.markdown(f"✅ {info['name']}")
        else:
            st.sidebar.markdown(f"❌ {info['name']} (not found)")
    
except FileNotFoundError as e:
    st.error(f"""
    ❌ **Data files not found!**
    
    Please ensure the following files exist:
    - `Data/xp.csv`
    - `Data/metro.csv`
    - `src/model_cat.pkl` and `src/preprocessor_cat.pkl` (for CatBoost)
    - `src/model_random_forest.pkl` and `src/preprocessor_random_forest.pkl` (for Random Forest)
    - `src/model_numpy.pkl` and `src/preprocessor_numpy.pkl` (for NumPy Random Forest)
    
    Error details: {str(e)}
    """)
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.exception(e)
