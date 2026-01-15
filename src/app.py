import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Model Comparison - Real vs Predicted", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("📊 Comparaison Modèle: Données Réelles vs Prédites")
st.markdown("---")

# Load and preprocess data (matching notebook exactly)
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data exactly as in the notebook"""
    # Load Excel data
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "Data", "BD_resultat_tableau.xlsx")
    df = pd.read_excel(data_path)
    
    # Filter data (matching notebook)
    df = df[df.type_local != "Maison"]
    df = df[df.nature_mutation != "Vente en l'état futur d'achèvement"]
    
    # Drop missing values
    df = df.dropna(how='any', axis=0)
    
    # Date processing
    df['date_mutation'] = pd.to_datetime(df['date_mutation'])
    df['annee_mois'] = df['date_mutation'].dt.to_period("M").astype(str)
    
    # Group by arrondissement for price per m²
    df_agg = df.groupby(['code_postal', 'annee_mois', 'nature_mutation']).agg({
        'valeur_par_surface_bati': 'mean',
        'nombre_pieces_principales': 'mean',
        'surface_reelle_bati': 'mean',
        'valeur_fonciere': 'mean',
        'longitude': 'mean',
        'latitude': 'mean'
    }).reset_index()
    
    return df, df_agg

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    """
    Load saved model and preprocessor.

    NOTE:
    You are seeing an error like:
      AttributeError: Can't get attribute '_RemainderColsList' on module 'sklearn.compose._column_transformer'
    This happens because the pickled preprocessor was saved with a different
    scikit-learn version than the one currently installed.

    To avoid crashing the app, we catch ANY exception here and force a retrain.
    If you want to reuse persisted models, regenerate them with the same
    scikit-learn version as your app environment.
    """
    possible_paths = [
        ("src/model.pkl", "src/preprocessor.pkl"),
        ("model.pkl", "preprocessor.pkl"),
    ]

    for model_path, prep_path in possible_paths:
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(prep_path, "rb") as f:
                preprocessor = pickle.load(f)
            return model, preprocessor
        except Exception:
            # Any error (FileNotFoundError, AttributeError due to sklearn version, etc.)
            continue

    # If we reach here, we couldn't safely load a model → trigger training path
    return None, None

# Train model if not saved (fallback)
@st.cache_resource
def train_model(df_agg):
    """Train model if not saved"""
    # Split features and target
    target = 'valeur_par_surface_bati'
    x = df_agg.drop(columns=[target, 'valeur_fonciere'])
    y = df_agg[target]
    
    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    
    # Separate numeric and categorical features
    numeric_features = x.select_dtypes(include='number').columns
    categoric_features = x.select_dtypes(exclude='number').columns
    
    # Preprocessing
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    sc = StandardScaler()
    
    preprocessor = ColumnTransformer([
        ('cat', ohe, categoric_features),
        ('num', sc, numeric_features)
    ])
    
    # Fit and transform
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    
    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train_processed, y_train)
    
    # Make predictions
    y_train_pred = model.predict(x_train_processed)
    y_test_pred = model.predict(x_test_processed)
    
    return model, preprocessor, x_train, x_test, y_train, y_test, y_train_pred, y_test_pred

# Generate comparison dataframe
def create_comparison_df(x_test, y_test, y_test_pred):
    """Create comparison dataframe with actual vs predicted"""
    results_df = pd.DataFrame({
        'actual_prix_m2': y_test.values,
        'predicted_prix_m2': y_test_pred,
        'code_postal': x_test['code_postal'].values,
        'annee_mois': x_test['annee_mois'].values,
        'nature_mutation': x_test['nature_mutation'].values if 'nature_mutation' in x_test.columns else None,
        'nombre_pieces_principales': x_test['nombre_pieces_principales'].values if 'nombre_pieces_principales' in x_test.columns else None,
        'surface_reelle_bati': x_test['surface_reelle_bati'].values if 'surface_reelle_bati' in x_test.columns else None,
        'longitude': x_test['longitude'].values if 'longitude' in x_test.columns else None,
        'latitude': x_test['latitude'].values if 'latitude' in x_test.columns else None,
    })
    
    # Calculate errors
    results_df['error'] = results_df['predicted_prix_m2'] - results_df['actual_prix_m2']
    results_df['abs_error'] = results_df['error'].abs()
    results_df['pct_error'] = (results_df['error'] / results_df['actual_prix_m2'] * 100).round(2)
    results_df['abs_pct_error'] = results_df['pct_error'].abs()
    
    # Extract year and month for filtering
    results_df['year'] = results_df['annee_mois'].str[:4].astype(int)
    results_df['month'] = results_df['annee_mois'].str[5:7].astype(int)
    
    return results_df

# Load data
with st.spinner('Chargement des données...'):
    df_raw, df_agg = load_and_preprocess_data()
    model, preprocessor = load_model_and_preprocessor()

# Sidebar
st.sidebar.header("⚙️ Paramètres")

# Check if model exists
if model is None or preprocessor is None:
    st.sidebar.warning("⚠️ Modèle non trouvé. Entraînement en cours...")
    with st.spinner('Entraînement du modèle...'):
        model, preprocessor, x_train, x_test, y_train, y_test, y_train_pred, y_test_pred = train_model(df_agg)
    st.sidebar.success("✅ Modèle entraîné avec succès!")
else:
    st.sidebar.success("✅ Modèle chargé avec succès!")
    # Need to recreate test split to get predictions
    target = 'valeur_par_surface_bati'
    x = df_agg.drop(columns=[target, 'valeur_fonciere'])
    y = df_agg[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    
    # Transform and predict
    x_test_processed = preprocessor.transform(x_test)
    y_test_pred = model.predict(x_test_processed)
    y_train_pred = model.predict(preprocessor.transform(x_train))

# Create comparison dataframe
results_df = create_comparison_df(x_test, y_test, y_test_pred)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

# Display metrics
st.header("📈 Métriques du Modèle")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("R² Score (Train)", f"{train_r2:.3f}")
with col2:
    st.metric("R² Score (Test)", f"{test_r2:.3f}")
with col3:
    st.metric("MAE", f"{mae:.2f} €/m²")
with col4:
    st.metric("RMSE", f"{rmse:.2f} €/m²")
with col5:
    mean_error = results_df['error'].mean()
    st.metric("Erreur Moyenne", f"{mean_error:.2f} €/m²", delta=f"{mean_error:.2f}")

st.markdown("---")

# Filters
st.sidebar.subheader("🔍 Filtres")
filter_by_postal = st.sidebar.checkbox("Filtrer par code postal")
if filter_by_postal:
    postal_codes = sorted(results_df['code_postal'].unique())
    selected_postals = st.sidebar.multiselect("Codes postaux", postal_codes, default=postal_codes[:5] if len(postal_codes) >= 5 else postal_codes)
    results_df_filtered = results_df[results_df['code_postal'].isin(selected_postals)]
else:
    results_df_filtered = results_df.copy()

filter_by_year = st.sidebar.checkbox("Filtrer par année")
if filter_by_year:
    years = sorted(results_df_filtered['year'].unique())
    selected_years = st.sidebar.multiselect("Années", years, default=years)
    results_df_filtered = results_df_filtered[results_df_filtered['year'].isin(selected_years)]

filter_by_error = st.sidebar.checkbox("Filtrer par erreur")
if filter_by_error:
    max_error = st.sidebar.slider("Erreur maximale (%)", 0, 100, 100)
    results_df_filtered = results_df_filtered[results_df_filtered['abs_pct_error'] <= max_error]

# Visualizations
st.header("📊 Visualisations")

# Tab layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prédictions vs Réel", 
    "📉 Analyse des Erreurs", 
    "🗺️ Par Code Postal", 
    "📅 Par Période",
    "📋 Données Détaillées"
])

with tab1:
    st.subheader("Graphique de Dispersion: Prédictions vs Valeurs Réelles")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Scatter plot
        fig_scatter = px.scatter(
            results_df_filtered,
            x='actual_prix_m2',
            y='predicted_prix_m2',
            hover_data=['code_postal', 'annee_mois', 'error', 'pct_error'],
            title='Prédictions vs Valeurs Réelles',
            labels={
                'predicted_prix_m2': 'Prix moyen €/m² prédit',
                'actual_prix_m2': 'Prix moyen €/m² réel'
            },
            color='abs_error',
            color_continuous_scale='RdYlGn_r',
            size='abs_error',
            size_max=15
        )
        
        # Add perfect prediction line
        min_val = min(results_df_filtered['actual_prix_m2'].min(), results_df_filtered['predicted_prix_m2'].min())
        max_val = max(results_df_filtered['actual_prix_m2'].max(), results_df_filtered['predicted_prix_m2'].max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Prédiction Parfaite',
                line=dict(dash='dash', color='red', width=2)
            )
        )
        
        fig_scatter.update_layout(
            height=600,
            showlegend=True,
            xaxis_title="Prix réel €/m²",
            yaxis_title="Prix prédit €/m²"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Best and worst predictions
        st.subheader("Meilleures prédictions")
        best = results_df_filtered.nsmallest(5, 'abs_error')[['code_postal', 'annee_mois', 'actual_prix_m2', 'predicted_prix_m2', 'abs_error']]
        st.dataframe(best, use_container_width=True, hide_index=True)
        
        st.subheader("Pires prédictions")
        worst = results_df_filtered.nlargest(5, 'abs_error')[['code_postal', 'annee_mois', 'actual_prix_m2', 'predicted_prix_m2', 'abs_error']]
        st.dataframe(worst, use_container_width=True, hide_index=True)

    
    with col_right:
        st.subheader("Statistiques")
        st.metric("Nombre d'observations", len(results_df_filtered))
        st.metric("MAE filtré", f"{results_df_filtered['abs_error'].mean():.2f} €/m²")
        st.metric("Erreur médiane", f"{results_df_filtered['abs_error'].median():.2f} €/m²")
        st.metric("Erreur max", f"{results_df_filtered['abs_error'].max():.2f} €/m²")
        st.metric("Erreur min", f"{results_df_filtered['abs_error'].min():.2f} €/m²")
        st.metric("Erreur moyenne %", f"{results_df_filtered['abs_pct_error'].mean():.2f}%")
        
with tab2:
    st.subheader("Analyse des Erreurs de Prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Error distribution histogram
        fig_hist = px.histogram(
            results_df_filtered,
            x='error',
            nbins=50,
            title='Distribution des Erreurs',
            labels={'error': 'Erreur (€/m²)', 'count': 'Fréquence'},
            color_discrete_sequence=['#FF6B6B']
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Erreur = 0")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Percentage error distribution
        fig_pct = px.histogram(
            results_df_filtered,
            x='pct_error',
            nbins=50,
            title='Distribution des Erreurs en Pourcentage',
            labels={'pct_error': 'Erreur (%)', 'count': 'Fréquence'},
            color_discrete_sequence=['#4ECDC4']
        )
        fig_pct.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Erreur = 0%")
        fig_pct.update_layout(height=400)
        st.plotly_chart(fig_pct, use_container_width=True)
    
    with col2:
        # Absolute error distribution
        fig_abs = px.histogram(
            results_df_filtered,
            x='abs_error',
            nbins=50,
            title='Distribution des Erreurs Absolues',
            labels={'abs_error': 'Erreur Absolue (€/m²)', 'count': 'Fréquence'},
            color_discrete_sequence=['#95E1D3']
        )
        fig_abs.update_layout(height=400)
        st.plotly_chart(fig_abs, use_container_width=True)
        
        # Box plot of errors
        fig_box = px.box(
            results_df_filtered,
            y='error',
            title='Boîte à Moustaches des Erreurs',
            labels={'error': 'Erreur (€/m²)'},
            color_discrete_sequence=['#F38181']
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="red")
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Residuals plot
    st.subheader("Graphique des Résidus")
    fig_residuals = px.scatter(
        results_df_filtered,
        x='predicted_prix_m2',
        y='error',
        hover_data=['code_postal', 'annee_mois', 'actual_prix_m2'],
        title='Résidus vs Prédictions',
        labels={
            'predicted_prix_m2': 'Prix prédit €/m²',
            'error': 'Résidu (€/m²)'
        },
        color='abs_error',
        color_continuous_scale='RdYlGn_r'
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Résidu = 0")
    fig_residuals.update_layout(height=500)
    st.plotly_chart(fig_residuals, use_container_width=True)

with tab3:
    st.subheader("Analyse par Code Postal")
    
    # Aggregate by postal code
    postal_stats = results_df_filtered.groupby('code_postal').agg({
        'actual_prix_m2': 'mean',
        'predicted_prix_m2': 'mean',
        'error': 'mean',
        'abs_error': 'mean',
        'abs_pct_error': 'mean',
        'code_postal': 'count'
    }).rename(columns={'code_postal': 'count'}).reset_index()
    postal_stats['count'] = postal_stats['count'].astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart comparing actual vs predicted by postal code
        postal_stats_sorted = postal_stats.sort_values('actual_prix_m2', ascending=False).head(20)
        fig_postal = go.Figure()
        fig_postal.add_trace(go.Bar(
            x=postal_stats_sorted['code_postal'].astype(str),
            y=postal_stats_sorted['actual_prix_m2'],
            name='Prix Réel',
            marker_color='#FF6B6B'
        ))
        fig_postal.add_trace(go.Bar(
            x=postal_stats_sorted['code_postal'].astype(str),
            y=postal_stats_sorted['predicted_prix_m2'],
            name='Prix Prédit',
            marker_color='#4ECDC4'
        ))
        fig_postal.update_layout(
            title='Comparaison Prix Réel vs Prédit par Code Postal (Top 20)',
            xaxis_title='Code Postal',
            yaxis_title='Prix €/m²',
            barmode='group',
            height=500
        )
        st.plotly_chart(fig_postal, use_container_width=True)
    
    with col2:
        # Error by postal code
        postal_stats_error = postal_stats.sort_values('abs_error', ascending=False).head(20)
        fig_error_postal = px.bar(
            postal_stats_error,
            x='code_postal',
            y='abs_error',
            title='Erreur Absolue Moyenne par Code Postal (Top 20)',
            labels={'abs_error': 'Erreur Absolue Moyenne (€/m²)', 'code_postal': 'Code Postal'},
            color='abs_error',
            color_continuous_scale='Reds'
        )
        fig_error_postal.update_layout(height=500)
        st.plotly_chart(fig_error_postal, use_container_width=True)
    
    # Table with all postal codes
    st.subheader("Statistiques par Code Postal")
    postal_stats_display = postal_stats.rename(columns={
        'code_postal': 'Code Postal',
        'actual_prix_m2': 'Prix Réel Moyen',
        'predicted_prix_m2': 'Prix Prédit Moyen',
        'error': 'Erreur Moyenne',
        'abs_error': 'Erreur Absolue Moyenne',
        'abs_pct_error': 'Erreur % Moyenne',
        'count': 'Nombre Observations'
    })
    postal_stats_display = postal_stats_display.round(2)
    st.dataframe(postal_stats_display.sort_values('Erreur Absolue Moyenne', ascending=False), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Analyse par Période")
    
    # Aggregate by period
    period_stats = results_df_filtered.groupby('annee_mois').agg({
        'actual_prix_m2': 'mean',
        'predicted_prix_m2': 'mean',
        'error': 'mean',
        'abs_error': 'mean',
        'abs_pct_error': 'mean',
        'annee_mois': 'count'
    }).rename(columns={'annee_mois': 'count'}).reset_index()
    period_stats = period_stats.sort_values('annee_mois')
    period_stats['count'] = period_stats['count'].astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time series comparison
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=period_stats['annee_mois'],
            y=period_stats['actual_prix_m2'],
            mode='lines+markers',
            name='Prix Réel',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=8)
        ))
        fig_time.add_trace(go.Scatter(
            x=period_stats['annee_mois'],
            y=period_stats['predicted_prix_m2'],
            mode='lines+markers',
            name='Prix Prédit',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=8)
        ))
        fig_time.update_layout(
            title='Évolution des Prix Réels vs Prédits dans le Temps',
            xaxis_title='Période',
            yaxis_title='Prix €/m²',
            height=500,
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Error over time
        fig_error_time = px.line(
            period_stats,
            x='annee_mois',
            y='abs_error',
            markers=True,
            title='Évolution de l\'Erreur Absolue dans le Temps',
            labels={'abs_error': 'Erreur Absolue Moyenne (€/m²)', 'annee_mois': 'Période'},
            color_discrete_sequence=['#F38181']
        )
        fig_error_time.update_layout(height=500, xaxis=dict(tickangle=45))
        st.plotly_chart(fig_error_time, use_container_width=True)
    
    # Table with period statistics
    st.subheader("Statistiques par Période")
    period_stats_display = period_stats.rename(columns={
        'annee_mois': 'Période',
        'actual_prix_m2': 'Prix Réel Moyen',
        'predicted_prix_m2': 'Prix Prédit Moyen',
        'error': 'Erreur Moyenne',
        'abs_error': 'Erreur Absolue Moyenne',
        'abs_pct_error': 'Erreur % Moyenne',
        'count': 'Nombre Observations'
    })
    period_stats_display = period_stats_display.round(2)
    st.dataframe(period_stats_display, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Données Détaillées")
    
    # Display full dataframe
    display_df = results_df_filtered[[
        'code_postal', 'annee_mois', 'actual_prix_m2', 'predicted_prix_m2',
        'error', 'abs_error', 'pct_error', 'abs_pct_error'
    ]].copy()
    
    display_df = display_df.rename(columns={
        'code_postal': 'Code Postal',
        'annee_mois': 'Période',
        'actual_prix_m2': 'Prix Réel €/m²',
        'predicted_prix_m2': 'Prix Prédit €/m²',
        'error': 'Erreur €/m²',
        'abs_error': 'Erreur Absolue €/m²',
        'pct_error': 'Erreur %',
        'abs_pct_error': 'Erreur Absolue %'
    })
    
    display_df = display_df.round(2)
    
    st.dataframe(
        display_df.sort_values('Erreur Absolue €/m²', ascending=False),
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    # Download button
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données (CSV)",
        data=csv,
        file_name="comparison_results.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.sidebar.markdown("""
---
**ℹ️ À propos**
- **Modèle:** Random Forest Regressor
- **Cible:** Prix par m² (valeur_par_surface_bati)
- **Données:** Transactions immobilières Paris
- **Test Size:** 25%
""")

