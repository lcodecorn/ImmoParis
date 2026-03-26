# ImmoParis – Prédiction des prix immobiliers à Paris

Un projet Data Science complet pour analyser les transactions immobilières à Paris et prédire le prix au m² en combinant données DVF et informations sur la proximité des stations de métro.  
Trois modèles sont comparés : **CatBoost**, **Random Forest sklearn** et **Random Forest “from scratch” en NumPy**.

---

## 🎯 Objectif

- Fournir une **estimation fiable des prix immobiliers à Paris**  
- Identifier les **facteurs influençant le prix** (localisation, surface, accessibilité métro…)  
- Permettre aux utilisateurs de **comparer plusieurs modèles** via un dashboard interactif

---

## 📊 Résumé des résultats

| Modèle          |  RMSE (€/m²)  |   MAE (€)   |   R²   |
|-----------------|---------------|-------------|--------|
| CatBoost        | 533.47        | 354.60      | 0.8990 |
| Random Forest   | 515.76        | 363.18      | 0.9056 |
| NumPy RF        | 687.54        | 545.14      | 0.8322 |

**Insights clés :**
- La localisation (arrondissement / cluster) est le facteur dominant  
- La proximité du métro influence fortement le prix  
- Les modèles non linéaires (CatBoost) capturent mieux les interactions complexes  

---

## 🚀 Fonctionnalités principales

- Téléchargement automatique des **données DVF et métro**  
- Prétraitement et **feature engineering avancé** (KMeans, BallTree, agrégations)  
- Comparaison de **3 modèles différents**  
- Dashboard Streamlit interactif pour **visualiser les prédictions et comparer les modèles**  
- Notebooks de référence pour explorer les données et comprendre la logique des modèles  

---

## 📂 Contenu du projet

- `src/run.py` : pipeline complet pour télécharger, nettoyer les données et entraîner les modèles  
- `src/app.py` : dashboard Streamlit interactif  
- `src/numpy_models.py` : implémentation Random Forest en NumPy  
- `Data/` : fichiers CSV générés automatiquement (`xp.csv`, `metro.csv`)  
- `Model/` : modèles entraînés et preprocessors (`.pkl`)  
- `Notebooks/` : notebooks d’exploration et entraînement de référence  
- `docs/` : screenshots et visualisations pour le README  

---

## ⚙️ Prérequis

- Python `>=3.10`  
- pip  
---

## 💻 Installation

1. Créer un environnement virtuel :  

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

## Installer les dépendances :

```bash
pip install -r requirements.txt
```

## 🛠 Génération des données et entraînement des modèles

```bash
python src/run.py
```

Ce script exécute automatiquement :

1. Téléchargement et nettoyage des données (`prepare_data`)
2. Feature engineering partagé (`build_features`)
3. Entraînement CatBoost (`train_catboost`)
4. Entraînement Random Forest sklearn (`train_random_forest`)
5. Entraînement Random Forest NumPy (`train_numpy_rf`)
   
(cette execution peut donc prendre un peu de temps ~15-20min)

## Fichiers générés :

- Data/xp.csv (transactions pré-traitées)
- Data/metro.csv (stations métro)
- Model/*.pkl (modèles et preprocessors)

## 🌐 Lancer le dashboard Streamlit

```bash
streamlit run src/app.py
```
- Choisir le modèle dans la sidebar
- Visualiser les prédictions, erreurs et comparaisons
- Filtrer par arrondissement ou distance métro (optionnel)

## 📈 Visualisations

Distribution des erreurs pour CatBoost :

![Dashboard Streamlit CatBoost](https://github.com/lcodecorn/ImmoParis/blob/main/docs/CatBoost.jpg)


---

Distribution des erreurs pour RandomForest :

![Dashboard Streamlit RandomForest](https://github.com/lcodecorn/ImmoParis/blob/main/docs/RandomForest.png)



## 📚 Notebooks de référence
jupyter lab Notebooks/
- `dvf_vis.ipynb` : visualisations exploratoires
- `load_clean.ipynb` : nettoyage et prétraitement
- `ml_catboost.ipynb` : entraînement CatBoost
- `ml_numpy.ipynb` : Random Forest NumPy
- `ml.ipynb` : Random Forest sklearn

## 💡 Prochaines étapes / améliorations possibles
- Hyperparameter tuning et modèles supplémentaires (XGBoost, LightGBM)
- Déploiement Cloud du dashboard Streamlit
- Ajout d’autres features géographiques ou socio-économiques
- Tests unitaires pour preprocessing et modèles
