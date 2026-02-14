# Paris Real Estate Analysis

A Streamlit dashboard and Jupyter notebooks for exploring and modeling Paris real-estate transactions using machine learning.

## 🎯 Overview

This project analyzes French property transaction data from Paris, computes key metrics (e.g., price per m²), and provides interactive visualizations and ML models to analyze price drivers across Paris arrondissements.

The main application uses a Random Forest Regressor to predict property prices per square meter and provides comprehensive visualizations comparing predicted vs actual values.

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lcodecorn/ImmoParis
   cd Immo
   ```

2. **Create and activate a virtual environment** (recommended)
   
   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**

   Appartements: [Demandes de valeurs foncieres geolocalisees](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees)
   Metro: [Lignes et stations de metro en France](https://www.data.gouv.fr/datasets/lignes-et-stations-de-metro-en-france)
   Geojson Paris : [Frontière arrondissement](https://opendata.paris.fr/explore/dataset/arrondissements/download/?format=geojson)

5. **Run the Streamlit app**
   ```bash
   streamlit run src/app.py
   ```
   
   The app will automatically open in your browser at `http://localhost:8501`

### Running Jupyter Notebooks

If you want to explore the notebooks:

```bash
jupyter notebook Notebooks/
```

Or use JupyterLab:

```bash
jupyter lab Notebooks/
```

## 📁 Project Structure

```
Immo/
├── src/
│   ├──  app_dif_model.py  
|   ├──  app.py         
│   ├──  model_cat.pkl
|   ├──  model_numpy.pkl
|   ├──  model_random_forest.pkl
|   ├──  numpy_models.py
|   ├──  preprocessor_cat.pkl
|   ├──  preprocessor_numpy.pkl
│   └──  preprocessor_random_forest.pkl
├── Notebooks/
│   ├──  dvf_vis.ipynb         
│   ├──  ml.ipynb         
|   ├──  ml-catboost.ipynb
|   └──  ml_numpy.ipynb
├── Data/
│   ├──  75_2021.csv
|   ├──  75_2022.csv
│   ├──  75_2023.csv
|   ├──  75_2024.csv
|   ├──  75_2025.csv
|   ├──  metro-france.csv
|   ├──  metro.csv
|   └──  xp.csv
|
├── requirements.txt       
├── .gitignore             
└── README.md              
```

## 🔧 Features

- **Interactive Dashboard**: Compare real vs predicted property prices
- **Multiple Visualizations**: 
  - Scatter plots (predictions vs actual)
  - Error analysis (distribution, residuals)
  - Analysis by postal code
- **Filtering Options**: Filter by postal code, year, and error threshold
- **Model Metrics**: R² score, MAE, RMSE, and more
- **Export Functionality**: Download results as CSV

## 📊 Model Details

- **Algorithm**: Random Forest Regressor / Cat Boost Regressor / Numpy Random Forest Regressor
- **Target Variable**: Price per m² (`price_per_sqrtm`)
- **Test Size**: 20%

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📝 Notes

- The application uses relative paths, making it portable across different systems
- Model files (`.pkl`) are auto-generated chen running the model's notebook
- Data files in the `Data/` directory are not tracked by git (add them manually if needed)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!
There is still some stuff to fix !

## Author

Léo Souris

## 📄 License

This project is open source and available under the MIT License.
