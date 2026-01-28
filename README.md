# Paris Real Estate Analysis

A Streamlit dashboard and Jupyter notebooks for exploring and modeling Paris real-estate transactions using machine learning.

## 📋 Contents

- **[src/app.py](src/app.py)** — Interactive Streamlit dashboard for model comparison and visualization
- **[Notebooks/ml.ipynb](Notebooks/ml.ipynb)** — Machine learning experiments & feature importance analysis
- **[Notebooks/dvf.ipynb](Notebooks/dvf.ipynb)** — Data exploration and DVF (Demandes de Valeurs Foncières) analysis
- **[Data/](Data/)** — Data file
- **[requirements.txt](requirements.txt)** — Python dependencies

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

   Data source: [Demandes de valeurs foncieres geolocalisees](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees)  
   Geojson Paris : [Frontière arrondissement](https://opendata.paris.fr/explore/dataset/arrondissements/download/?format=geojson)


   Please note that the data for this project was cleaned in Tableau; therefore, you will need to perform the cleaning yourself.

   Place your Excel data file (`BD_resultat_tableau.xlsx`) in the `Data/` directory

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
│   ├── app.py              # Main Streamlit application
│   ├── model.pkl           # Trained model (auto-generated if missing)
│   └── preprocessor.pkl    # Data preprocessor (auto-generated if missing)
├── Notebooks/
│   ├── ml.ipynb            # ML experiments
│   └── dvf.ipynb           # Data exploration
├── Data/
│   └─── BD_resultat_tableau.xlsx  # Main data file
│   
├── requirements.txt        # Python dependencies
├── .gitignore             
└── README.md              
```

## 🔧 Features

- **Interactive Dashboard**: Compare real vs predicted property prices
- **Multiple Visualizations**: 
  - Scatter plots (predictions vs actual)
  - Error analysis (distribution, residuals)
  - Analysis by postal code
  - Time series analysis
- **Filtering Options**: Filter by postal code, year, and error threshold
- **Model Metrics**: R² score, MAE, RMSE, and more
- **Export Functionality**: Download results as CSV

## 📊 Model Details

- **Algorithm**: Random Forest Regressor
- **Target Variable**: Price per m² (`valeur_par_surface_bati`)
- **Test Size**: 25%
- **Features**: Postal code, date, transaction type, number of rooms, surface area, coordinates

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📝 Notes

- The application uses relative paths, making it portable across different systems
- Model files (`.pkl`) are auto-generated if missing when running the `ml.ipynb` notebook
- Data files in the `Data/` directory are not tracked by git (add them manually if needed)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## Author

Léo Souris

## 📄 License

This project is open source and available under the MIT License.
