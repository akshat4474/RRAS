

# 🚑 RRAS — Relief Routing & Allocation System

**Prototype Notice:** This app is an early-stage prototype developed for a competition. Features and results may be incomplete, inaccurate, or subject to change.

## What is RRAS?
RRAS is an emergency resource planning and routing system for disaster relief. It combines AI-based allocation and smart routing to help relief teams deliver food, water, and medicine efficiently.

### Key Features
- Predicts resource demand for each area using ML
- Calculates shortest, safest delivery routes
- Interactive dashboard: upload data, view maps
- Scenario simulation: adjust parameters, see impact live

## Project Structure
```
api/         # FastAPI backend (allocation, routing)
core/        # Algorithms (allocation, routing)
app/         # Streamlit dashboard (main UI)
data/        # Sample datasets (areas, depots, roads)
models/      # ML models (not tracked in git)
scripts/     # Helper scripts (run, test)
requirements.txt
.env.example
README.md
```


## Setup Instructions

1. **Install Python 3.10+**
2. **Clone the repo:**
  ```sh
  git clone https://github.com/akshat4474/RRAS.git
  cd RRAS
  ```
3. **Create and activate a virtual environment:**
  ```sh
  python -m venv venv
  venv\Scripts\activate   # Windows
  source venv/bin/activate # Mac/Linux
  ```
4. **Install dependencies:**
  ```sh
  pip install -r requirements.txt
  ```
5. **Copy environment file:**
  ```sh
  copy .env.example .env   # Windows
  cp .env.example .env     # Mac/Linux
  ```

## How to Run (Windows)

**Always use the provided .bat files to run and test the app!**

### 1. Start the backend (API)
Double-click or run in terminal:
```
scripts\run_api.bat
```

### 2. Start the dashboard (Streamlit)
Double-click or run in terminal:
```
scripts\run_app.bat
```

### 3. Test API endpoints
Double-click or run in terminal:
```
scripts\test_api.bat
```

**Do not run uvicorn or streamlit directly. The .bat files set up everything for you.**


## How the App Works

1. **Upload CSVs:** areas.csv, depots.csv, roads.csv
2. **Run allocation:** ML model predicts food, water, medicine for each area
3. **Run routing:** Calculates shortest/safest route from depot to area
4. **View results:** Interactive map
5. **Scenario simulation:** Adjust parameters, see live impact

## Data Format
- **areas.csv:** area_id, name, lat, lon, severity, population, accessibility
- **depots.csv:** depot_id, name, lat, lon, capacity_food, capacity_water, capacity_meds
- **roads.csv:** from_node, to_node, distance_km, is_blocked

## Example Output
```json
{
  "trips": [
   {"trip_id":"D1->A3","total_km":6,"est_time_min":12},
   {"trip_id":"D2->A2","total_km":4,"est_time_min":8},
   {"trip_id":"D1->A1","total_km":9,"est_time_min":18}
  ],
  "allocation_table": [
   {"area_id":"A1","food_units":1800,"water_units":2400},
   {"area_id":"A2","food_units":2430,"water_units":3240},
   {"area_id":"A3","food_units":1050,"water_units":1400}
  ]
}
```

## Team
- Akshat: Backend, Routing, Integration
- Manthan: ML Allocation Model
- Yash & Himanshu: Dashboard, UI, Testing

## Future Plans
- Live traffic and blocked road updates
- Animated truck simulation on map
- Drone route planning
- ML-based disaster demand prediction

**Quick Demo:**

1. Upload sample CSVs from `data/mini/`
2. Click to run allocation and routing

