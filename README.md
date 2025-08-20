# 🚑 RRAS — Relief Routing & Allocation System

**Hackathon Project — Emergency Resource Planning & Routing**

In disasters, relief gets delayed because:
- Resources aren't allocated properly.
- Trucks don't know which roads are safe/fast.

RRAS solves this by combining AI-based relief allocation + emergency routing algorithms in one system.

It gives:
- ✅ Predicted resource demand (food, water, medicine).
- ✅ Shortest safe delivery routes.
- ✅ Interactive dashboard with maps, routes, and downloadable CSVs.

## 📂 Project Structure

```
.
├─ api/                  # FastAPI backend
│  ├─ main.py            # Entry point for backend
│  ├─ util.py
│  └─ routers/
│     ├─ allocate.py     # Relief allocation endpoints
│     ├─ route.py        # Single route calculation
│     └─ plan.py         # Full plan (routes + allocations)
├─ core/                 # Algorithms
│  ├─ allocation.py      # Relief allocation logic
│  └─ routing.py         # Routing logic (Dijkstra/A*)
├─ app/                  # Streamlit frontend
│  └─ app.py             # Dashboard UI
├─ data/
│  └─ mini/              # Sample dataset (Jaipur, India)
│     ├─ areas.csv
│     ├─ depots.csv
│     └─ roads.csv
├─ scripts/
│  └─ test_api.bat       # Simple test script for backend
├─ sample_response.json  # Example output
├─ requirements.txt      # Python dependencies
├─ .env.example          # Example env config
└─ README.md
```

## 🛠 Setup Instructions (Step-by-Step)

### 1) Install Python

Make sure you have Python 3.10+ installed.
Check with:

```bash
python --version
```

### 2) Clone the Repo

Download the code:

```bash
git clone <repo_url>
cd IIC2.0
```

### 3) Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**On Windows (PowerShell):**
```powershell
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

### 4) Install Requirements

```bash
pip install -r requirements.txt
```

### 5) Setup Environment

Copy the example env file:

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```

By default, it points to:
```
API_URL=http://127.0.0.1:8000
```

## ▶️ Running the Project

### 1) Start Backend (FastAPI)

```bash
uvicorn api.main:app --reload
```

If successful, you'll see:
```
Uvicorn running on http://127.0.0.1:8000
```

Check health:
```bash
curl http://127.0.0.1:8000/health
```

### 2) Test Backend Endpoints

Use the helper script:

```bash
scripts\test_api.bat   # Windows
```

Expected outputs:
- **Allocate** → Allocation table for each area.
- **Route** → Path from depot → area.
- **Plan** → Full plan (all depots, all areas).

### 3) Start Frontend (Streamlit Dashboard)

In another terminal (keep backend running!):

```bash
streamlit run app/app.py
```

This opens a browser tab at:
```
http://localhost:8501
```

## 📊 How It Works

### Input Data (CSV format)
- `areas.csv` → disaster-affected areas (id, lat, lon, severity, population).
- `depots.csv` → relief depots (id, lat, lon, capacities).
- `roads.csv` → road network (graph edges).

### Allocation Logic
- Uses population × severity to compute relief demand.
- Outputs units of food, water, medicine, + priority score.

### Routing Logic
- Uses Dijkstra (shortest path) or A* for speed.
- Avoids blocked roads if marked.
- Routes follow real roads using OSRM (OpenStreetMap API).

### Dashboard Features
- Upload CSVs → run allocation + routing.
- Map view with depots, areas, routes.
- Download CSV outputs.
- (Coming soon 🚧) Simulation layer → trucks moving on routes.

## 🧪 Example Run

**Input:**
- 2 depots (D1, D2).
- 3 areas (A1, A2, A3).
- Road connections.

**Output:**

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

## 👥 Team Roles

- **Akshat** → Backend + Routing + Integration.
- **Manthan** → Allocation model, math.
- **Yash & Himanshu** → Dashboard, UI polish, PPT, testing.

## 🌍 Future Extensions

- 🚦 Live traffic & blocked road updates.
- 🚑 Animated simulation of trucks on map.
- 📡 Drone route planning.
- 🔮 ML-based disaster demand prediction.

⚡ **Judges will see:** upload data → system auto-predicts demand + routes → shows map.
