import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import folium
import streamlit.components.v1 as components
import polyline
from functools import lru_cache
from datetime import timedelta

# ---------------------------
# Config
# ---------------------------
load_dotenv()
API = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60

st.set_page_config(page_title="🚑 RRAS Dashboard", layout="wide")

# ---------------------------
# Session state
# ---------------------------
for key in ["route_result", "plan_result", "route_map_key", "route_map_html", "plan_map_key", "plan_map_html"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------
# Helpers
# ---------------------------
def df_from_upload(f):
    if f is None:
        return pd.DataFrame()
    return pd.read_csv(f)

def make_node_coords(areas_df, depots_df):
    coords = {}
    if not areas_df.empty and {"area_id", "lat", "lon"}.issubset(areas_df.columns):
        for _, r in areas_df.iterrows():
            coords[str(r["area_id"])] = (float(r["lat"]), float(r["lon"]))
    if not depots_df.empty and {"depot_id", "lat", "lon"}.issubset(depots_df.columns):
        for _, r in depots_df.iterrows():
            coords[str(r["depot_id"])] = (float(r["lat"]), float(r["lon"]))
    return coords

def severity_color(sev: int):
    return {0:"green",1:"green",2:"lightblue",3:"blue",4:"orange",5:"red"}.get(int(sev), "blue")

def transport_mode(sev:int):
    """Assign transport mode based on severity"""
    return {0:"Bike",1:"Bike",2:"Van",3:"Truck",4:"Truck",5:"Helicopter"}.get(int(sev),"Truck")

def speed_kmh(mode:str):
    """Average speed per transport mode in km/h"""
    return {"Bike":20,"Van":40,"Truck":60,"Helicopter":150}.get(mode,40)

def folium_base_map(node_coords, areas_df, depots_df):
    if node_coords:
        lats = [lt for lt,_ in node_coords.values()]
        lons = [ln for _,ln in node_coords.values()]
        center = (sum(lats)/len(lats), sum(lons)/len(lons))
    else:
        center = (20.5937,78.9629)

    m = folium.Map(location=center, zoom_start=6, control_scale=True)

    # Depots
    if not depots_df.empty:
        for _, r in depots_df.iterrows():
            did = str(r["depot_id"])
            lat, lon = node_coords.get(did,(None,None))
            if lat is None: continue
            popup_html = f"<b>Depot {did}</b><br>Food: {r.get('capacity_food','N/A')}<br>Water: {r.get('capacity_water','N/A')}<br>Medkits: {r.get('capacity_meds',r.get('capacity_med','N/A'))}"
            folium.Marker([lat,lon],tooltip=f"Depot {did}",popup=popup_html,icon=folium.Icon(color="green",icon="truck",prefix="fa")).add_to(m)

    # Areas
    if not areas_df.empty:
        for _, r in areas_df.iterrows():
            aid = str(r["area_id"])
            lat, lon = node_coords.get(aid,(None,None))
            if lat is None: continue
            sev = int(r.get("severity",0))
            popup_html = f"<b>Area {aid}</b><br>Severity: {sev}<br>Population: {r.get('population','N/A')}<br>Mode: {transport_mode(sev)}"
            folium.CircleMarker([lat,lon], radius=6, tooltip=f"Area {aid} | Sev {sev}", popup=popup_html, color=severity_color(sev), fill=True, fill_opacity=0.8).add_to(m)

    # Severity Legend
    legend_html = """
     <div style="position: fixed; bottom: 30px; left: 30px; width: 220px; height: 180px; 
                 border:2px solid grey; z-index:9999; font-size:14px; background:white; padding:10px;">
     <b>Severity Legend</b><br>
     <i style="background:green;width:10px;height:10px;float:left;margin-right:5px"></i> Low (0-1)<br>
     <i style="background:lightblue;width:10px;height:10px;float:left;margin-right:5px"></i> Moderate (2)<br>
     <i style="background:blue;width:10px;height:10px;float:left;margin-right:5px"></i> Serious (3)<br>
     <i style="background:orange;width:10px;height:10px;float:left;margin-right:5px"></i> Severe (4)<br>
     <i style="background:red;width:10px;height:10px;float:left;margin-right:5px"></i> Critical (5)<br>
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Urgency box
    if not areas_df.empty:
        areas_sorted = areas_df.sort_values(by=["severity","population"], ascending=[False,False])
        urgency_html = "<b>Urgency Priority</b><ol style='padding-left:15px'>"
        for _, r in areas_sorted.iterrows():
            urgency_html += f"<li>Area {r['area_id']} | Sev {r.get('severity',0)} | Pop {r.get('population','N/A')} | Mode {transport_mode(r.get('severity',0))}</li>"
        urgency_html += "</ol>"
        box_html = f"""
        <div style="position: fixed; top: 30px; right: 30px; width: 280px; height: 320px;
                    border:2px solid grey; z-index:9999; font-size:12px; background:white; overflow:auto; padding:10px">
        {urgency_html}
        </div>
        """
        m.get_root().html.add_child(folium.Element(box_html))

    return m

def map_html(m): 
    try:
        return m.get_root().render()  # Newer Folium versions
    except AttributeError:
        return m._repr_html_()    # Fallback for older Folium versions

def download_btn(df,label,filename,key):
    if df is None or df.empty: return
    st.download_button(label=label, data=df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key, type="secondary")

def files_bytes(**named_files):
    out = {}
    for field,f in named_files.items():
        if f is None: continue
        out[field] = (f.name or f"{field}.csv", f.getvalue(), "text/csv")
    return out

@lru_cache(maxsize=1024)
def osrm_seg(lat1,lon1,lat2,lon2):
    url=f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=polyline"
    try:
        r=requests.get(url,timeout=20)
        r.raise_for_status()
        data=r.json()
        if "routes" in data and data["routes"]:
            return tuple(polyline.decode(data["routes"][0]["geometry"]))
    except:
        pass
    return ((lat1,lon1),(lat2,lon2))

def osrm_path_for_nodes(nodes,node_coords):
    coords_seq=[]
    for i in range(len(nodes)-1):
        n1,n2 = nodes[i],nodes[i+1]
        if n1 in node_coords and n2 in node_coords:
            lat1,lon1=node_coords[n1]
            lat2,lon2=node_coords[n2]
            seg=osrm_seg(lat1,lon1,lat2,lon2)
            coords_seq.extend(seg if i==0 else seg[1:])
    return coords_seq

def folium_route_map_osrm(node_coords,areas_df,depots_df,path_nodes,sev=None):
    m=folium_base_map(node_coords,areas_df,depots_df)
    if path_nodes and len(path_nodes)>=2:
        coords_seq=osrm_path_for_nodes(path_nodes,node_coords)
        mode = transport_mode(sev) if sev is not None else "Truck"
        speed = speed_kmh(mode)
        # compute ETA for each segment
        total_distance = 0
        for i in range(len(coords_seq)-1):
            lat1,lon1 = coords_seq[i]
            lat2,lon2 = coords_seq[i+1]
            # Haversine distance
            from math import radians, cos, sin, asin, sqrt
            def haversine(lat1,lon1,lat2,lon2):
                R = 6371
                dlat = radians(lat2-lat1)
                dlon = radians(lon2-lon1)
                a = sin(dlat/2)*2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)*2
                # Clamp 'a' to valid range [0, 1] to avoid math domain errors
                a = max(0, min(1, a))
                c = 2*asin(sqrt(a))
                return R*c
            seg_dist = haversine(lat1,lon1,lat2,lon2)
            total_distance += seg_dist
        eta_min = int(total_distance/speed*60)
        folium.PolyLine(coords_seq,weight=5,color=severity_color(sev) if sev is not None else "blue",
                        opacity=0.8, tooltip=f"Mode: {mode} | ETA: {eta_min} min").add_to(m)
    return m

def folium_plan_map_osrm(node_coords,areas_df,depots_df,trips):
    m=folium_base_map(node_coords,areas_df,depots_df)
    colors=["blue","red","green","purple","orange","black","cadetblue","darkred"]
    for idx, trip in enumerate(trips):
        nodes = trip.get("path_nodes") or []
        if len(nodes)<2: continue
        coords_seq = osrm_path_for_nodes(nodes,node_coords)
        sev = trip.get("severity",3)
        mode = transport_mode(sev)
        speed = speed_kmh(mode)
        total_distance = 0
        from math import radians, cos, sin, asin, sqrt
        def haversine(lat1,lon1,lat2,lon2):
            R = 6371
            dlat = radians(lat2-lat1)
            dlon = radians(lon2-lon1)
            a = sin(dlat/2)*2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)*2
            # Clamp 'a' to valid range [0, 1] to avoid math domain errors
            a = max(0, min(1, a))
            c = 2*asin(sqrt(a))
            return R*c
        for i in range(len(coords_seq)-1):
            lat1,lon1=coords_seq[i]
            lat2,lon2=coords_seq[i+1]
            total_distance+=haversine(lat1,lon1,lat2,lon2)
        eta_min = int(total_distance/speed*60)
        color=colors[idx%len(colors)]
        folium.PolyLine(coords_seq,weight=5,color=color,opacity=0.8,
                        tooltip=f"Trip {trip.get('trip_id','')} | Mode: {mode} | ETA: {eta_min} min").add_to(m)
    return m

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙ Settings")
areas_file=st.sidebar.file_uploader("📍 Areas CSV", type="csv")
depots_file=st.sidebar.file_uploader("🏭 Depots CSV", type="csv")
roads_file=st.sidebar.file_uploader("🛣 Roads CSV", type="csv")
st.sidebar.divider()
st.sidebar.caption("API Health")
try: st.sidebar.json(requests.get(f"{API}/health",timeout=5).json())
except Exception as e: st.sidebar.error(f"API not reachable: {e}")

areas_df=df_from_upload(areas_file)
depots_df=df_from_upload(depots_file)
roads_df=df_from_upload(roads_file)
node_coords=make_node_coords(areas_df,depots_df)

# ---------------------------
# Tabs
# ---------------------------
tab1,tab2,tab3,tab4 = st.tabs(["📦 Allocation","🛣 Single Route","🗺 Full Plan","🧪 Scenario Simulation"])

# --- Allocation ---
with tab1:
    st.subheader("📦 Predict Allocation")

    # ML Model Integration will run only when button is pressed
    if st.button("Run Allocation", key="ml_alloc"):
        if areas_file is None or depots_file is None:
            st.warning("Upload both Areas and Depots CSV files")
        else:
            import joblib
            import numpy as np
            ml_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_allocation_model.pkl")
            try:
                ml_model = joblib.load(ml_model_path)
                st.markdown("#### ML Allocation Predictor")
                from math import radians, cos, sin, asin, sqrt
                def haversine(lat1, lon1, lat2, lon2):
                    R = 6371
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    return R * c
                results = []
                for _, area in areas_df.iterrows():
                    # Find nearest depot
                    min_dist = float('inf')
                    nearest_depot = None
                    for _, depot in depots_df.iterrows():
                        dist = haversine(float(area['lat']), float(area['lon']), float(depot['lat']), float(depot['lon']))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_depot = depot
                    # Prepare features for prediction (must match model training)
                    features = {
                        'depot_capacity': nearest_depot['capacity_food'] + nearest_depot['capacity_water'] + nearest_depot['capacity_meds'],
                        'depot_food': nearest_depot['capacity_food'],
                        'depot_water': nearest_depot['capacity_water'],
                        'depot_meds': nearest_depot['capacity_meds'],
                        'area_population': area['population'],
                        'severity': area['severity'],
                        'accessibility': area['accessibility'],
                        'distance': min_dist,
                        'risk_level': 0
                    }
                    X_pred = pd.DataFrame([features])
                    alloc_food, alloc_water, alloc_meds = ml_model.predict(X_pred)[0]
                    results.append({
                        'area_id': area['area_id'],
                        'area_name': area.get('name', ''),
                        'depot_id': nearest_depot['depot_id'],
                        'depot_name': nearest_depot.get('name', ''),
                        'food_supplied': alloc_food,
                        'water_supplied': alloc_water,
                        'medicine_supplied': alloc_meds,
                        'severity': area['severity'],
                        'population': area['population'],
                        'accessibility': area['accessibility'],
                        'distance_to_depot': min_dist
                    })
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                download_btn(df_results, "⬇ Download ML Allocation", "ml_allocation_table.csv", "dl_ml_alloc")
            except Exception as e:
                st.warning(f"ML model not available: {e}")

    # ...existing code...

# --- Single Route ---
with tab2:
    st.subheader("🛣 Compute Single Route")
    depot_options=depots_df["depot_id"].astype(str).tolist() if not depots_df.empty else []
    area_options=areas_df["area_id"].astype(str).tolist() if not areas_df.empty else []
    depot_node = st.selectbox("Depot ID", depot_options) if depot_options else None
    area_node = st.selectbox("Area ID", area_options) if area_options else None

    if st.button("Run Route", key="route"):
        if not all([areas_file,depots_file,roads_file]): st.warning("Upload all CSVs")
        elif not depot_node or not area_node: st.warning("Select depot and area")
        else:
            try:
                r=requests.post(f"{API}/route/run", files=files_bytes(areas=areas_file,depots=depots_file,roads=roads_file), data={"src":depot_node,"dst":area_node},timeout=TIMEOUT)
                route_json=r.json().get("route",{})
                st.json(route_json)
                path=route_json.get("path_nodes",[])
                
                # Fixed severity lookup - compare area_id as strings
                sev = 3  # default severity
                if not areas_df.empty and area_node:
                    # Convert both to strings for comparison
                    matching_rows = areas_df[areas_df["area_id"].astype(str) == str(area_node)]
                    if not matching_rows.empty:
                        sev = int(matching_rows["severity"].iloc[0])
                
                if path:
                    route_key=("route_osrm", tuple(path), tuple(sorted(node_coords.items())))
                    if st.session_state.route_map_key != route_key:
                        st.session_state.route_map_key=route_key
                        m=folium_route_map_osrm(node_coords,areas_df,depots_df,path,sev=sev)
                        st.session_state.route_map_html=map_html(m)
                    with st.expander("📍 Show Route Map", expanded=True):
                        components.html(st.session_state.route_map_html,height=600)
                    row=pd.DataFrame([{"depot_id":depot_node,"area_id":area_node,"path":" -> ".join(path),"total_km":route_json.get("total_km"),"est_time_min":route_json.get("est_time_min"),"mode":transport_mode(sev)}])
                    download_btn(row,"⬇ Download Route","route_table.csv","dl_route_single")
            except Exception as e: st.error(f"Route failed: {e}")

# --- Full Plan ---
with tab3:
    st.subheader("🗺 Generate Full Plan")
    if st.button("Run Full Plan", key="plan"):
        if areas_file is None or depots_file is None:
            st.warning("Upload both Areas and Depots CSV files")
        else:
            import joblib
            import numpy as np
            ml_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_allocation_model.pkl")
            try:
                ml_model = joblib.load(ml_model_path)
                st.markdown("#### ML Allocation Full Plan")
                from math import radians, cos, sin, asin, sqrt
                def haversine(lat1, lon1, lat2, lon2):
                    R = 6371
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    return R * c
                results = []
                for _, area in areas_df.iterrows():
                    # Find nearest depot
                    min_dist = float('inf')
                    nearest_depot = None
                    for _, depot in depots_df.iterrows():
                        dist = haversine(float(area['lat']), float(area['lon']), float(depot['lat']), float(depot['lon']))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_depot = depot
                    # Prepare features for prediction (must match model training)
                    features = {
                        'depot_capacity': nearest_depot['capacity_food'] + nearest_depot['capacity_water'] + nearest_depot['capacity_meds'],
                        'depot_food': nearest_depot['capacity_food'],
                        'depot_water': nearest_depot['capacity_water'],
                        'depot_meds': nearest_depot['capacity_meds'],
                        'area_population': area['population'],
                        'severity': area['severity'],
                        'accessibility': area['accessibility'],
                        'distance': min_dist,
                        'risk_level': 0
                    }
                    X_pred = pd.DataFrame([features])
                    alloc_food, alloc_water, alloc_meds = ml_model.predict(X_pred)[0]
                    results.append({
                        'area_id': area['area_id'],
                        'area_name': area.get('name', ''),
                        'depot_id': nearest_depot['depot_id'],
                        'depot_name': nearest_depot.get('name', ''),
                        'food_supplied': alloc_food,
                        'water_supplied': alloc_water,
                        'medicine_supplied': alloc_meds,
                        'severity': area['severity'],
                        'population': area['population'],
                        'accessibility': area['accessibility'],
                        'distance_to_depot': min_dist
                    })
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                download_btn(df_results, "⬇ Download ML Full Plan", "ml_full_plan_table.csv", "dl_ml_full_plan")

                # Generate trips for plan map (one trip per area from depot)
                trips = []
                for row in results:
                    trips.append({
                        "trip_id": f"{row['depot_id']}_{row['area_id']}",
                        "path_nodes": [row['depot_id'], row['area_id']],
                        "severity": row['severity']
                    })
                m = folium_plan_map_osrm(node_coords, areas_df, depots_df, trips)
                plan_map_html = map_html(m)
                with st.expander("🗺 Show Full Plan Map", expanded=True):
                    components.html(plan_map_html, height=600)
            except Exception as e:
                st.warning(f"ML model not available: {e}")

# --- Scenario Simulation ---
with tab4:
    st.subheader("🧪 Scenario Simulation Demo")
    st.caption("Adjust parameters in the sidebar and click Simulate Scenario to see changes.")
    # Use uploaded CSVs if available, else fallback to demo data
    if areas_df.empty:
        areas = [
            {"area_id": "A1", "name": "Vaishali Nagar", "population": 150000, "severity": 3, "accessibility": 0.8},
            {"area_id": "A2", "name": "Mansarovar", "population": 180000, "severity": 4, "accessibility": 0.7},
            {"area_id": "A3", "name": "Johri Bazaar", "population": 70000, "severity": 5, "accessibility": 0.5},
        ]
    else:
        areas = areas_df.to_dict(orient="records")
    if depots_df.empty:
        depots = [
            {"depot_id": "D1", "name": "VKI Central", "capacity_food": 20000, "capacity_water": 20000, "capacity_meds": 10000},
            {"depot_id": "D2", "name": "Sitapura South", "capacity_food": 15000, "capacity_water": 15000, "capacity_meds": 8000},
        ]
    else:
        depots = depots_df.to_dict(orient="records")
    # Sidebar controls for scenario simulation
    for area in areas:
        st.sidebar.subheader(f"Area {area['area_id']} - {area.get('name','')}")
        area["population"] = st.sidebar.number_input(f"Population ({area['area_id']})", min_value=1000, max_value=500000, value=int(area["population"]), step=1000)
        area["severity"] = st.sidebar.slider(f"Severity ({area['area_id']})", min_value=1, max_value=5, value=int(area["severity"]))
        area["accessibility"] = st.sidebar.slider(f"Accessibility ({area['area_id']})", min_value=0.0, max_value=1.0, value=float(area["accessibility"]), step=0.05)
    for depot in depots:
        st.sidebar.subheader(f"Depot {depot['depot_id']} - {depot.get('name','')}")
        depot["capacity_food"] = st.sidebar.number_input(f"Food Capacity ({depot['depot_id']})", min_value=1000, max_value=50000, value=int(depot["capacity_food"]), step=500)
        depot["capacity_water"] = st.sidebar.number_input(f"Water Capacity ({depot['depot_id']})", min_value=1000, max_value=50000, value=int(depot["capacity_water"]), step=500)
        depot["capacity_meds"] = st.sidebar.number_input(f"Medkits Capacity ({depot['depot_id']})", min_value=500, max_value=30000, value=int(depot["capacity_meds"]), step=500)
    if st.button("Simulate Scenario", key="scenario_sim"):
        # Use mini/roads.csv for demo if no upload
        import pandas as pd, os
        if roads_df.empty:
            roads_path = os.path.join(os.path.dirname(__file__), "..", "data", "mini", "roads.csv")
            roads_df = pd.read_csv(roads_path)
        # Build graph from roads.csv (custom Dijkstra)
        graph = {}
        for _, row in roads_df.iterrows():
            if str(row.get('is_blocked','')).lower() == 'true':
                continue
            a, b = str(row['from_node']), str(row['to_node'])
            dist = float(row['distance_km'])
            graph.setdefault(a, []).append((b, dist))
            graph.setdefault(b, []).append((a, dist))
        # ML allocation
        import joblib
        ml_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_allocation_model.pkl")
        try:
            ml_model = joblib.load(ml_model_path)
            coords = {
                "D1": (26.9860, 75.8240),
                "D2": (26.7920, 75.8770),
                "A1": (26.9110, 75.7410),
                "A2": (26.8460, 75.7700),
                "A3": (26.9230, 75.8260)
            }
            results = []
            features_used = []
            from math import radians, cos, sin, asin, sqrt
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371
                dlat = radians(lat2 - lat1)
                dlon = radians(lat2 - lat1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                return R * c
            for area in areas:
                area_lat, area_lon = coords.get(area['area_id'], (0,0))
                min_dist = float('inf')
                nearest_depot = None
                for depot in depots:
                    depot_lat, depot_lon = coords.get(depot['depot_id'], (0,0))
                    dist = haversine(area_lat, area_lon, depot_lat, depot_lon)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_depot = depot
                features = {
                    'depot_capacity': nearest_depot['capacity_food'] + nearest_depot['capacity_water'] + nearest_depot['capacity_meds'],
                    'depot_food': nearest_depot['capacity_food'],
                    'depot_water': nearest_depot['capacity_water'],
                    'depot_meds': nearest_depot['capacity_meds'],
                    'area_population': area['population'],
                    'severity': area['severity'],
                    'accessibility': area['accessibility'],
                    'distance': min_dist,
                    'risk_level': 0
                }
                features_used.append({
                    "area_id": area["area_id"],
                    "depot_id": nearest_depot["depot_id"],
                    "population": area["population"],
                    "severity": area["severity"],
                    "accessibility": area["accessibility"],
                    "depot_food": nearest_depot["capacity_food"],
                    "depot_water": nearest_depot["capacity_water"],
                    "depot_meds": nearest_depot["capacity_meds"],
                    "distance": min_dist
                })
                X_pred = pd.DataFrame([features])
                alloc_food, alloc_water, alloc_meds = ml_model.predict(X_pred)[0]
                results.append({
                    "area_id": area["area_id"],
                    "area_name": area.get("name", ""),
                    "depot_id": nearest_depot["depot_id"],
                    "depot_name": nearest_depot.get("name", ""),
                    "food_supplied": alloc_food,
                    "water_supplied": alloc_water,
                    "medicine_supplied": alloc_meds,
                    "severity": area["severity"],
                    "population": area["population"],
                    "accessibility": area["accessibility"],
                    "distance_to_depot": min_dist
                })
            # Show verification table for ML input features
            st.subheader("ML Model Input Data (from sliders)")
            st.dataframe(pd.DataFrame(features_used), use_container_width=True)
            df_results = pd.DataFrame(results)
            st.subheader("Simulated ML Allocation Results")
            st.dataframe(df_results, use_container_width=True)
            # Folium map logic
            coords = {
                "D1": (26.9860, 75.8240),
                "D2": (26.7920, 75.8770),
                "A1": (26.9110, 75.7410),
                "A2": (26.8460, 75.7700),
                "A3": (26.9230, 75.8260)
            }
            all_lats = [c[0] for c in coords.values()]
            all_lons = [c[1] for c in coords.values()]
            center = (sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons))
            import folium
            from folium.plugins import AntPath
            import streamlit.components.v1 as components
            m = folium.Map(location=center, zoom_start=12, control_scale=True)
            for depot in depots:
                lat, lon = coords.get(depot['depot_id'], (0,0))
                folium.Marker([lat, lon], tooltip=f"Depot {depot['depot_id']}", popup=depot.get('name',''), icon=folium.Icon(color="green", icon="truck", prefix="fa")).add_to(m)
            for area in areas:
                lat, lon = coords.get(area['area_id'], (0,0))
                folium.Marker([lat, lon], tooltip=f"Area {area['area_id']}", popup=area.get('name',''), icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
            for row in results:
                depot_id = row['depot_id']
                area_id = row['area_id']
                import requests
                start_lat, start_lon = coords.get(depot_id, (0,0))
                end_lat, end_lon = coords.get(area_id, (0,0))
                url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
                try:
                    r = requests.get(url, timeout=10)
                    data = r.json()
                    if "routes" in data and data["routes"]:
                        route_coords = data["routes"][0]["geometry"]["coordinates"]
                        path_coords = [(lat, lon) for lon, lat in route_coords]
                    else:
                        path_coords = [coords.get(depot_id, (0,0)), coords.get(area_id, (0,0))]
                except Exception:
                    path_coords = [coords.get(depot_id, (0,0)), coords.get(area_id, (0,0))]
                AntPath(path_coords, color="blue", weight=5, delay=800).add_to(m)
                folium.Marker(path_coords[0], icon=folium.Icon(color="blue", icon="truck", prefix="fa"), tooltip=f"Truck from {depot_id} to {area_id}").add_to(m)
            map_html = m.get_root().render()
            st.subheader("Truck Routes Map")
            components.html(map_html, height=600)
        except Exception as e:
            st.error(f"ML model not available: {e}")
    else:
        st.info("Adjust parameters in the sidebar and click Simulate Scenario to run the simulation.")