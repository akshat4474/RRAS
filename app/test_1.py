import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(page_title="Scenario Simulation Demo", layout="wide")
st.title("Scenario Simulation for RRAS Allocation")

# Example area and depot data
areas = [
    {"area_id": "A1", "name": "Vaishali Nagar", "population": 150000, "severity": 3, "accessibility": 0.8},
    {"area_id": "A2", "name": "Mansarovar", "population": 180000, "severity": 4, "accessibility": 0.7},
    {"area_id": "A3", "name": "Johri Bazaar", "population": 70000, "severity": 5, "accessibility": 0.5},
]
depots = [
    {"depot_id": "D1", "name": "VKI Central", "capacity_food": 20000, "capacity_water": 20000, "capacity_meds": 10000},
    {"depot_id": "D2", "name": "Sitapura South", "capacity_food": 15000, "capacity_water": 15000, "capacity_meds": 8000},
]

st.sidebar.header("Adjust Scenario Parameters")

# User controls for each area
for area in areas:
    st.sidebar.subheader(f"Area {area['area_id']} - {area['name']}")
    area["population"] = st.sidebar.number_input(f"Population ({area['area_id']})", min_value=1000, max_value=500000, value=area["population"], step=1000)
    area["severity"] = st.sidebar.slider(f"Severity ({area['area_id']})", min_value=1, max_value=5, value=area["severity"])
    area["accessibility"] = st.sidebar.slider(f"Accessibility ({area['area_id']})", min_value=0.0, max_value=1.0, value=area["accessibility"], step=0.05)

# User controls for each depot
for depot in depots:
    st.sidebar.subheader(f"Depot {depot['depot_id']} - {depot['name']}")
    depot["capacity_food"] = st.sidebar.number_input(f"Food Capacity ({depot['depot_id']})", min_value=1000, max_value=50000, value=depot["capacity_food"], step=500)
    depot["capacity_water"] = st.sidebar.number_input(f"Water Capacity ({depot['depot_id']})", min_value=1000, max_value=50000, value=depot["capacity_water"], step=500)
    depot["capacity_meds"] = st.sidebar.number_input(f"Medkits Capacity ({depot['depot_id']})", min_value=500, max_value=30000, value=depot["capacity_meds"], step=500)

if st.button("Simulate Scenario"):
    # Load roads.csv for road network
    roads_path = os.path.join(os.path.dirname(__file__), "..", "data", "mini", "roads.csv")
    roads_df = pd.read_csv(roads_path)

    # Build graph from roads.csv (custom Dijkstra)
    graph = {}
    for _, row in roads_df.iterrows():
        if str(row['is_blocked']).lower() == 'true':
            continue  # skip blocked roads
        a, b = str(row['from_node']), str(row['to_node'])
        dist = float(row['distance_km'])
        graph.setdefault(a, []).append((b, dist))
        graph.setdefault(b, []).append((a, dist))

    def dijkstra(graph, start, end):
        import heapq
        queue = [(0, start, [start])]
        visited = set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == end:
                return path
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
        return [start, end]  # fallback to straight line if no path

    import folium
    from folium.plugins import AntPath
    import streamlit.components.v1 as components

    # ...existing ML allocation code...
    try:
        ml_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_allocation_model.pkl")
        ml_model = joblib.load(ml_model_path)
        coords = {
            "D1": (26.9860, 75.8240),
            "D2": (26.7920, 75.8770),
            "A1": (26.9110, 75.7410),
            "A2": (26.8460, 75.7700),
            "A3": (26.9230, 75.8260)
        }
        results = []
        for area in areas:
            # Find nearest depot by straight-line distance
            from math import radians, cos, sin, asin, sqrt
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371
                dlat = radians(lat2 - lat1)
                dlon = radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                return R * c
            area_lat, area_lon = coords[area['area_id']]
            min_dist = float('inf')
            nearest_depot = None
            for depot in depots:
                depot_lat, depot_lon = coords[depot['depot_id']]
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
            X_pred = pd.DataFrame([features])
            alloc_food, alloc_water, alloc_meds = ml_model.predict(X_pred)[0]
            results.append({
                "area_id": area["area_id"],
                "area_name": area["name"],
                "depot_id": nearest_depot["depot_id"],
                "depot_name": nearest_depot["name"],
                "food_supplied": alloc_food,
                "water_supplied": alloc_water,
                "medicine_supplied": alloc_meds,
                "severity": area["severity"],
                "population": area["population"],
                "accessibility": area["accessibility"],
                "distance_to_depot": min_dist
            })
        df_results = pd.DataFrame(results)
        st.subheader("Simulated ML Allocation Results")
        st.dataframe(df_results, use_container_width=True)
        st.caption("Adjust parameters in the sidebar and click Simulate Scenario to see changes.")

        # Folium map logic (after results are defined)
        coords = {
            "D1": (26.9860, 75.8240),
            "D2": (26.7920, 75.8770),
            "A1": (26.9110, 75.7410),
            "A2": (26.8460, 75.7700),
            "A3": (26.9230, 75.8260)
        }
        all_lats = [c[0] for c in coords.values()]
        all_lons = [c[1] for c in coords.values()]
        center = (np.mean(all_lats), np.mean(all_lons))
        m = folium.Map(location=center, zoom_start=12, control_scale=True)
        for depot in depots:
            lat, lon = coords[depot['depot_id']]
            folium.Marker([lat, lon], tooltip=f"Depot {depot['depot_id']}", popup=depot['name'], icon=folium.Icon(color="green", icon="truck", prefix="fa")).add_to(m)
        for area in areas:
            lat, lon = coords[area['area_id']]
            folium.Marker([lat, lon], tooltip=f"Area {area['area_id']}", popup=area['name'], icon=folium.Icon(color="red", icon="home", prefix="fa")).add_to(m)
        for row in results:
            depot_id = row['depot_id']
            area_id = row['area_id']
            import requests
            start_lat, start_lon = coords[depot_id]
            end_lat, end_lon = coords[area_id]
            url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
            try:
                r = requests.get(url, timeout=10)
                data = r.json()
                if "routes" in data and data["routes"]:
                    route_coords = data["routes"][0]["geometry"]["coordinates"]
                    path_coords = [(lat, lon) for lon, lat in route_coords]
                else:
                    path_coords = [coords[depot_id], coords[area_id]]
            except Exception:
                path_coords = [coords[depot_id], coords[area_id]]
            AntPath(path_coords, color="blue", weight=5, delay=800).add_to(m)
            folium.Marker(
                path_coords[0],
                icon=folium.Icon(color="blue", icon="truck", prefix="fa"),
                tooltip=f"Truck from {depot_id} to {area_id}"
            ).add_to(m)
        map_html = m.get_root().render()
        st.subheader("Truck Routes Map")
        components.html(map_html, height=600)
    except Exception as e:
        st.error(f"ML model not available: {e}")
    import joblib
    ml_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_allocation_model.pkl")
    try:
        ml_model = joblib.load(ml_model_path)
        results = []
        for area in areas:
            # Find nearest depot (for demo, just pick D1)
            nearest_depot = depots[0]
            features = {
                'depot_capacity': nearest_depot['capacity_food'] + nearest_depot['capacity_water'] + nearest_depot['capacity_meds'],
                'depot_food': nearest_depot['capacity_food'],
                'depot_water': nearest_depot['capacity_water'],
                'depot_meds': nearest_depot['capacity_meds'],
                'area_population': area['population'],
                'severity': area['severity'],
                'accessibility': area['accessibility'],
                'distance': 10.0,  # Demo: fixed distance
                'risk_level': 0
            }
            X_pred = pd.DataFrame([features])
            alloc_food, alloc_water, alloc_meds = ml_model.predict(X_pred)[0]
            results.append({
                "area_id": area["area_id"],
                "area_name": area["name"],
                "depot_id": nearest_depot["depot_id"],
                "depot_name": nearest_depot["name"],
                "food_supplied": alloc_food,
                "water_supplied": alloc_water,
                "medicine_supplied": alloc_meds,
                "severity": area["severity"],
                "population": area["population"],
                "accessibility": area["accessibility"]
            })
        df_results = pd.DataFrame(results)
        st.subheader("Simulated ML Allocation Results")
        st.dataframe(df_results, use_container_width=True)
        st.caption("Adjust parameters in the sidebar and click Simulate Scenario to see changes.")
    except Exception as e:
        st.error(f"ML model not available: {e}")
else:
    st.info("Adjust parameters in the sidebar and click Simulate Scenario to run the simulation.")
