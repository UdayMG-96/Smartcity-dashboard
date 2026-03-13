from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

print("Loading datasets...")
traffic_df   = pd.read_csv(os.path.join(DATASET_DIR, 'cleaned_traffic_data.csv'))
air_df       = pd.read_csv(os.path.join(DATASET_DIR, 'cleaned_AirQuality.csv'))
pollution_df = pd.read_csv(os.path.join(DATASET_DIR, 'clean_pollution_dataset.csv'))
water_df     = pd.read_csv(os.path.join(DATASET_DIR, 'cleaned_water_consumption_dataset.csv'))
energy_df    = pd.read_csv(os.path.join(DATASET_DIR, 'energy_preprocessed.csv'))
pop_df       = pd.read_csv(os.path.join(DATASET_DIR, 'cleaned_population_dataset.csv'))

# Pre-process traffic
t_min = traffic_df['Total'].min(); t_max = traffic_df['Total'].max()
traffic_df['congestion_level'] = ((traffic_df['Total'] - t_min) / (t_max - t_min)).clip(0, 1)
vc = traffic_df['CarCount'] + traffic_df['BikeCount'] + traffic_df['BusCount'] + traffic_df['TruckCount']
vc_min = vc.min(); vc_max = vc.max()
traffic_df['vehicle_count_real'] = ((vc - vc_min) / (vc_max - vc_min) * 1400 + 50).astype(int)
traffic_df['avg_speed'] = (60 - traffic_df['congestion_level'] * 50).round(1)

# Pre-process air quality
aq_col = 'PT08.S1(CO)'
aq_min = air_df[aq_col].min(); aq_max = air_df[aq_col].max()
air_df['aqi_scaled'] = ((air_df[aq_col] - aq_min) / (aq_max - aq_min) * 280 + 20).round(0).astype(int)

# Pre-process energy
energy_df.columns = energy_df.columns.str.strip().str.lstrip('\ufeff')
energy_df['Global_active_power'] = pd.to_numeric(energy_df['Global_active_power'], errors='coerce')
energy_df = energy_df.dropna(subset=['Global_active_power'])

# Pre-process water
w_col = 'Per_Capita_Water_Use_Liters_per_Day'
w_min = water_df[w_col].min(); w_max = water_df[w_col].max()
water_df['per_capita_real'] = ((water_df[w_col] - w_min) / (w_max - w_min) * 300 + 80).round(1)

JUNCTIONS  = ["MG Road Junction", "Shivaji Nagar", "FC Road Junction", "Hinjewadi IT Hub"]
ENV_ZONES  = ["Central City", "Industrial Zone", "Residential East", "Green Belt Area"]
WATER_ZONES = ["North District", "South District", "East District", "West District"]
WASTE_ZONES = ["Zone A – Central", "Zone B – North", "Zone C – Industrial", "Zone D – Residential"]

print("All datasets loaded.")


@app.route('/api/dashboard/summary')
def get_summary():
    hour = datetime.now().hour
    h_rows = traffic_df[traffic_df['Hour'] == hour] if 'Hour' in traffic_df.columns and len(traffic_df[traffic_df['Hour'] == hour]) >= 4 else traffic_df
    sample = h_rows.sample(n=min(4, len(h_rows)), random_state=random.randint(0, 999))
    avg_congestion = round(float(sample['congestion_level'].mean()), 2)

    aq_row  = air_df.sample(1, random_state=random.randint(0, 9999)).iloc[0]
    avg_aqi = int(aq_row['aqi_scaled'])

    en_sample = energy_df.sample(n=min(50, len(energy_df)), random_state=random.randint(0, 9999))
    total_energy = round(float(en_sample['Global_active_power'].sum()), 1)
    renewable    = round(random.uniform(0.22, 0.38), 2)

    t_status = "Critical" if avg_congestion > 0.7 else "Moderate" if avg_congestion > 0.4 else "Stable"
    a_status = "Hazardous" if avg_aqi > 200 else "Unhealthy" if avg_aqi > 150 else "Moderate" if avg_aqi > 100 else "Good"
    critical = (1 if avg_congestion > 0.7 else 0) + (1 if avg_aqi > 200 else 0)

    return jsonify({
        "traffic":     {"avg_congestion": avg_congestion, "status": t_status, "insight": "Deploy wardens to busy junctions." if avg_congestion > 0.7 else "Traffic normal."},
        "environment": {"avg_aqi": avg_aqi, "status": a_status},
        "energy":      {"total_consumption_mwh": total_energy, "renewable_ratio": renewable},
        "alerts":      {"active_critical": critical, "total": critical + random.randint(1, 3)},
        "timestamp":   datetime.now().isoformat()
    })


@app.route('/api/traffic/current')
def get_traffic():
    hour = datetime.now().hour
    h_rows = traffic_df[traffic_df['Hour'] == hour] if 'Hour' in traffic_df.columns and len(traffic_df[traffic_df['Hour'] == hour]) >= 4 else traffic_df
    sample = h_rows.sample(n=min(4, len(h_rows)), random_state=random.randint(0, 9999))
    result = []
    for i, (_, row) in enumerate(sample.iterrows()):
        result.append({
            "junction_name":     JUNCTIONS[i],
            "congestion_level":  round(float(row['congestion_level']), 2),
            "vehicle_count":     int(row['vehicle_count_real']),
            "avg_speed":         round(float(row['avg_speed']), 1),
            "traffic_situation": str(row.get('Traffic Situation', 'normal'))
        })
    return jsonify(result)


@app.route('/api/traffic/heatmap')
def get_traffic_heatmap():
    sample = pollution_df[['latitude', 'longitude', 'pollutant_avg']].dropna().sample(
        n=min(30, len(pollution_df)), random_state=random.randint(0, 999))
    p_min = sample['pollutant_avg'].min(); p_max = sample['pollutant_avg'].max()
    return jsonify([{"lat": round(float(r['latitude']), 5), "lng": round(float(r['longitude']), 5),
                     "intensity": round(float((r['pollutant_avg'] - p_min) / (p_max - p_min + 1e-9)), 2)}
                    for _, r in sample.iterrows()])


@app.route('/api/environment/current')
def get_environment():
    sample = air_df.sample(n=min(4, len(air_df)), random_state=random.randint(0, 9999))
    result = []
    for i, (_, row) in enumerate(sample.iterrows()):
        result.append({
            "location":    ENV_ZONES[i],
            "aqi":         int(row['aqi_scaled']),
            "temperature": round(float(row['T']), 1) if pd.notna(row.get('T')) else round(random.uniform(22,36),1),
            "humidity":    round(float(row['RH']), 1) if pd.notna(row.get('RH')) else random.randint(40,85),
            "pm25":        round(float(row['PT08.S2(NMHC)']) / 20, 1),
            "no2":         round(float(row['NO2(GT)']), 1) if pd.notna(row.get('NO2(GT)')) else round(random.uniform(10,80),1),
            "co2":         round(float(row['CO(GT)']) * 100, 1) if pd.notna(row.get('CO(GT)')) else round(random.uniform(350,600),1),
        })
    return jsonify(result)


@app.route('/api/energy/current')
def get_energy():
    sample = energy_df.sample(n=min(50, len(energy_df)), random_state=random.randint(0, 9999))
    total  = round(float(sample['Global_active_power'].sum()), 2)
    avg_v  = round(float(sample['Voltage'].mean()), 1)
    avg_i  = round(float(sample['Global_intensity'].mean()), 1)
    renewable = round(random.uniform(0.22, 0.38), 2)
    zone_names = ["Commercial District", "Residential", "Industrial", "Government Buildings"]
    weights    = [0.35, 0.30, 0.25, 0.10]
    zones = [{"name": n, "consumption": round(total * w + random.uniform(-2, 2), 1), "status": "High" if w > 0.3 else "Normal"}
             for n, w in zip(zone_names, weights)]
    return jsonify({"total_consumption_mwh": total, "renewable_ratio": renewable,
                    "avg_voltage": avg_v, "avg_intensity": avg_i,
                    "peak_demand_time": "18:00 - 20:00", "zones": zones})


@app.route('/api/water/current')
def get_water():
    sample = water_df.sample(n=min(10, len(water_df)), random_state=random.randint(0, 9999))
    avg_pc = round(float(sample['per_capita_real'].mean()), 1)
    scarcity = float(sample['Water_Scarcity_Level'].mean())
    reservoir = round(max(20, min(95, 85 - scarcity * 12 + random.uniform(-5, 5))), 1)
    agri  = round(float(sample['Agricultural_Water_Use_Percent'].mean()) * 40 + 60, 1)
    ind   = round(float(sample['Industrial_Water_Use_Percent'].mean()) * 15 + 20, 1)
    house = round(max(5, 100 - agri - ind + random.uniform(-2, 2)), 1)
    zones = [{"name": z, "usage": round(avg_pc * random.uniform(0.8, 1.2), 1), "pressure_bar": round(random.uniform(2.5, 5.0), 1)}
             for z in WATER_ZONES]
    return jsonify({"daily_consumption_mld": round(avg_pc * 0.0003, 1), "per_capita_lpd": avg_pc,
                    "reservoir_level_pct": reservoir, "leakage_detected": scarcity > 1.5,
                    "quality_index": round(random.uniform(82, 98), 1),
                    "agricultural_pct": agri, "industrial_pct": ind, "household_pct": house, "zones": zones})


@app.route('/api/alerts')
def get_alerts():
    traffic_data = get_traffic().get_json()
    env_data     = get_environment().get_json()
    alerts = []
    now = datetime.now()
    for t in traffic_data:
        if t['congestion_level'] > 0.75:
            alerts.append({"type": "traffic", "severity": "critical",
                "message": f"Severe congestion at {t['junction_name']} — {t['avg_speed']} km/h avg speed",
                "timestamp": (now - timedelta(minutes=random.randint(1,8))).isoformat(), "location": t['junction_name']})
        elif t['congestion_level'] > 0.55:
            alerts.append({"type": "traffic", "severity": "high",
                "message": f"Moderate congestion at {t['junction_name']} ({int(t['congestion_level']*100)}%)",
                "timestamp": (now - timedelta(minutes=random.randint(5,25))).isoformat(), "location": t['junction_name']})
    for e in env_data:
        if e['aqi'] > 200:
            alerts.append({"type": "environment", "severity": "critical",
                "message": f"Hazardous air quality in {e['location']} — AQI {e['aqi']}",
                "timestamp": (now - timedelta(minutes=random.randint(3,15))).isoformat(), "location": e['location']})
        elif e['aqi'] > 150:
            alerts.append({"type": "environment", "severity": "high",
                "message": f"Unhealthy air in {e['location']} — AQI {e['aqi']}",
                "timestamp": (now - timedelta(minutes=random.randint(10,40))).isoformat(), "location": e['location']})
    alerts.append({"type": "maintenance", "severity": "medium",
        "message": "Scheduled water grid maintenance in Sector 7 tonight 23:00–01:00",
        "timestamp": (now - timedelta(hours=1)).isoformat(), "location": "Sector 7"})
    alerts.sort(key=lambda a: {"critical": 0, "high": 1, "medium": 2}[a["severity"]])
    return jsonify(alerts[:8])


@app.route('/api/predictions/traffic')
def get_predictions():
    now = datetime.now()
    preds = []
    for i in range(1, 7):
        future = now + timedelta(hours=i)
        hour   = future.hour
        h_rows = traffic_df[traffic_df['Hour'] == hour] if 'Hour' in traffic_df.columns and len(traffic_df[traffic_df['Hour'] == hour]) > 0 else traffic_df
        row    = h_rows.sample(1, random_state=random.randint(0, 9999)).iloc[0]
        preds.append({"time": future.strftime("%H:%M"), "congestion": round(float(row['congestion_level']), 2),
                      "expected_volume": int(row['vehicle_count_real']), "confidence": round(random.uniform(0.78, 0.95), 2)})
    return jsonify(preds)


@app.route('/api/waste/current')
def get_waste():
    sample = pop_df.sample(n=min(4, len(pop_df)), random_state=random.randint(0, 9999))
    result = []
    for i, (_, row) in enumerate(sample.iterrows()):
        swm  = row.get('Number of Villages Covered with SWM', 100)
        both = row.get('Number of Villages Covered with Both SWM and LWM', 100)
        fill = int(min(95, max(15, 100 - (both / (swm + 1)) * 80 + random.randint(-5, 5))))
        result.append({"zone": WASTE_ZONES[i], "fill_level": fill,
                       "last_pickup": (datetime.now() - timedelta(hours=random.randint(2, 36))).strftime("%Y-%m-%d %H:%M"),
                       "status": "Overdue" if fill > 85 else "Full" if fill > 70 else "OK"})
    return jsonify(result)


@app.route('/api/traffic/hourly')
def get_traffic_hourly():
    if 'Hour' not in traffic_df.columns:
        return jsonify([])
    hourly = traffic_df.groupby('Hour')['congestion_level'].mean().reset_index()
    return jsonify([{"hour": f"{int(r['Hour']):02d}:00", "congestion": round(float(r['congestion_level']) * 100, 1)}
                    for _, r in hourly.iterrows()])


@app.route('/api/environment/trend')
def get_env_trend():
    sample = air_df[['Datetime', 'aqi_scaled', 'T', 'RH']].dropna().tail(24)
    return jsonify([{"time": str(r['Datetime'])[-8:-3], "aqi": int(r['aqi_scaled']),
                     "temperature": round(float(r['T']), 1), "humidity": round(float(r['RH']), 1)}
                    for _, r in sample.iterrows()])


@app.route('/api/energy/trend')
def get_energy_trend():
    sample = energy_df[['DateTime', 'Global_active_power', 'Sub_metering_1',
                         'Sub_metering_2', 'Sub_metering_3']].dropna().tail(24)
    return jsonify([{"time": str(r['DateTime'])[-8:-3], "power": round(float(r['Global_active_power']), 2),
                     "s1": float(r['Sub_metering_1']), "s2": float(r['Sub_metering_2']), "s3": float(r['Sub_metering_3'])}
                    for _, r in sample.iterrows()])


if __name__ == '__main__':
    print("\n✅  Smart City API → http://localhost:5000\n")
    app.run(debug=True, port=5000)
