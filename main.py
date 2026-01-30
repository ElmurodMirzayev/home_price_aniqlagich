import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🏠 Flat Price Prediction")

total_area = st.number_input("Total area (m²)", min_value=10.0, value=108.0)
rooms = st.number_input("Rooms", min_value=1, step=1, value=3)
ceiling_height = st.number_input("Ceiling height (m)", value=2.7)
floors_total = st.number_input("Total floors", min_value=1, step=1, value=16)
living_area = st.number_input("Living area (m²)", value=51.0)
floor = st.number_input("Floor", min_value=1, step=1, value=8)
kitchen_area = st.number_input("Kitchen area (m²)", value=25.0)
airports_nearest = st.number_input("Nearest airport distance (m)", value=18863)
cityCenters_nearest = st.number_input("City center distance (m)", value=16028)
parks_around3000 = st.number_input("Parks within 3km", min_value=0, step=1, value=1)
ponds_around3000 = st.number_input("Ponds within 3km", min_value=0, step=1, value=2)

new_flat = pd.DataFrame([{
    "total_area": total_area,
    "rooms": rooms,
    "ceiling_height": ceiling_height,
    "floors_total": floors_total,
    "living_area": living_area,
    "floor": floor,
    "kitchen_area": kitchen_area,
    "airports_nearest": airports_nearest,
    "cityCenters_nearest": cityCenters_nearest,
    "parks_around3000": parks_around3000,
    "ponds_around3000": ponds_around3000
}])

if st.button("🔮 Predict price"):
    new_flat_scaled = scaler.transform(new_flat)  # ✅ TO‘G‘RI
    prediction = model.predict(new_flat_scaled)
    st.success(f"Estimated price: {int(prediction[0]):,} рубыл")
