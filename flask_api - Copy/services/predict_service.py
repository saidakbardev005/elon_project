# flask_api_copy/flask_api - Copy/services/predict_service.py

import os
import joblib
import numpy as np
import googlemaps
import pandas as pd
import warnings
from config import Config
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# “feature names” bilan bog‘liq warning’ni susaytiramiz (ixtiyoriy)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Loyihaning ildiz papkasini aniqlash (…/flask_api)
base_dir = os.path.abspath(os.path.dirname(__file__))        
project_root = os.path.abspath(os.path.join(base_dir, os.pardir))

# —————————————————————————————————————————————————————————————
# 1) Modellarni yuklash (project ildizida pkl fayllar joylashgan deb faraz qilamiz)
# —————————————————————————————————————————————————————————————
kmeans_path      = os.path.join(project_root, "kmeans_model.pkl")
scaler_path      = os.path.join(project_root, "scaler.pkl")
price_model_path = os.path.join(project_root, "price_predictor.pkl")

kmeans      = joblib.load(kmeans_path)
scaler      = joblib.load(scaler_path)
price_model = joblib.load(price_model_path)

# Google Maps Client (geo‐kodlash uchun)
gmaps = googlemaps.Client(key=Config.GOOGLE_MAPS_API_KEY)

# —————————————————————————————————————————————————————————————
# 2) CSV fayllarni yuklash funksiyasi
# —————————————————————————————————————————————————————————————
def load_csv_data():
    """
    data/ papkadan quyidagi CSV’larni o‘qib, pandas DataFrame qaytaradi:
      - direction_prices.csv
      - driver_locations.csv
      - my_autos.csv
      - users.csv
    """
    data_dir = os.path.join(project_root, "data")

    price_csv_path   = os.path.join(data_dir, "direction_prices.csv")
    drivers_csv_path = os.path.join(data_dir, "driver_locations.csv")
    autos_csv_path   = os.path.join(data_dir, "my_autos.csv")
    users_csv_path   = os.path.join(data_dir, "users.csv")

    df_price   = pd.read_csv(price_csv_path)
    drivers_df = pd.read_csv(drivers_csv_path)
    my_autos   = pd.read_csv(autos_csv_path)
    users      = pd.read_csv(users_csv_path)

    return df_price, drivers_df, my_autos, users

# —————————————————————————————————————————————————————————————
# 3) Manzilni koordinatalarga aylantirish (Google Geocoding)
# —————————————————————————————————————————————————————————————
def get_coordinates(location: str):
    """
    location: matn ko‘rinishida (masalan, "Toshkent" yoki "Fargona").
    Agar Google API orqali koordinata topilsa, (lat, lon) qaytaradi.
    Aks holda (None, None) qaytaradi.
    """
    try:
        result = gmaps.geocode(location)
        if result and len(result) > 0:
            loc = result[0]["geometry"]["location"]
            return loc["lat"], loc["lng"]
    except Exception:
        pass
    return None, None

# —————————————————————————————————————————————————————————————
# 4) Narxni bashorat qilish
# —————————————————————————————————————————————————————————————
def predict_price(f_enc: int, t_enc: int) -> int:
    """
    f_enc, t_enc: LabelEncoder orqali kodlangan kiruvchi (from) va chiquvchi (to) shahar indekslari.
    price_model.predict([[f_enc, t_enc]]) natijasidan butun son sifatida 3 ta nol qo‘shish.
    """
    price_raw = price_model.predict([[f_enc, t_enc]])[0]
    return int(str(int(price_raw)) + "000")

# —————————————————————————————————————————————————————————————
# 5) Eng yaxshi haydovchilarni topish (faqat weight/volume asosida KMeans bilan klasterlaydi)
# —————————————————————————————————————————————————————————————
def find_best_drivers(lat: float, lon: float, weight: float, volume: float):
    """
    1) load_csv_data() chaqirib, barcha CSV fayllarni o‘qiydi.
    2) driver_locations.csv + my_autos.csv + users.csv jadvalini birlashtiradi.
    3) “transport_weight” va “transport_volume” ustunlari asosida KMeans klastering qiladi.
    4) Kiruvchi (weight, volume) nuqtani ham shu KMeans modeliga yuklaydi va 
       ularga mos cluster_id oladi.
    5) Ushbu cluster ichidagi haydovchilar ro‘yxatini “same_cluster” deb oladi.
    6) Har bir haydovchida “capacity_distance” (transport_capacity/hajm bo‘yicha masofa) 
       ni hisoblaydi:  
         capacity_distance = sqrt((transport_weight – weight)² + (transport_volume – volume)²)
    7) “distance_km” ni ham hisoblaydi (joylashuv bo‘yicha; 1° ≈ 111 km).
    8) capacity_distance birinchi, distance_km ikkinchi mezon bo‘yicha saralaydi 
       va top 5 ni qaytaradi.
    9) Agar shu cluster bo‘sh boʻlsa, [] qaytaradi.
    """

    # 5.1) CSV fayllarni yuklaymiz
    df_price, drivers_df, my_autos, users = load_csv_data()

    # 5.2) Haydovchilar + avtolar + foydalanuvchilar jadvalini birlashtiramiz
    try:
        drivers_merged = drivers_df.merge(my_autos, on="user_id")
        drivers_merged = drivers_merged.merge(
            users[["user_id", "fullname", "phone", "status"]],
            on="user_id"
        )
    except KeyError as e:
        raise RuntimeError(f"Expected column not found during merge: {e}")

    # 5.3) “transport_weight” va “transport_volume” ustunlarini numeric`ga oʻtkazamiz
    drivers_merged["transport_weight"] = pd.to_numeric(
        drivers_merged["transport_weight"], errors="coerce"
    )
    drivers_merged["transport_volume"] = pd.to_numeric(
        drivers_merged["transport_volume"], errors="coerce"
    )
    # NaN bo‘lsa, o‘chirib tashlaymiz
    drivers_merged = drivers_merged.dropna(subset=["transport_weight", "transport_volume"])

    # 5.4) Weight/volume array (N x 2) yaratamiz
    weight_volume_array = drivers_merged[["transport_weight", "transport_volume"]].values

    # 5.5) KMeans modelini dinamik yaratamiz (masalan, 4 cluster,
    #     lekin agar haydovchilar soni kichik boʻlsa, n_clusters ≤ n_drivers)
    n_drivers = weight_volume_array.shape[0]
    n_clusters = min(4, n_drivers) if n_drivers > 1 else 1

    kmeans_local = KMeans(n_clusters=n_clusters, random_state=42)
    if n_drivers > 0:
        kmeans_local.fit(weight_volume_array)

    # 5.6) Kiruvchi nuqtani cluster ga joylaymiz
    input_feat = np.array([[weight, volume]])
    input_cluster_id = int(kmeans_local.predict(input_feat)[0])

    # 5.7) Shu cluster ichidagi haydovchilar qatorini olamiz
    drivers_merged["cluster_wv"] = kmeans_local.labels_
    same_cluster = drivers_merged[drivers_merged["cluster_wv"] == input_cluster_id].copy()

    # 5.8) Agar bu cluster bo‘sh boʻlsa, [] qaytaramiz
    if same_cluster.empty:
        return []

    # 5.9) Har bir haydovchi uchun capacity_distance hisoblaymiz
    same_cluster["capacity_distance"] = np.sqrt(
        (same_cluster["transport_weight"] - weight) ** 2 +
        (same_cluster["transport_volume"] - volume) ** 2
    )

    # 6) “distance_km” ni ham hisoblaymiz (taxminan: 1° lat/lon ≈ 111 km)
    same_cluster["distance_km"] = np.sqrt(
        (same_cluster["latitude"] - lat) ** 2 +
        (same_cluster["longitude"] - lon) ** 2
    ) * 111

    # 7) Saralaymiz: capacity_distance birinchi, distance_km ikkinchi
    same_cluster = same_cluster.sort_values(
        by=["capacity_distance", "distance_km"],
        ascending=[True, True]
    )

    # 8) Top 5 haydovchini JSON‐formatda qaytaramiz
    result = same_cluster[[
        "fullname",
        "phone",
        "transport_model",
        "transport_weight",
        "transport_volume",
        "distance_km"
    ]].head(5).to_dict(orient="records")

    return result
