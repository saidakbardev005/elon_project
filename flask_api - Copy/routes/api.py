# flask_api_copy/flask_api - Copy/routes/api.py

from flask import Blueprint, request, jsonify
import pandas as pd
from transliteration.latin_to_cyrillic import latin_to_cyrillic
from services.predict_service import get_coordinates, predict_price, find_best_drivers
from sklearn.preprocessing import LabelEncoder
import os

api = Blueprint("api", __name__)

def is_cyrillic(text: str) -> bool:
    """
    Matn ichida kirilcha harf mavjud bo'lsa, True qaytaradi.
    (Agar foydalanuvchi allaqachon kirilcha kiritsa, 
    qoʻshimcha transliteratsiya qilinmasin.)
    """
    for ch in text:
        if ('А' <= ch <= 'я') or (ch in ['Ё', 'ё']):
            return True
    return False

@api.route("/predict", methods=["GET", "POST"])
def predict_route():
    try:
        # ---------------------------------------------------
        # 1) GET yoki POST so‘rovini ajratib olamiz
        # ---------------------------------------------------
        if request.method == "POST":
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid JSON"}), 400

            from_city = data.get("from")
            to_city   = data.get("to")
            weight    = data.get("weight")
            volume    = data.get("volume")

            if from_city is None or to_city is None or weight is None or volume is None:
                return jsonify({
                    "error": "Missing required parameters: 'from', 'to', 'weight', 'volume'"
                }), 400

        else:
            from_city = request.args.get("from")
            to_city   = request.args.get("to")
            weight    = request.args.get("weight")
            volume    = request.args.get("volume")

            if from_city is None or to_city is None or weight is None or volume is None:
                return jsonify({
                    "error": "Missing required query parameters: 'from', 'to', 'weight', 'volume'"
                }), 400

        # ---------------------------------------------------
        # 2) “weight” va “volume” son ekanligini tekshiramiz
        # ---------------------------------------------------
        try:
            weight = float(weight)
            volume = float(volume)
        except (ValueError, TypeError):
            return jsonify({"error": "Parameters 'weight' and 'volume' must be numbers"}), 400

        # ---------------------------------------------------
        # 3) Shahar nomlarining kiril/lotin holatini normallashtiramiz
        # ---------------------------------------------------
        try:
            # Agar allaqachon kirilcha yozilgan bo‘lsa:
            if is_cyrillic(from_city):
                f_cyr = from_city.strip().capitalize()
            else:
                f_cyr = latin_to_cyrillic(from_city.strip()).capitalize()

            if is_cyrillic(to_city):
                t_cyr = to_city.strip().capitalize()
            else:
                t_cyr = latin_to_cyrillic(to_city.strip()).capitalize()

        except Exception as e:
            return jsonify({"error": f"Transliteration failed: {str(e)}"}), 400

        # ---------------------------------------------------
        # 4) direction_prices.csv dan “From” va “To” ustunlarini o‘qiymiz
        #    va LabelEncoder bilan kodlaymiz
        # ---------------------------------------------------
        try:
            base_dir = os.path.abspath(os.path.dirname(__file__))   # .../routes
            data_dir = os.path.join(base_dir, os.pardir, "data")    # .../data

            # direction_prices.csv faylini o‘qiymiz
            direction_prices_path = os.path.join(data_dir, "direction_prices.csv")
            df_price = pd.read_csv(direction_prices_path)

            # Biz CSV ichidagi “From” va “To” ustunlarini ham strip() + capitalize() qilamiz
            df_price["From"] = df_price["From"].astype(str).str.strip().str.capitalize()
            df_price["To"]   = df_price["To"].astype(str).str.strip().str.capitalize()

            le_from = LabelEncoder()
            le_to   = LabelEncoder()
            le_from.fit(df_price["From"])
            le_to.fit(df_price["To"])
        except FileNotFoundError as e:
            return jsonify({"error": f"CSV file not found: {str(e)}"}), 500
        except KeyError as e:
            return jsonify({"error": f"Expected column not found in CSV: {str(e)}"}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to load or encode direction_prices.csv: {str(e)}"}), 500

        # Agar shahar nomlari LabelEncoder sinflarida bo‘lmasa:
        if f_cyr not in le_from.classes_ or t_cyr not in le_to.classes_:
            return jsonify({"error": "Invalid city names"}), 400

        f_enc = le_from.transform([f_cyr])[0]
        t_enc = le_to.transform([t_cyr])[0]

        # ---------------------------------------------------
        # 5) Narxni bashorat qilamiz
        # ---------------------------------------------------
        try:
            predicted_price = predict_price(f_enc, t_enc)
        except Exception as e:
            return jsonify({"error": f"Price prediction failed: {str(e)}"}), 500

        # ---------------------------------------------------
        # 6) “from_city” manzilini koordinatalarga aylantiramiz
        # ---------------------------------------------------
        lat, lon = get_coordinates(from_city)
        if lat is None or lon is None:
            return jsonify({"error": "Location not found"}), 400

        # ---------------------------------------------------
        # 7) Eng yaxshi haydovchilarni tanlaymiz (CSV asosida)
        # ---------------------------------------------------
        try:
            best_drivers_list = find_best_drivers(lat, lon, weight, volume)
        except Exception as e:
            return jsonify({"error": f"Driver selection failed: {str(e)}"}), 500

        # ---------------------------------------------------
        # 8) Oxirgi javobni qaytaramiz
        # ---------------------------------------------------
        return jsonify({
            "predicted_price": predicted_price,
            "drivers": best_drivers_list
        }), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500
