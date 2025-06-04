# flask_api_copy/flask_api - Copy/app.py

from flask import Flask
from routes.api import api
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.register_blueprint(api, url_prefix="/api")

@app.route("/", methods=["GET"])
def home():
    return "✅ Server is up and running!", 200

if __name__ == "__main__":
    # Host va portni belgilaymiz
    app.run(debug=True, host="0.0.0.0",port=5000)
