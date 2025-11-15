from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    else:
        # -----------------------------
        # 1. Read inputs from HTML form
        # -----------------------------
        date_input = request.form.get("date")                        # Required
        prev_value = request.form.get("prev_value")                 # Required lag value
        prev_value = float(prev_value) if prev_value else None

        # Collect all other dynamic features the user provides
        extra_features = {}
        for key, value in request.form.items():
            if key not in ["date", "prev_value"]:
                extra_features[key] = value

        # -----------------------------
        # 2. Build CustomData object
        # -----------------------------
        data = CustomData(
            prev_target_value=prev_value,
            date=date_input,
            extra_features=extra_features,
            target_column_name="Demand Forecast"
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("\n--- Input DataFrame ---")
        print(pred_df)

        # -----------------------------
        # 3. Run prediction pipeline
        # -----------------------------
        predict_pipeline = PredictPipeline()
        predictions = predict_pipeline.predict(pred_df)

        print("Prediction completed:", predictions)

        return render_template(
            "home.html",
            results=round(predictions[0], 2)  # Display nicely rounded output
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
