
import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


st.set_page_config(page_title="Demand Forecasting App", layout="wide")

st.title("📊 Demand Forecasting (Units Sold Prediction)")
st.write("Predict **Units Sold** for a specific store-product-date using your trained ML model.")

# -----------------------------------
# Sidebar Inputs (same as dataset)
# -----------------------------------
st.sidebar.header("🔧 Input Features")

date = st.sidebar.date_input("Date")

store_id = st.sidebar.selectbox(
    "Store ID",
    ["S001", "S002", "S003", "S004", "S005"]
)

product_id = st.sidebar.text_input("Product ID", value="P0001")

category = st.sidebar.selectbox(
    "Category",
    ["Groceries", "Toys", "Electronics", "Clothing", "Furniture", "Other"]
)

region = st.sidebar.selectbox(
    "Region",
    ["North", "South", "East", "West"]
)

inventory_level = st.sidebar.number_input("Inventory Level", min_value=0, value=200, step=1)

units_ordered = st.sidebar.number_input("Units Ordered", min_value=0, value=50, step=1)

price = st.sidebar.number_input("Price", min_value=0.0, value=50.0, step=1.0)

discount = st.sidebar.number_input("Discount", min_value=0, value=10, step=1)

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["Sunny", "Rainy", "Cloudy", "Snowy"]
)

holiday_promo = st.sidebar.selectbox(
    "Holiday/Promotion",
    [0, 1]
)

competitor_pricing = st.sidebar.number_input("Competitor Pricing", min_value=0.0, value=45.0, step=1.0)

seasonality = st.sidebar.selectbox(
    "Seasonality",
    ["Summer", "Winter", "Autumn", "Spring"]
)

# Lag feature input (required by your current pipeline)
st.sidebar.subheader("📌 Previous Demand Info")
prev_units_sold = st.sidebar.number_input(
    "Previous Units Sold (prev_Units Sold)",
    min_value=0.0,
    value=100.0,
    step=1.0
)

# -----------------------------------
# Main Layout
# -----------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Data Preview")

    extra_features = {
        "Store ID": store_id,
        "Product ID": product_id,
        "Category": category,
        "Region": region,
        "Inventory Level": inventory_level,
        "Units Ordered": units_ordered,
        "Price": price,
        "Discount": discount,
        "Weather Condition": weather_condition,
        "Holiday/Promotion": holiday_promo,
        "Competitor Pricing": competitor_pricing,
        "Seasonality": seasonality
    }

    data_obj = CustomData(
        date=str(date),
        prev_target_value=float(prev_units_sold),
        extra_features=extra_features,
        target_column_name="Units Sold"
    )

    input_df = data_obj.get_data_as_data_frame()
    st.dataframe(input_df, use_container_width=True)

with col2:
    st.subheader("✅ Prediction Result")

    if st.button("Predict Units Sold 🚀", use_container_width=True):
        try:
            pipeline = PredictPipeline(target_column_name="Units Sold")
            preds = pipeline.predict(input_df)

            prediction = float(preds[0])

            st.success(f"📌 Predicted Units Sold on {date}: **{prediction:.2f}** units")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# -----------------------------------
# Optional: CSV Upload (for checking row)
# -----------------------------------
st.markdown("---")
st.subheader("📂 Optional: Upload CSV for single-row prediction (same dataset format)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview uploaded dataset:")
    st.dataframe(df.head(), use_container_width=True)

    st.info("Select 1 row from your dataset to predict.")

    row_index = st.number_input("Row Index", min_value=0, max_value=len(df)-1, value=0, step=1)

    selected_row = df.iloc[[row_index]].copy()

    # Create prev_Units Sold automatically if Units Sold exists
    if "Units Sold" in df.columns:
        selected_row["prev_Units Sold"] = df["Units Sold"].shift(1).fillna(method="bfill").iloc[row_index]

    st.write("Selected row input:")
    st.dataframe(selected_row, use_container_width=True)

    if st.button("Predict for Selected Row ✅"):
        try:
            pipeline = PredictPipeline(target_column_name="Units Sold")
            preds = pipeline.predict(selected_row)
            st.success(f"✅ Predicted Units Sold: **{float(preds[0]):.2f}**")
        except Exception as e:
            st.error(str(e))
