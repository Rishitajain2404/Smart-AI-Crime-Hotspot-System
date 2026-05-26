import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
import joblib

# Page Configuration
st.set_page_config(
    page_title="Jaipur AI Crime Analytics",
    page_icon="🚨",
    layout="wide"
)

# Load Dataset
crime_data = pd.read_csv("crime_data.csv")

# Load Model
model = joblib.load("crime_model.pkl")

# Time Conversion
crime_data['Hour'] = pd.to_datetime(
    crime_data['Time']
).dt.hour

# Risk Score
crime_data['Risk_Score'] = (
    crime_data['Crime_Count'] * 2 +
    crime_data['Night_Crime'] * 3 +
    crime_data['Violent_Crime'] * 5
)

# Women Safety Index
crime_data['Women_Safety_Index'] = (
    100 - crime_data['Risk_Score']
)

# Sidebar
st.sidebar.title("🚨 Jaipur Crime AI System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Heatmap",
        "AI Prediction",
        "Safety Analysis"
    ]
)

# Main Dashboard
if menu == "Dashboard":

    st.title(
        "🚨 Smart AI Crime Hotspot Prediction System"
    )

    st.markdown(
        "### Advanced Crime Analytics Dashboard for Jaipur, Rajasthan"
    )

    # Metrics
    total_crimes = crime_data['Crime_Count'].sum()

    high_risk = len(
        crime_data[
            crime_data['Risk_Score'] > 35
        ]
    )

    avg_safety = round(
        crime_data['Women_Safety_Index'].mean(),
        2
    )

    night_crimes = crime_data['Night_Crime'].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Crimes",
        total_crimes
    )

    col2.metric(
        "High Risk Areas",
        high_risk
    )

    col3.metric(
        "Women Safety Score",
        f"{avg_safety}%"
    )

    col4.metric(
        "Night Crimes",
        night_crimes
    )

    st.divider()

    # Crime Distribution
    st.subheader("📊 Crime Distribution by Area")

    fig = px.bar(
        crime_data,
        x='Area',
        y='Crime_Count',
        color='Crime_Count',
        title='Crime Distribution Across Jaipur'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Crime Type Chart
    st.subheader("📈 Crime Type Analysis")

    fig2 = px.pie(
        crime_data,
        names='Crime_Type',
        title='Crime Type Percentage'
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # Hourly Trend
    st.subheader("⏰ Crime Trend by Hour")

    fig3 = px.line(
        crime_data,
        x='Hour',
        y='Crime_Count',
        markers=True,
        title='Hourly Crime Analysis'
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# Heatmap
elif menu == "Heatmap":

    st.title("🗺 Jaipur Crime Hotspot Heatmap")

    crime_map = folium.Map(
        location=[26.9124, 75.7873],
        zoom_start=11
    )

    for index, row in crime_data.iterrows():

        if row['Risk_Score'] > 35:
            color = 'red'

        elif row['Risk_Score'] > 20:
            color = 'orange'

        else:
            color = 'green'

        folium.CircleMarker(
            location=[
                row['Latitude'],
                row['Longitude']
            ],
            radius=12,
            popup=f"""
            Area: {row['Area']}
            Risk Score: {row['Risk_Score']}
            Crime Type: {row['Crime_Type']}
            """,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(crime_map)

    folium_static(
        crime_map,
        width=1400,
        height=700
    )

# AI Prediction
elif menu == "AI Prediction":

    st.title("🤖 Future Crime Prediction")

    st.subheader(
        "Enter Crime Details"
    )

    area = st.number_input(
        "Encoded Area",
        min_value=0
    )

    crime_type = st.number_input(
        "Encoded Crime Type",
        min_value=0
    )

    latitude = st.number_input(
        "Latitude"
    )

    longitude = st.number_input(
        "Longitude"
    )

    crime_count = st.number_input(
        "Crime Count",
        min_value=0
    )

    night_crime = st.selectbox(
        "Night Crime",
        [0, 1]
    )

    violent_crime = st.selectbox(
        "Violent Crime",
        [0, 1]
    )

    hour = st.slider(
        "Hour",
        0,
        23
    )

    risk_score = st.slider(
        "Risk Score",
        0,
        100
    )

    if st.button("Predict Crime Severity"):

        prediction = model.predict([[
            area,
            crime_type,
            latitude,
            longitude,
            crime_count,
            night_crime,
            violent_crime,
            hour,
            risk_score
        ]])

        if prediction[0] == 0:

            st.success(
                "🟢 LOW Crime Severity"
            )

        elif prediction[0] == 1:

            st.warning(
                "🟠 MEDIUM Crime Severity"
            )

        else:

            st.error(
                "🔴 HIGH Crime Severity"
            )

# Safety Analysis
elif menu == "Safety Analysis":

    st.title("🛡 AI Safety Recommendations")

    for index, row in crime_data.iterrows():

        if row['Risk_Score'] > 35:

            st.error(
                f"⚠ Avoid travelling alone in {row['Area']} at night."
            )

        elif row['Risk_Score'] > 20:

            st.warning(
                f"Stay alert in crowded places in {row['Area']}."
            )

        else:

            st.success(
                f"{row['Area']} is relatively safe."
            )

    st.subheader("👩 Women Safety Index")

    st.dataframe(
        crime_data[
            ['Area', 'Women_Safety_Index']
        ]
    )