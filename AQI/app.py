import os
from pathlib import Path
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)
from langchain_groq import ChatGroq
import numpy as np
import pandas as pd
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="AQI AI Intelligence Analyst",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Configuration ---
os.environ["GROQ_API_KEY"] = (
    "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"
)


@st.cache_resource
def init_agent():
  try:
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUTS_DIR = BASE_DIR / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    city_sum_path = OUTPUTS_DIR / "city_summary.csv"
    station_sum_path = OUTPUTS_DIR / "station_summary.csv"
    pollutant_stats_path = OUTPUTS_DIR / "pollutant_statistics.csv"

    # Generate summary files if they don't exist yet
    if not (
        city_sum_path.exists()
        and station_sum_path.exists()
        and pollutant_stats_path.exists()
    ):
      cleaned_file = BASE_DIR / "Data" / "processed" / "air_quality_cleaned.csv"
      if not cleaned_file.exists():
        raise FileNotFoundError(
            f"Could not find cleaned dataset at {cleaned_file}"
        )

      df = pd.read_csv(cleaned_file)

      pollutant_cols = [
          c
          for c in ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]
          if c in df.columns
      ]
      df["Calculated_AQI"] = (
          df[pollutant_cols].max(axis=1) if pollutant_cols else 100.0
      )

      p33, p66, p90 = df["Calculated_AQI"].quantile([0.33, 0.66, 0.90])

      def categorize_aqi(val):
        if pd.isna(val):
          return "Unknown"
        elif val <= p33:
          return "Good"
        elif val <= p66:
          return "Moderate"
        elif val <= p90:
          return "Poor"
        else:
          return "Severe"

      df["AQI_Bucket"] = df["Calculated_AQI"].apply(categorize_aqi)

      city_col = next(
          (c for c in ["city", "City", "state", "State"] if c in df.columns),
          None,
      )
      if city_col:
        city_summary = (
            df.groupby(city_col)
            .agg(
                Record_Count=("Calculated_AQI", "count"),
                Mean_AQI=("Calculated_AQI", "mean"),
                Max_AQI=("Calculated_AQI", "max"),
                Min_AQI=("Calculated_AQI", "min"),
            )
            .reset_index()
        )
        city_summary["Dominant_Bucket"] = city_summary["Mean_AQI"].apply(
            categorize_aqi
        )
        city_summary.to_csv(city_sum_path, index=False)

      station_col = next(
          (
              c
              for c in ["station", "Station", "location", "Location"]
              if c in df.columns
          ),
          None,
      )
      if station_col:
        station_summary = (
            df.groupby(station_col)
            .agg(
                Record_Count=("Calculated_AQI", "count"),
                Mean_AQI=("Calculated_AQI", "mean"),
                Max_AQI=("Calculated_AQI", "max"),
            )
            .reset_index()
        )
        station_summary["AQI_Bucket"] = station_summary["Mean_AQI"].apply(
            categorize_aqi
        )
        station_summary.to_csv(station_sum_path, index=False)

      if pollutant_cols:
        pollutant_stats = (
            df[pollutant_cols]
            .agg(["mean", "median", "max", "min", "std"])
            .T.reset_index()
        )
        pollutant_stats.columns = [
            "Pollutant",
            "Mean",
            "Median",
            "Max",
            "Min",
            "StdDev",
        ]
        pollutant_stats.to_csv(pollutant_stats_path, index=False)

    # Load summaries into agent
    city_summary = pd.read_csv(city_sum_path)
    station_summary = pd.read_csv(station_sum_path)
    pollutant_stats = pd.read_csv(pollutant_stats_path)

    # Use active Groq production models (e.g., openai/gpt-oss-20b or llama-3.3-70b-versatile)
    models_to_try = [
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    llm = None
    last_err = None

    for model_name in models_to_try:
      try:
        test_llm = ChatGroq(model=model_name, temperature=0)
        llm = test_llm
        break
      except Exception as ex:
        last_err = ex
        continue

    if llm is None:
      raise RuntimeError(f"Could not initialize any Groq model: {last_err}")

    agent = create_pandas_dataframe_agent(
        llm,
        [city_summary, station_summary, pollutant_stats],
        verbose=False,
        agent_type="openai-tools",
        allow_dangerous_code=True,
        prefix="""You are an expert Air Quality Data Analyst. 
            You have access to three dataframes derived directly from the dataset:
            - df1: city_summary (City-wise aggregated statistics including Mean_AQI, Max_AQI, Min_AQI, and Dominant_Bucket)
            - df2: station_summary (Granular data per monitoring station including Mean_AQI and AQI_Bucket)
            - df3: pollutant_statistics (Statistical breakdowns for individual pollutants like PM2.5, PM10, NO2, SO2, CO, OZONE, NH3)
            
            Answer the user's questions accurately by querying these dataframes. Keep your answers concise, informative, and professional.""",
    )
    return agent, city_summary, station_summary, pollutant_stats, None
  except Exception as e:
    return None, None, None, None, str(e)


agent, city_summary, station_summary, pollutant_stats, init_error = init_agent()

# --- Sidebar Control Center ---
with st.sidebar:
  st.image("https://img.icons8.com/color/96/earth-planet.png", width=56)
  st.title("🌍 AQI Control Center")
  st.markdown("---")

  st.markdown("### 📊 Dataset Overview")
  if city_summary is not None:
    st.metric("Total Cities", len(city_summary))
  if station_summary is not None:
    st.metric("Monitoring Stations", len(station_summary))
  if pollutant_stats is not None:
    st.metric("Tracked Pollutants", len(pollutant_stats))

  st.markdown("---")
  st.markdown("### 💡 Quick Analysis Prompts")
  if st.button("🌆 City-wise AQI Ranking"):
    st.session_state.preset_prompt = (
        "What are the average AQI scores and dominant category buckets across"
        " different cities?"
    )
  if st.button("🏭 Top Polluted Stations"):
    st.session_state.preset_prompt = (
        "Which monitoring stations record the highest mean AQI and maximum AQI?"
    )
  if st.button("📈 Pollutant Statistics"):
    st.session_state.preset_prompt = (
        "Show me the statistical breakdowns (mean, median, max) for PM2.5, PM10,"
        " and other pollutants."
    )

  st.markdown("---")
  st.markdown("*Powered by Groq & Streamlit*")

# --- Main Chat Interface ---
st.title("🌍 Air Quality Index (AQI) AI Intelligence Analyst")
st.markdown(
    "Analyze dataset-derived AQI metrics, city rankings, station summaries,"
    " and pollutant breakdowns interactively."
)

if init_error:
  st.error(f"Initialization Error: {init_error}")

if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
  default_input = st.session_state.preset_prompt
  st.session_state.preset_prompt = ""
else:
  default_input = ""

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

prompt = st.chat_input(
    "Ask about city AQI rankings, station summaries, or pollutant stats..."
)

if not prompt and default_input:
  prompt = default_input

if prompt:
  with st.chat_message("user"):
    st.markdown(prompt)
  st.session_state.messages.append({"role": "user", "content": prompt})

  with st.chat_message("assistant"):
    if agent:
      with st.spinner("Analyzing dataset metrics..."):
        try:
          response = agent.invoke({"input": prompt})
          answer = response["output"]
          st.markdown(answer)
          st.session_state.messages.append(
              {"role": "assistant", "content": answer}
          )
        except Exception as e:
          st.error(f"Error processing query: {e}")
    else:
      st.error("Agent is not initialized. Check the error message above.")