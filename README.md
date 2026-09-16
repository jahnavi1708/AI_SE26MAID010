# 🌍 AQI AI Intelligence Analyst & Calculation Pipeline

An end-to-end data science and generative AI application that processes air quality datasets, computes robust Air Quality Index (AQI) metrics and categorization buckets dynamically from raw pollutant levels, and provides an interactive AI chat interface powered by **LangChain**, **Pandas Agents**, and **Groq**.

---

## 🚀 Project Architecture & File Structure

```text
├── Data/
│   ├── raw/                      # Raw input CSV datasets
│   └── processed/
│       └── air_quality_cleaned.csv # Cleaned and pivoted dataset
├── outputs/
│   ├── city_summary.csv          # City-wise aggregated AQI metrics
│   ├── station_summary.csv       # Station-wise granular metrics
│   └── pollutant_statistics.csv  # Statistical breakdowns per pollutant
├── Preprocessing.py              # Data cleaning and pivoting pipeline
├── DataAnalysis.py               # AQI computation and summary generator
├── aqi_agent.py                  # Standalone terminal AI agent
├── app.py                        # Streamlit web application & chat UI
└── README.md                     # Project documentation

Key Features
1.Data-Driven Processing: Automatically cleans raw air quality measurements, normalizes columns, and pivots long-format station readings into wide-format multi-pollutant records.

2.Dynamic AQI & Categorization: Computes peak pollution index and assigns qualitative category buckets (Good, Moderate, Poor, Severe) using statistical distribution percentiles directly from your dataset.

3.Groq-Powered AI Analyst: Leverages lightning-fast LLMs (llama-3.3-70b-versatile or openai/gpt-oss-20b) via LangChain's Pandas DataFrame Agent to answer natural language queries about pollution levels, city rankings, and station anomalies.

4.Streamlit Control Center: Interactive web UI featuring sidebar metrics, preset quick-analysis prompts, and chat history.

Installation & Setup
1. Clone or Open the Project Directory
Make sure your project workspace contains your dataset inside Data/raw/ or the project root.

2. Install Required Dependencies
Run the following command in your terminal to install the necessary Python libraries:

```bash
pip install streamlit pandas numpy langchain langchain-experimental langchain-groq

Configure API Key
Ensure your Groq API key is configured in your scripts or environment variables:

Python
os.environ["GROQ_API_KEY"] = "your-groq-api-key-here"

How to Run the Application
Step 1: Preprocess the Raw Dataset
Clean and pivot your raw CSV file into the processed directory:

```bash
python Preprocessing.py
Step 2: Run Data Analysis & Generate Summaries
Compute calculated AQI scores, category buckets, and summary CSVs into the outputs/ folder:

```bash
python DataAnalysis.py
Step 3: Launch the Streamlit Web Application
Start the interactive dashboard and AI chat analyst:

```bash
streamlit run app.py

Sample Queries You Can Ask the AI Agent
“What are the average AQI scores and dominant category buckets across different cities?”

“Which monitoring stations record the highest mean AQI and maximum AQI?”

“Show me the statistical breakdowns (mean, median, max) for PM2.5 and PM10.”

“Which city has the best air quality in the dataset?”
