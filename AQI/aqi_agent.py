import os
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)
from langchain_groq import ChatGroq
import pandas as pd

# Set your Groq API key here or via environment variables
os.environ["GROQ_API_KEY"] = (
    "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"
)


def create_aqi_agent():
  try:
    # Load the summary outputs generated directly from your cleaned dataset
    city_summary = pd.read_csv("outputs/city_summary.csv")
    station_summary = pd.read_csv("outputs/station_summary.csv")
    pollutant_stats = pd.read_csv("outputs/pollutant_statistics.csv")
  except FileNotFoundError as e:
    print(f"Error loading summary files. Please run DataAnalysis.py first: {e}")
    return None

  # Initialize the LLM using Groq (fast and efficient for data analysis)
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

  # Create the Pandas DataFrame Agent
  agent_executor = create_pandas_dataframe_agent(
      llm,
      [city_summary, station_summary, pollutant_stats],
      verbose=True,
      agent_type="openai-tools",
      allow_dangerous_code=True,
      prefix="""You are an expert Air Quality Data Analyst agent. 
        You have been provided with three dataframes derived directly from the user's cleaned air quality dataset:
        - df1: city_summary (City-wise aggregated metrics including Mean_AQI, Max_AQI, Min_AQI, and Dominant_Bucket)
        - df2: station_summary (Granular data per monitoring station including Mean_AQI and AQI_Bucket)
        - df3: pollutant_statistics (Statistical breakdowns like mean, median, max, min, and std for individual pollutants like PM2.5, PM10, NO2, etc.)
        
        Always query these dataframes accurately to answer user questions with concise, informative, and professional insights.""",
  )

  return agent_executor


if __name__ == "__main__":
  agent = create_aqi_agent()

  if agent:
    print("\n🤖 AQI AI Intelligence Analyst is online!")
    print("Ask me anything about your air quality dataset (type 'quit' to exit).")

    while True:
      user_input = input("\nQuery: ")
      if user_input.lower() in ["quit", "exit"]:
        break

      try:
        response = agent.invoke({"input": user_input})
        print(f"\n🤖 Answer: {response['output']}")
      except Exception as e:
        print(f"\n🤖 An error occurred while reasoning: {e}")