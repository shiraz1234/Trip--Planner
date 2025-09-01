import streamlit as st
import random
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# =====================
# Initialize Groq client
# =====================
llm = ChatGroq(
    api_key="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",  # 🔑 replace with your key
    model="llama-3.1-8b-instant",
    temperature=0.6
)

# =====================
# Define Chains
# =====================
budget_prompt = PromptTemplate(
    input_variables=["trip"],
    template="""
    I want to travel {trip}.
    Give me a budget which is eco-friendly and cheap.
    """
)
budget_chain = LLMChain(llm=llm, prompt=budget_prompt, output_key="budget")

places_prompt = PromptTemplate(
    input_variables=["budget"],
    template="Suggest some must-visit eco-friendly places in that country, based on this budget: {budget}"
)
places_chain = LLMChain(llm=llm, prompt=places_prompt, output_key="places")

chain = SequentialChain(
    chains=[budget_chain, places_chain],
    input_variables=["trip"],
    output_variables=["budget", "places"],
    verbose=True
)

# =====================
# Mock Flight Generator
# =====================
def get_mock_flights(origin, destination, date):
    base_price = random.randint(250, 600)
    options = [
        f"Economy Non-stop: ${base_price}–${base_price+80}",
        f"Economy 1-stop: ${base_price-50}–${base_price+30}",
        f"Flexi (Changeable): ${base_price+120}–${base_price+200}",
    ]
    return (
        f"✈️ Flight options from **{origin} → {destination}** on **{date}**:\n\n"
        + "\n".join([f"• {opt}" for opt in options])
    )

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Eco Travel Planner", page_icon="🌍", layout="wide")

st.title("🌍 Eco-Friendly Travel Planner")

# Sidebar for inputs
st.sidebar.header("✈️ Plan Your Trip")

# Dropdown for origin & destination
origins = ["DEL", "BOM", "MAA", "BLR"]  # Delhi, Mumbai, Chennai, Bangalore
destinations = ["DXB", "LON", "NYC", "BKK", "PAR"]  # Dubai, London, New York, Bangkok, Paris

origin = st.sidebar.selectbox("Choose Origin (Airport Code)", origins)
destination = st.sidebar.selectbox("Choose Destination (Airport Code)", destinations)

days = st.sidebar.slider("Days of Travel", 3, 30, 7)
date = st.sidebar.date_input("Departure Date")

if st.sidebar.button("Generate Itinerary"):
    trip = f"to {destination} for {days} days"
    result = chain({"trip": trip})

    st.subheader("💰 Suggested Budget")
    st.write(result["budget"])

    st.subheader("📍 Places to Visit")
    st.write(result["places"])

    st.subheader("✈️ Flight Suggestions")
    flights = get_mock_flights(origin, destination, date.strftime("%Y-%m-%d"))
    st.markdown(flights)
