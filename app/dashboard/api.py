import os
import requests
import streamlit as st

BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def get(path, **params):
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def post(path, payload):
    r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def model_info():
    return get("/model-info")
