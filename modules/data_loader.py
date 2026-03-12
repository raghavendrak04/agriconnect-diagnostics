import json
import streamlit as st

@st.cache_data
def load_info_data(file_path='info.json'):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

def get_info_by_class_name(info_data, class_name):
    if not class_name: return None
    
    search_term = class_name.lower().replace("_", " ") 
    
    for item in info_data:
        candidates = [
            item.get('Stage_Name', ''),
            item.get('Disease_Name', ''),
            item.get('Variety_Name', ''),
            item.get('Stage_Title', ''),
            item.get('Disease_Title', ''),
            item.get('Variety_Title', '')
        ]
        
        valid_candidates = [c for c in candidates if c and c.strip()]
        
        if any(c.lower() == search_term for c in valid_candidates):
            return item
            
        if any(search_term in c.lower() for c in valid_candidates) or any(c.lower() in search_term for c in valid_candidates):
            return item
            
    return None
