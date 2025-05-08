import streamlit as st

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def search_medication_1mg(medication_name):
    """
    Generate search URL for medication on 1mg
    
    Args:
        medication_name: Name of the medication to search for
        
    Returns:
        Search URL for the medication on 1mg
    """
    # Format the search URL
    search_query = medication_name.replace(' ', '+')
    search_url = f"https://www.1mg.com/search/all?name={search_query}"
    return search_url

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def search_medication_pharmeasy(medication_name):
    """
    Generate search URL for medication on PharmEasy
    """
    # Format the search URL
    search_query = medication_name.replace(' ', '+')
    search_url = f"https://pharmeasy.in/search/all?name={search_query}"
    return search_url

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def search_medication_netmeds(medication_name):
    """
    Generate search URL for medication on Netmeds
    """
    # Format the search URL
    search_query = medication_name.replace(' ', '+')
    search_url = f"https://www.netmeds.com/catalogsearch/result/{search_query}/all"
    return search_url

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def search_medication_cultfit(medication_name):
    """
    Generate search URL for medication on Cult.fit
    """
    # Format the search URL
    search_query = medication_name.replace(' ', '+')
    search_url = f"https://www.cure.fit/care/medicines/search?q={search_query}"
    return search_url
