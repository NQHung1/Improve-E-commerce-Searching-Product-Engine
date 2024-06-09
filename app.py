import streamlit as st
from Test import main

st.set_page_config(
    page_title="Fashion IR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
sidebar = st.sidebar
page = sidebar.radio("Choose a page", ["Search", "Manage"])
sidebar.title("Image retrieval application")

if page == "Search":
    with st.container(border=True):
        sidebar.subheader("Search page")
        main()

if page == "Manage":
    sidebar.subheader("Manage page")


