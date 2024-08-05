import streamlit as st
from streamlit_option_menu import option_menu
from pages import home
from pages import chat
from pages import signup_login

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Navbar
st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=["Home", "Chat with AI", "About Us", "Signup/Login"],
    icons=["house", "chat-dots", "info-circle", "person"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    home.show()

elif selected == "Chat with AI":
    chat.show()

elif selected == "About Us":
    st.write("about us")

elif selected == "Signup/Login":
    tab1,tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        signup_login.login()
    with tab2:
        signup_login.signup()
