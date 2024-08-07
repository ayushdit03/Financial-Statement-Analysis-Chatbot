import streamlit as st
from streamlit_option_menu import option_menu
from pages import home
from pages import signup_login
from pages import aboutus
#from pages import chat
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(layout="centered", initial_sidebar_state="collapsed")

# Navbar
st.markdown("<style> ul {display: none;} </style>", unsafe_allow_html=True)

if st.session_state.get('switch_button', False):
    st.session_state['menu_option'] = (st.session_state.get('menu_option', 0) + 1) % 4
    manual_select = st.session_state['menu_option']
else:
    manual_select = None
    
selected = option_menu(None, ["Home", "About Us", 'Signup/Login'], 
    icons=['house', 'info-circle', 'person'], 
    orientation="horizontal", manual_select=manual_select, key='menu_4')



if selected == "Home":
    home.show()

#elif selected == "Chat with AI":
#    chat.show()

elif selected == "About Us":
    aboutus.about_us()

elif selected == "Signup/Login":
    tab1,tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        flag = signup_login.login()

        if flag == 1:
            switch_page("chat")

    with tab2:
        flag = signup_login.signup()

        if flag == 1:
            switch_page("chat")