import streamlit as st
from pages.Cipher_1 import login_user, register_user  #make comment while page testing 
#from Cipher_1 import login_user, register_user       #make comment while app testing

from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/nikhil_db')
db = client['chatbot']
user_collection = db["users"]

#"""def store_user_data(mail, password):
#    user_entry_data={
#        "user_id":mail,
#        "password":password
#    }
#    user_collection.insert_one(user_entry_data)

#def check_authorization(mail, password):
#    if user_collection.count_documents({"user_id":mail}) == 1:
#        for doc in user_collection.find({"user_id":mail}):
#            stored_pass = doc["password"]
#
#        if password == stored_pass:
#            st.success("login Successful")
#        else:
#            st.warning("Please check your password or email")
#        
#    else:
#        st.warning("Please create an account first!")
#"""


def login():
    flag = 0
    st.header("Login here!")
    with st.form("login_form"):
        mail = st.text_input("Enter your email", placeholder="ex: test@gmail.com")
        password = st.text_input("Enter your password", placeholder="must be minimum 8 characters", type="password")
        st.checkbox("Remember Me!")
        login_btn = st.form_submit_button("Login")

        if login_btn:
            response = login_user(mail, password)

            if response == True:
                st.success("Login successfully")
                flag = 1
            elif response == False:
                st.warning("Invalid Password or email")
            else :
                st.warning("user not found")
            #check_authorization(mail, password)
    return flag

                

def signup():
    flag = 0
    st.header("Sign Up here!")
    with st.form("signup_form"):
        mail = st.text_input("Enter your email", placeholder="ex: test@gmail.com")
        password = st.text_input("Enter your password", placeholder="must be minimum 8 characters", type="password")
        confirm_password = st.text_input("Confirm Password", placeholder="Re-enter password", type="password")
        signup_btn = st.form_submit_button("Sign Up")

        if signup_btn:
            if mail:
                if len(password)<8 or len(confirm_password)<8:
                    st.warning("Password must be 8 characters")

                elif password == confirm_password:
                    if user_collection.count_documents({"user_id": mail}) == 0:
                        register_user(mail, password)
                        #store_user_data(mail, password)
                        st.success("Sign up successful!")
                        flag = 1

                    else:
                        st.warning("User with this mail is already present")

                else:
                    st.warning("Please recheck your password")
            else:
                st.warning("Please provide mail")
    return flag


#login()
#signup()