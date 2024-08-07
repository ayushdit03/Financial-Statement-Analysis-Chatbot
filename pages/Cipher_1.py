from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from pymongo import MongoClient
import base64
import os
import hashlib

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/nikhil_db')
db = client.chatbot
collection = db.users
config_collection = db.config

# Encryption setup
def get_or_create_key_iv():
    config = config_collection.find_one({"name": "encryption_config"})
    if not config:
        key = os.urandom(32)
        iv = os.urandom(16)
        config_collection.insert_one({
            "name": "encryption_config",
            "key": base64.b64encode(key).decode(),
            "iv": base64.b64encode(iv).decode()
        })
    else:
        key = base64.b64decode(config["key"])
        iv = base64.b64decode(config["iv"])
    return key, iv

key, iv = get_or_create_key_iv()

def generate_salt() -> str:
    return base64.b64encode(os.urandom(16)).decode()

def encrypt_data(data: str) -> str:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(encrypted).decode()

def decrypt_data(encrypted_data: str) -> str:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    encrypted_data = base64.b64decode(encrypted_data)
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    return data.decode()

def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode()).hexdigest()

def register_user(username: str, password: str):
    if collection.find_one({"user_id": username}):
        print("User already exists.")
        return
    
    salt = generate_salt()
    hashed_password = hash_password(password, salt)
    encrypted_password = encrypt_data(hashed_password)
    collection.insert_one({"user_id": username, "salt": salt, "password": encrypted_password})
    print("User registered successfully.")

def login_user(username: str, password: str):
    user = collection.find_one({"user_id": username})
    if user:
        salt = user['salt']
        hashed_password = hash_password(password, salt)
        encrypted_password = user['password']
        decrypted_password = decrypt_data(encrypted_password)
        if hashed_password == decrypted_password:
            print("Login successful.")
            return True
        else:
            print("Invalid password.")
            return False
    else:
        print("User not found.")
        return "usernotfound"

def main():
    action = input("Do you want to (R)egister or (L)ogin? ").strip().upper()
    username = input("Enter username: ").strip()
    password = input("Enter password: ").strip()
    
    if action == 'R':
        register_user(username, password)
    elif action == 'L':
        login_user(username, password)
    else:
        print("Invalid action.")

if __name__ == "__main__":
    main()