
import streamlit as st
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd

# Create a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to SQLite database
conn = sqlite3.connect('matrimonial.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users (name text, age integer, gender text, location text, description text)''')
conn.commit()


tab1, tab2, tab3 = st.tabs(["Matrimonial User Registration","Users' Details","Find Your Match"])
# Tab1: User Registration
with tab1:
    st.write("User Registration")
    name = st.text_input("Name", key ="name_ele")
    age = st.number_input("Age", min_value=18, max_value=100, value=25, key ="age_ele")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender_ele")
    location = st.text_input("Location",key="location_ele")
    description = st.text_area("Description", key="description_ele")
    submit_button = st.button("Register")

    if submit_button:
        # Add the new profile to the SQLite database
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (name, age, gender, location, description))
        conn.commit()

        # Create a vector representation for the user
        text = f"{name} is a {age} year old {gender} who lives in {location} and is looking for {description}"
        vector = model.encode(text)

        # Store the original text data along with the vector data
        data = {
            "name": name,
            "age": age,
            "gender": gender,
            "location": location,
            "description": description,
            "vector": vector
        }

        # Add the data to the vector database
        try:
            with open("vector_db.pkl", "rb") as f:
                db = pickle.load(f)
        except FileNotFoundError:
            db = []
        db.append(data)
        with open("vector_db.pkl", "wb") as f:
            pickle.dump(db, f)

        # Display a success message
        st.write("User registered successfully!")

# Tab2: Display Registered Users
with tab2:
    st.write("Display Registered Users")
    submit_button = st.button("Display")

    if submit_button:
        # Get the users from the SQLite database
        c.execute("SELECT * FROM users")
        users = c.fetchall()
        df = pd.DataFrame(users, columns=["Name", "Age", "Gender", "Location", "Description"])

        # Display the dataframe
        st.write(df)

        # Convert the users to vector database
        try:
            with open("vector_db.pkl", "rb") as f:
                db = pickle.load(f)
        except FileNotFoundError:
            db = []
        for user in users:
            name, age, gender, location, description = user
            text = f"{name} is a {age} year old {gender} who lives in {location} and is looking for {description}"
            vector = model.encode(text)
            data = {
                "name": name,
                "age": age,
                "gender": gender,
                "location": location,
                "description": description,
                "vector": vector
            }
            db.append(data)
        with open("vector_db.pkl", "wb") as f:
            pickle.dump(db, f)

# Tab3: Display Users Based on Specific Criteria
with tab3:
    st.write("Display Users Based on Specific Criteria")
    filter_type = st.radio("Select Filter Type", ["Age", "Gender", "Location"])
    if filter_type == "Age":
        age = st.number_input("Age", min_value=18, max_value=100, value=25, key="age_filter")
    elif filter_type == "Gender":
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], key ="gender_filter")
    elif filter_type == "Location":
        location = st.text_input("Location", key="location_filter")

    submit_button = st.button("Search")

    if submit_button:
        # Get the vector database
        try:
            with open("vector_db.pkl", "rb") as f:
                db = pickle.load(f)
        except FileNotFoundError:
            st.write("No users registered.")

        # Filter the users based on the specific criteria
        if filter_type == "Age":
            filtered_db = [user for user in db if user["age"] == age]
        elif filter_type == "Gender":
            filtered_db = [user for user in db if user["gender"] == gender]
        elif filter_type == "Location":
            filtered_db = [user for user in db if user["location"]== location]
        seen = set()
        filtered_db=[user for user in filtered_db if not(user["name"], user["age"], user["gender"], user["location"]) in seen and not seen.add((user["name"], user["age"], user["gender"], user["location"]))]

        # Create a pandas dataframe
        df = pd.DataFrame(filtered_db)
        # Display the dataframe
        st.write(df)
