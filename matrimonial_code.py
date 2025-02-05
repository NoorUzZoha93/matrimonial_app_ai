import streamlit as st
import sqlite3
import faiss
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Database configuration
DB_PATH = os.path.join(os.getcwd(), "MatrimonialAPP.db")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

st.title("WELCOME TO MATRIMONIAL APP")

# Create table if not exists
c.execute("""CREATE TABLE IF NOT EXISTS matri_users (
            name TEXT, 
            age INTEGER, 
            gender TEXT, 
            education TEXT, 
            location TEXT, 
            preferences TEXT)""")


def registration_function():
    st.subheader("USER REGISTRATION")
    name = st.text_input("Enter Your Name:")
    age = st.number_input("Enter Your Age:", min_value=18, max_value=100)
    gender = st.selectbox("Select Your Gender:", ["Male", "Female"])
    education = st.text_input("Education:")
    location = st.text_input("Enter Your Location:")
    preferences = st.text_input("Enter Your Requirements:")

    if st.button("Submit"):
        c.execute("INSERT INTO matri_users VALUES (?,?,?,?,?,?)",
                  (name, age, gender, education, location, preferences))
        conn.commit()
        st.success("User Registered Successfully")
        st.balloons()


def display_function():
    st.subheader("USERS' DATA")
    data = c.execute("SELECT * FROM matri_users")
    df = pd.DataFrame(data, columns=["Name", "Age", "Gender", "Education", "Location", "Preferences"])
    st.write(df)


def Matching_function():
    st.subheader("FIND SIMILAR PROFILES")

    # Fetch all users
    c.execute("SELECT * FROM matri_users")
    users = c.fetchall()

    if not users:
        st.warning("No users registered yet!")
        return

    # Initialize model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create profiles and embeddings
    user_profiles = []
    all_embeddings = []
    gender_groups = {"Male": [], "Female": []}  # Track embeddings by gender

    for user in users:
        profile = {
            "name": user[0],
            "age": user[1],
            "gender": user[2],
            "education": user[3],
            "location": user[4],
            "preferences": user[5]
        }
        user_profiles.append(profile)

        # Create embedding from profile data
        user_text = f"{profile['name']} {profile['age']} {profile['gender']} " \
                    f"{profile['education']} {profile['location']} {profile['preferences']}"
        embedding = model.encode(user_text)

        # Store embeddings by gender
        gender_groups[profile["gender"]].append(embedding)
        all_embeddings.append(embedding)

    # Create FAISS indices for each gender
    indices = {
        "Male": faiss.IndexFlatL2(384),
        "Female": faiss.IndexFlatL2(384)
    }

    # Add embeddings to respective indices
    for gender in ["Male", "Female"]:
        if gender_groups[gender]:
            embeddings = np.array(gender_groups[gender]).astype('float32')
            indices[gender].add(embeddings)

    # User selection
    options = [profile['name'] for profile in user_profiles]
    selected_user = st.selectbox("Select a user", [""] + options)

    if selected_user and selected_user in options:
        selected_idx = options.index(selected_user)
        selected_profile = user_profiles[selected_idx]
        target_gender = "Female" if selected_profile['gender'] == "Male" else "Male"

        # Display selected profile
        st.subheader("Selected Profile")
        st.table(pd.DataFrame([selected_profile]).T.rename(columns={0: "Details"}))

        # Get query embedding
        query_text = f"{selected_profile['name']} {selected_profile['age']} " \
                     f"{selected_profile['gender']} {selected_profile['education']} " \
                     f"{selected_profile['location']} {selected_profile['preferences']}"
        query_embedding = model.encode(query_text).reshape(1, -1).astype('float32')

        # Search in opposite gender index
        if not gender_groups[target_gender]:
            st.warning(f"No {target_gender} profiles available for matching")
            return

        # Perform similarity search
        k = min(5, len(gender_groups[target_gender]))  # Don't request more than available
        distances, indices = indices[target_gender].search(query_embedding, k)

        # Get matching profiles
        matches = []
        for i, distance in zip(indices[0], distances[0]):
            if i >= 0:  # FAISS returns -1 for invalid indices
                match_profile = [p for p in user_profiles if p["gender"] == target_gender][i]
                matches.append({
                    "Name": match_profile["name"],
                    "Age": match_profile["age"],
                    "Gender": match_profile["gender"],
                    "Education": match_profile["education"],
                    "Location": match_profile["location"],
                    "Preferences": match_profile["preferences"],
                    "Similarity Score": f"{1 / (1 + distance):.2%}"
                })

        if matches:
            st.subheader(f"Top {len(matches)} Matches ({target_gender})")
            st.dataframe(pd.DataFrame(matches), use_container_width=True)
        else:
            st.warning("No suitable matches found")
    else:
        st.info("Please select a user from the dropdown")


# Create tabs
tab1, tab2, tab3 = st.tabs(["USER REGISTRATION", "USERS' INFO", "FIND SIMILAR PROFILES"])
with tab1:
    registration_function()
with tab2:
    display_function()
with tab3:
    Matching_function()

conn.commit()
conn.close()