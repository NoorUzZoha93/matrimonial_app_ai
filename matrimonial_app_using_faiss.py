import pdb
import streamlit as st
import sqlite3
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
# st.balloons()
st.title("WELCOME TO MATRIMONIAL APP")
# Connect to SQLite database
conn = sqlite3.connect("Matrimonial_app.db")
c = conn.cursor()
def registration_function():
    st.subheader("USER REGISTRATION")
    st.markdown("## Enter Your Details")
    name = st.text_input("Enter Your Name:")
    age = st.text_input("Enter Your Age:")
    gender = st.selectbox("Select Your Gender:", ["Male", "Female"])
    education = st.text_input("Education:")
    location = st.text_input("Enter Your Location:")
    submitted = st.button("Submit")
    if submitted:
        c.execute("insert into users values (?,?,?,?,?)", (name, age, gender, education, location))
        conn.commit()
        st.success("User Registered Successfully")
def display_function():
    st.subheader("USERS' DATA")
    #         # fetch data from SQLITE
    data = c.execute("select * from users")
    df = pd.DataFrame(data)
    st.write(df)

def Matching_function():
    st.subheader("FIND SIMILAR PROFILES")
# Create a SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Create a Faiss index
    index = faiss.IndexFlatL2(384)
    # Load user data from SQLite database
    c.execute("SELECT * FROM users")
    users = c.fetchall()
    # Create user embeddings using SentenceTransformer model
    embeddings = []
    for user in users:
        user_text = " ".join(str(column) for column in user)
        # user_text = f"{user[0]} {user[1]} {user[2]} {user[3]}"  # Combine name, age, gender, education and location
        embedding = model.encode(user_text)
        embeddings.append(embedding)
    # Add user embeddings to Faiss index
    index.add(np.array(embeddings))
    # User query input
    query = st.text_input("Enter your query (e.g., age, gender, education,location):")
    # Search for similar users
    if st.button("Search"):
        query_embedding = model.encode(query)
        query_embeddings = np.array([query_embedding])
        D, I = index.search(query_embeddings, k=5)
        similar_users = []
        for i in I[0]:
            # pdb.set_trace()
            similar_users.append(users[i])
        st.write("Similar users:")
        for user in similar_users:
            st.write(f"Name: {user[0]}, Age: {user[1]}, Gender: {user[2]}, Education: {user[3]}, Location: {user[4]}")


tab1, tab2, tab3 =st.tabs(["USER REGISTRATION", "USERS' INFO","FIND SIMILAR PROFILES"])
with tab1:
    registration_function()
with tab2:
    display_function()
with tab3:
    Matching_function()
