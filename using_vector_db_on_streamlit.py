import sqlite3
import pdb
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Create a connection to the SQLite database
pdb.set_trace()
conn = sqlite3.connect("Matrimonial_sample_db.db")
c = conn.cursor()

# Create a table to store user bio data
c.execute("create table if not exists Users_bio_data(Name text, Bio text)")

# Create a table to store user embeddings
c.execute("create table if not exists Users_embeddings(Name text, Embedding blob)")

# Function to insert user bio data into the database
def insert_user_bio_data(name, bio):
    c.execute("Insert or Ignore into Users_bio_data values(?, ?)", (name, bio))
    conn.commit()

# Function to convert user bio data into embeddings
def convert_bio_to_embedding(bio):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(bio)
    return embedding

# Function to insert user embeddings into the database
def insert_user_embedding(name, embedding):
    c.execute("Insert or Ignore into Users_embeddings values(?, ?)", (name, embedding.tobytes()))
    conn.commit()

# Function to retrieve user embeddings from the database
def retrieve_user_embeddings():
    c.execute("select * from Users_embeddings")
    rows = c.fetchall()
    return rows

# Function to create a vector database
def create_vector_database(embeddings):
    if not embeddings:
        return None
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index

# Function to find the best matches for a given user
def find_best_matches(index, user_embedding, num_matches):
    distances, indices = index.search(np.array([user_embedding]), num_matches)
    # pdb.set_trace()
    return indices[0]

# Create a Streamlit app
st.title("Matrimonial App User Registration")

# Create a form to insert user bio data
with st.form("Matrimonial_Users_data_form"):
    user_name = st.text_input("Name")
    user_bio = st.text_area("Bio")
    submit_button = st.form_submit_button("Submit")

# Insert user bio data into the database and convert it into an embedding
if submit_button:
    insert_user_bio_data(user_name, user_bio)
    embedding = convert_bio_to_embedding(user_bio)
    insert_user_embedding(user_name, embedding)
    st.success("Data Inserted Successfully!!")

# Display user bio data
st.title("User's Bio Data")
c.execute("select * from Users_bio_data")
rows = c.fetchall()
df = pd.DataFrame(rows, columns=['Name', 'Bio'])
st.write(df)

# Create a vector database
embeddings = []
c.execute("select * from Users_embeddings")
rows = c.fetchall()
for row in rows:
    if row[1]:
        embedding = np.frombuffer(row[1], dtype=np.float32)
        embeddings.append(embedding)

index = create_vector_database(embeddings)

# Find the best matches for a given user
st.title("Best Matches")
user_name = st.selectbox("Select a user", [row[0] for row in retrieve_user_embeddings()])
num_matches = st.number_input("Number of matches", min_value=1, max_value=10)
if st.button("Find matches"):
    c.execute("select * from Users_embeddings where Name = ?", (user_name,))
    row = c.fetchone()
    if row[1]:
        user_embedding = np.frombuffer(row[1], dtype=np.float32)
        if index is not None:
            indices = find_best_matches(index, user_embedding, num_matches)
            matches = [row[0] for row in retrieve_user_embeddings()]
            matches = [matches[i] for i in indices]
            st.write("Best matches:")
            for match in matches:
                st.write(match)
        else:
            st.write("No embeddings found.")
    else:
        st.write("No embedding found for the selected user.")