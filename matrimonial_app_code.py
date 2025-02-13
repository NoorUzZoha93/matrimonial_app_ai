import pdb
import streamlit as st
import sqlite3
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sentence_transformers
st.title("WELCOME TO MATRIMONIAL APP")
# current_dir = os.path.dirname(__file__)
# db_path =os.path.join(current_dir, 'MatApp','matrimonial_app.db')
# @st.cache_data
conn = sqlite3.connect("matrimonial_app.db")
c = conn.cursor()
def registration_function():
    # Connect to SQLite database
    
    c.execute("Create table if not exists matri_users(name text, age integer, gender text, education text, location text, preferences text)")
    st.subheader("USER REGISTRATION")
    st.markdown("## Enter Your Details")
    name = st.text_input("Enter Your Name:")
    age = st.number_input("Enter Your Age:")
    gender = st.selectbox("Select Your Gender:", ["Male", "Female"])
    education = st.text_input("Education:")
    location = st.text_input("Enter Your Location:")
    preferences = st.text_input("Enter Your Requirements:")
    submitted = st.button("Submit")
    if submitted:
        c.execute("insert or ignore into matri_users values (?,?,?,?,?,?)", (name, age, gender, education, location,preferences))
        conn.commit()
        st.success("User Registered Successfully")
        st.balloons()


def display_function():
    st.subheader("USERS' DATA")
    # fetch data from SQLITE
    # c= conn.cursor()
    data = c.execute("select * from matri_users")
    df = pd.DataFrame(data)
    st.write(df)
    conn.commit()


def Matching_function():
#     st.subheader("FIND SIMILAR PROFILES")
#     c.execute("SELECT * FROM matri_users")
#     users = c.fetchall()
# # # Create user embeddings
#     user_embeddings = []
#     # Create a SentenceTransformer model
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     # Compute the user embeddings
#     for user in users:
#         user_text = " ".join(str(column) for column in user)
#         embedding = model.encode(user_text)
#         user_embeddings.append(embedding)
#     user_embeddings = np.array(user_embeddings).reshape(-1,384)
#     # Create a Faiss index
#     index = faiss.IndexFlatL2(384)
#     # Add the user embeddings to the index
#     index.add(np.array(user_embeddings).astype('float32'))
#     # Save the index to a file
#     # faiss.write_index(index, "matrimonialapp_embedding.index")
#     # Read the index from the file
#     # index = faiss.read_index("matrimonialapp_embedding.index")
#     # Create a list of user profiles
#     user_profiles = []
#     for user in users:
#         user_profile = {
#             "name": user[0],
#             "age": user[1],
#             "gender": user[2],
#             "education": user[3],
#             "location": user[4],
#             "preferences": user[5]
#         }
#         user_profiles.append(user_profile)

#     # Create a drop-down list of user profiles
#     options = []
#     for user_profile in user_profiles:
#         option = f"{user_profile['name']}"
#         # option = f"{user_profile['name']}, {user_profile['age']}, {user_profile['gender']}, {user_profile['education']}, {user_profile['location']}, {user_profile['preferences']}"
#         options.append(option)
#     selected_user = st.selectbox("Select a user", [""]+ options)


#     # Selection of the user profiles
#     if selected_user is not None:
#         # Check if the selected user is in the list of options
#         if selected_user in options:
#             selected_user_index = options.index(selected_user)
#             # Get the complete user profile of the selected user
#             selected_user_profile = user_profiles[selected_user_index]
#             # Get the gender of the selected user
#             selected_user_gender = selected_user_profile['gender']

#             # Create a list of user profiles of the opposite gender
#             opposite_gender_profiles = []
#             for user_profile in user_profiles:
#                 if user_profile['gender'] != selected_user_gender:
#                     opposite_gender_profiles.append(user_profile)

#             # Create a list of embeddings for the opposite gender profiles
#             opposite_gender_embeddings = []
#             for user_profile in opposite_gender_profiles:
#                 user_text = f"{user_profile['name']} {user_profile['age']} {user_profile['gender']} {user_profile['education']} {user_profile['location']} {user_profile['preferences']}"
#                 embedding = model.encode(user_text)
#                 opposite_gender_embeddings.append(embedding)
#             # Create a Faiss index
#             index = faiss.IndexFlatL2(len(opposite_gender_embeddings[0]))
#             # Add the opposite gender embeddings to the index
#             index.add(np.array(opposite_gender_embeddings))
#             # Get the embedding of the selected user
#             selected_user_text = f"{selected_user_profile['name']} {selected_user_profile['age']} {selected_user_profile['gender']} {selected_user_profile['education']} {selected_user_profile['location']} {selected_user_profile['preferences']}"
#             selected_user_embedding = model.encode(selected_user_text)

#             # Get the complete user profile of the  selected user
#             # df = pd.DataFrame([selected_user_profile])
#             st.write("Selected User Profile:")
#             st.table([selected_user_profile])
#             #st.table(df)

#             # Search for similar profiles
#             D, I = index.search(np.array([selected_user_embedding]),k=5)
#             # Display similar user profiles
#             similar_user_profiles = []
#             for i, distance in zip(I[0], D[0]):
#                 user_profile = opposite_gender_profiles[i]
#                 profiles ={"Name":user_profile['name'], "Age": user_profile['age'], "Gender":user_profile['gender'], "Education": user_profile['education'], "Location":user_profile['location'], "Preferences": user_profile['preferences'], "Distance": distance}
#                 similar_user_profiles = similar_user_profiles+ [profiles]
#             st.write("Similar User Profiles:")
#             # similar_users_df = pd.DataFrame(similar_user_profiles)
#             st.table(similar_user_profiles)
#             # st.table(similar_users_df)
#     else:
#         st.write("Please select a user.")
    st.subheader("FIND SIMILAR PROFILES")
    c.execute("SELECT * FROM matri_users")
    users = c.fetchall()

    # Create a list of user profiles
    user_profiles = []
    for user in users:
        user_profile = {
            "name": user[0],
            "age": user[1],
            "gender": user[2],
            "education": user[3],
            "location": user[4],
            "preferences": user[5]
        }
        user_profiles.append(user_profile)

    # Create a drop-down list of user profiles
    options = []
    for user_profile in user_profiles:
        option = f"{user_profile['name']}"
        options.append(option)
    selected_user = st.selectbox("Select a user", [""] + options)

    # Selection of the user profiles
    if selected_user is not None:
        # Check if the selected user is in the list of options
        if selected_user in options:
            selected_user_index = options.index(selected_user)
            # Get the complete user profile of the selected user
            selected_user_profile = user_profiles[selected_user_index]

            # Get the features to match
            features = ["Age", "Location", "Education", "Preferences"]
            selected_features = st.multiselect("Select features to match", features)

            # Create a sentence embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Generate embeddings for each user profile
            embeddings = []
            profiles = []
            for user_profile in user_profiles:
                if user_profile['gender'] != selected_user_profile['gender']:
                    profile_text = ""
                    if "Age" in selected_features:
                        profile_text += f"{user_profile['age']} "
                    if "Location" in selected_features:
                        profile_text += f"{user_profile['location']} "
                    if "Education" in selected_features:
                        profile_text += f"{user_profile['education']} "
                    if "Preferences" in selected_features:
                        profile_text += f"{user_profile['preferences']} "
                    embedding = model.encode(profile_text)
                    embeddings.append(embedding)
                    profiles.append(user_profile)

            # Create a Faiss index
            index = faiss.IndexFlatL2(len(embeddings[0]))

            # Add the embeddings to the index
            index.add(np.array(embeddings))

            # Generate an embedding for the selected user profile
            selected_user_text = ""
            if "Age" in selected_features:
                selected_user_text += f"{selected_user_profile['age']} "
            if "Location" in selected_features:
                selected_user_text += f"{selected_user_profile['location']} "
            if "Education" in selected_features:
                selected_user_text += f"{selected_user_profile['education']} "
            if "Preferences" in selected_features:
                selected_user_text += f"{selected_user_profile['preferences']} "
            selected_user_embedding = model.encode(selected_user_text)

            # Search for similar profiles using Faiss
            D, I = index.search(np.array([selected_user_embedding]), k=5)

            # Display similar user profiles
            similar_user_profiles = []
            for i, distance in zip(I[0], D[0]):
                user_profile = profiles[i]
                profiles_dict = {"Name": user_profile['name'], "Age": user_profile['age'], "Gender": user_profile['gender'],
                                 "Education": user_profile['education'], "Location": user_profile['location'],
                                 "Preferences": user_profile['preferences'], "Distance": distance}
                similar_user_profiles.append(profiles_dict)

            st.write("Selected User Profile:")
            df = pd.DataFrame([selected_user_profile])
            df.columns = ['Name', 'Age', 'Gender', 'Education', 'Location', 'Preferences']
            st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)

            st.write("Similar User Profiles:")
            similar_users_df = pd.DataFrame(similar_user_profiles)
            similar_users_df.columns = ['Name', 'Age', 'Gender', 'Education', 'Location', 'Preferences', 'Distance']
            st.markdown(similar_users_df.to_html(index=False, escape=False), unsafe_allow_html=True)
        else:
            st.write("Please select a user.")


tab1, tab2, tab3 =st.tabs(["USER REGISTRATION", "USERS' INFO", "FIND SIMILAR PROFILES"])
with tab1:
    registration_function()
with tab2:
    display_function()
with tab3:
    Matching_function()

conn.commit()
conn.close()
