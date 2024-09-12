import streamlit as st
from msclap import CLAP
import numpy as np
import locale

locale.getpreferredencoding = lambda: "UTF-8"
import torch
import pickle
import numpy as np
from pymongo.mongo_client import MongoClient

# Set page configuration
st.set_page_config(
    page_title="Ultimate Music Search", page_icon=":musical_note:", layout="wide"
)

# Add custom CSS
st.markdown(
    """
    <style>
    .stTextInput .st-av {
        font-size: 18px;
    }
    .stTextInput .st-aj {
        font-size: 18px;
    }
    .stTextInput label {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a header
st.title("Ultimate Music Search System")

# Text input to take user input
user_input = st.text_input(
    "What kind of music you want to listen today?", placeholder="Enter your prompt..."
)

# Add some spacing
st.markdown("---")


@st.cache_resource
def clap_model_init():
    clap_model = CLAP(version="2023", use_cuda=False)
    clap_model.clap.load_state_dict(
        torch.load("./clap_model_weight.pth", map_location=torch.device("cpu"))
    )
    return clap_model


clap_model = clap_model_init()

uri = "[YOUR_MONGODB_URI]"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

music_db = client["musicCaptionDB"]
music_cap_collection = music_db["music_captioning_data"]


def check_similarity(
    music_cap_collection, prompt_embeddings, size=128, pct=0.3, given_index=None
):
    prompt_embeddings = prompt_embeddings.reshape(-1)
    prompt_embeddings = prompt_embeddings[:size]
    cosine_sim_list = []
    if size == 128:
        extracted_index = music_cap_collection.find()
    else:
        given_index = given_index.tolist()
        query = {"index": {"$in": given_index}}
        extracted_index = music_cap_collection.find(query)

    for embeddings in extracted_index:
        index = embeddings["index"]
        embed_vector = pickle.loads(embeddings["embeddings"])
        slices_embed = embed_vector[:size]
        cosine_sim_list.append([np.dot(slices_embed, prompt_embeddings.T), index])

    cosine_sim_list.sort(reverse=True)
    lenght = len(cosine_sim_list)
    top_k_ele = np.array(cosine_sim_list[: int(pct * lenght) + 1], dtype=int)
    return top_k_ele[:, 1]


def get_music_address(index):
    given_index = index.tolist()
    query = {"index": {"$in": given_index}}
    extracted_index = music_cap_collection.find(query)
    address_list = []
    for address in extracted_index:
        address_list.append(address["address"])
    return address_list[:5]


if user_input:
    prompt_embeddings = clap_model.get_text_embeddings([user_input])
    prompt_embeddings = prompt_embeddings

    vec1 = check_similarity(
        music_cap_collection, prompt_embeddings, size=128, given_index=None
    )
    vec2 = check_similarity(
        music_cap_collection, prompt_embeddings, size=256, given_index=vec1
    )
    vec3 = check_similarity(
        music_cap_collection, prompt_embeddings, size=512, given_index=vec2
    )
    vec4 = check_similarity(
        music_cap_collection, prompt_embeddings, size=1024, given_index=vec3
    )

    selected_songs = get_music_address(vec4)

    # Display selected songs in a container
    with st.container():
        st.subheader("Selected Songs")
        for song in selected_songs:
            song_ids = song.split("/")[-1]
            st.write(f"Song URL: {song_ids}")

    # Add some spacing
    st.markdown("---")

    # Display a footer
    st.markdown(
        """
        <div style='text-align: center; font-size: 14px; color: gray;'>
            <p>Made with ❤️</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
