
import pinecone

from PIL import Image

from sentence_transformers import SentenceTransformer
import torch
from itertools import cycle
import streamlit as st
from datasets import load_dataset
from pinecone_text.sparse import SpladeEncoder
import os
from pinecone import Pinecone
# # init connection to pinecone
#Splade
print('!!')
pc=Pinecone(
    api_key='d3f75c33-f870-4c04-a4ce-36380d671fbc',
)
# # choose a name for your index
index_name = "cvnangcao2"
index = pc.Index(index_name)
# load the dataset from huggingface datasets hub
fashion = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train"
)
# assign the images and metadata to separate variables
images = fashion["image"]
metadata = fashion.remove_columns("image")
splade = SpladeEncoder()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# load a CLIP model from huggingface
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device=device
)

def hybrid_scale(dense, sparse, alpha: float):
    """Hybrid vector scaling using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: float between 0 and 1 where 0 == sparse only
               and 1 == dense only
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse
def search(query,image=None,alpha=0):
    sparse = splade.encode_queries(query)
    if image:
        dense = model.encode(image).tolist()
    else:
        dense = model.encode(query).tolist()
    hdense, hsparse = hybrid_scale(dense, sparse, alpha=alpha)
    result = index.query(
    top_k=10,
    vector=hdense,
    sparse_vector=hsparse,
    include_metadata=True
    )
    # use returned product ids to get images
    imgs = [images[int(r["id"])] for r in result["matches"]]
    label= [x["metadata"]['productDisplayName']  for x in result["matches"] ]
    score= [x["score"]  for x in result["matches"] ]
    return imgs, label, score
    # display the images


def main():
     # Devide 2 columns
  col1, col2 = st.columns([1, 2], gap="large")
  with col1:
    st.subheader("Alpha")
    alphaa = st.text_input(
       'Input alpha', '0.5',
        label_visibility="collapsed",)
    st.subheader("Text Query")
    title = st.text_input(
       '', 'Men watch',
        label_visibility="collapsed",
        )
    st.subheader("Query image")
    uploaded_file = st.file_uploader(
            "Drop file here", type=["png", "jpg"], accept_multiple_files=False
        )
    query_image=None
    if uploaded_file is not None:
            query_image = Image.open(uploaded_file)
            a=query_image.resize((224, 320))
            st.image(a, caption="Retrieved image")
    imgs, label , score = search(query=title,image=query_image,alpha=float(alphaa))
    dem=0
  with col2:
    st.subheader("Retrieved images")
    cols = cycle(st.columns(5))
    for idx in imgs:
        current_col = next(cols)
        current_col.subheader('Score: '+str(round(score[dem], 2)), divider='rainbow')
        current_col.image(
             idx, width=150, caption=label[dem], use_column_width=True
        )
        
        dem+=1


