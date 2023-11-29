import streamlit as st
from PIL import Image

st.set_page_config(layout='wide')

st.title("Avila's bible study")

col1, col2 = st.columns(2)
col1.markdown("""<br><br><br><br><br>
                    The dataset is made of 10,000 samples from 800 images of Avila's bible.  
                    It is a spanish manuscript from the XII century.  
                    The dataset contains 10 features, and 12 classes.  
                    The goal of this study is to predict the writer.  
                    The left panel allows you to explore dataviz and run predictions.
            """,
            unsafe_allow_html=True)

col2.image(Image.open("./streamlit_ui/bible.png"), width=400, caption="Illustration made with DALL-E 3")
