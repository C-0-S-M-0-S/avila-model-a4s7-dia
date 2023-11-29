import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px

st.set_page_config(layout='wide')

st.title("📊 Dataviz")

col_md, col_img = st.columns(2)
col_md.markdown("""<br><br><br><br><br>
                    This page presents graphical description of our dataset.""", unsafe_allow_html=True)

col_img.image(Image.open("./streamlit_ui/Dashboard.jpg"), width=400, caption="Illustration made with DALL-E 3")


df_train = pd.read_csv('avila/avila-tr.txt', header=None)
df_train.columns = ['intercolumnar_distance', 'upper_margin', 'lower_margin',
                    'exploitation', 'row_number', 'modular_ratio', 'interlinear_spacing',
                    'weight', 'peak_number', 'modular_ratio_interlinear_spacing', 'class']


# box plot preparation
df_train.set_index('class', inplace=True)
dfs = df_train.stack().reset_index()
dfs.columns = ['class', 'category', 'values']

# box plot that shows the distribution of the values of each attribute per class
box_fig1 = px.box(dfs, x='category', y='values', color='category', animation_frame='class', title='Distribution of the values of each attribute per class', range_y=[-4, 4])
box_fig1.update_layout(bargap=0.1, bargroupgap=0.1)
st.plotly_chart(box_fig1, use_container_width=True)

# box plot that shows the distribution of the class per attribute
box_fig2 = px.box(dfs, x='class', y='values', color='class', animation_frame='category', title='Distribution of the class per attribute', range_y=[-4, 4])
box_fig2.update_layout(bargap=0.1, bargroupgap=0.1)
st.plotly_chart(box_fig2, use_container_width=True)

