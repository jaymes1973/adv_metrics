
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import pandas as pd
from highlight_text import ax_text, fig_text

data="02-September-2021 11_32_28 team_stats.csv"


st.set_page_config(
     page_title="Advanced Metrics",
     #layout="wide",
     )

st.title('Tracking advanced metrics')

textc='#1d3557'
linec='#808080'
font='Arial'
bgcolor="#FAF9F6"#'#f1faee'
color1='#e63946'
color2='#a8dadc'
color3='#457b9d'
color4='#B2B3A9'
color5='#1d3557'
color6="#006daa"
pathcolor="#C4D5CB"
arrowedge=bgcolor

@st.cache(allow_output_mutation=True)
def get_data(file):
    df=pd.read_csv(file).fillna(0)
    return (df)

df=get_data(data)

teams = list(df['Home Team'].drop_duplicates())
teams=sorted(teams)
team = st.sidebar.selectbox(
    "Select a team:", teams, index=3)

conditions=[(df['Home Team']==team),(df['Away Team']==team)]
values=[df['Away Team'],df['Home Team']]
df['Opposition']=np.select(conditions, values)

df = df.loc[(df['Home Team'] == team)|(df['Away Team'] == team)]
df['Home Expected goals (xG)']=df['Home Expected goals (xG)']-df['Home xG penalty']
df['Away Expected goals (xG)']=df['Away Expected goals (xG)']-df['Away xG penalty']

df['xG For'] = df.apply(
    lambda x: x["Home Expected goals (xG)"] if x["Home Team"] == team 
    else x["Away Expected goals (xG)"], axis=1)

df['xG Against'] = df.apply(
    lambda x: x["Away Expected goals (xG)"] if x["Home Team"] == team 
    else x["Home Expected goals (xG)"], axis=1)

df['xG Open Play For'] = df.apply(
    lambda x: x["Home xG open play"] if x["Home Team"] == team 
    else x["Away xG open play"], axis=1)

df['xG Open Play Against'] = df.apply(
    lambda x: x["Away xG open play"] if x["Home Team"] == team 
    else x["Home xG open play"], axis=1)

df['xG Set Play For'] = df.apply(
    lambda x: x["Home xG set play"] if x["Home Team"] == team 
    else x["Away xG set play"], axis=1)

df['xG Set Play Against'] = df.apply(
    lambda x: x["Away xG set play"] if x["Home Team"] == team 
    else x["Home xG set play"], axis=1)

df['Goals For'] = df.apply(
    lambda x: x["Home Goals"] if x["Home Team"] == team 
    else x["Away Goals"], axis=1)

df['Goals Against'] = df.apply(
    lambda x: x["Away Goals"] if x["Home Team"] == team 
    else x["Home Goals"], axis=1)

cols= ['Home Team','Away Team','Opposition','xG For', 'xG Against', 'Goals For', 'Goals Against'
       ,'xG Open Play For','xG Open Play Against','xG Set Play For','xG Set Play Against']

df2 = df[cols].reset_index(drop=True)

df2 =df2.reindex(index=df2.index[::-1])
df2 = df2.reset_index(drop=True)
df2 = df2.reset_index()
df2=df2.rename(columns={"index":"Gameweek"})


df2['xG Difference']=df2['xG For']-df2['xG Against']
df2['Goal Difference']=df2['Goals For']-df2['Goals Against']


df_2=df2.iloc[:,4:]
metrics=df_2.columns.tolist()
metrics.append("None")
metric1 = st.sidebar.selectbox(
    "Choose a metric:", metrics, index=0)

metric2 = st.sidebar.selectbox(
    "Choose a metric:", metrics, index=1)

labels =df2['Opposition']
ticks=list(range(len(df2)))

fig1, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,10),dpi=80)
fig1.tight_layout(pad=6.0,w_pad=0.5)
fig1.patch.set_facecolor(bgcolor)

if metric1 != "None":

    X=np.arange(len(df2))
    y1=np.array(df2[metric1], dtype=float)
    z1 = np.polyfit(X, y1, 1)
    p1 = np.poly1d(z1)

    ax.plot(y1,label = metric1,color = color1,lw=5)
    ax.plot(X,p1(X),"--",color=color1,alpha=0.5)
    #ax[0, 0].set_title('xG Comparison', fontsize=14)

    for ax in fig1.axes:
        ax.tick_params(axis='x',labelrotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.label.set_color(textc)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,color=textc)
        ax.legend(facecolor=bgcolor)
        ax.set_facecolor(bgcolor)
        for x,y in zip(df2.Gameweek,df2[metric1]):

            label = "{:.2f}".format(y)
            
            plt.annotate(label, # this is the text
                         (x,y), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center

elif metric1 == "None":
    pass

if metric2 != "None":

    X=np.arange(len(df2))
    y2=np.array(df2[metric2], dtype=float)  
    z2 = np.polyfit(X, y2, 1)
    p2 = np.poly1d(z2)

    ax.plot(y2, label = metric2, color=color2,lw=5)
    ax.plot(X,p2(X),"--",color=color2,alpha=0.5)
    #ax[0, 0].set_title('xG Comparison', fontsize=14)

    for ax in fig1.axes:
        ax.tick_params(axis='x',labelrotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.label.set_color(textc)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,color=textc)
        ax.legend(facecolor=bgcolor)
        ax.set_facecolor(bgcolor)
        for x,y in zip(df2.Gameweek,df2[metric2]):

            label = "{:.2f}".format(y)
            
            plt.annotate(label, # this is the text
                         (x,y), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center

elif metric2 == "None":
    pass
#fig_text(s=f"<{team} Advanced Data>", #ha='center'
 #       x=0.075, y =1, fontsize=22,fontfamily=font,color=textc,highlight_colors = [textc],highlight_weights=['bold'])

#fig_text(s=f"All xG figures are not including penalties", #ha='center'
 #       x=0.075, y =0.95, fontsize=16,fontfamily=font,color=textc,highlight_colors = [textc],highlight_weights=['bold'])


st.pyplot(fig1)

df3=df2.drop(columns=["Opposition", "Gameweek"])
st.dataframe((df3).reset_index(drop=True))
