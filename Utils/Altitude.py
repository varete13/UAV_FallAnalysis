import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Pandas_altitutde(fichero:str)->pd.DataFrame:

    columnas=['Fecha','Hora']
    columnas=columnas+[chr(i) for i in range(ord('A'),ord('A')+8)]
    columnas=columnas+['medidas']
    columnas=columnas+[i for i in range(1,48)]

    df_0=df = pd.read_csv(fichero,delim_whitespace=True)
    df_0.columns=columnas


    df_0=df_0[df_0['medidas']=='mavlink_ahrs2_t']
    df_0=df_0.dropna(axis='columns')

    Fecha=df_0.iloc[0,0]
    df_0=df_0.iloc[:,1:]
    df_0=df_0.drop([chr(i) for i in range(ord('A'),ord('A')+8)]+[15,16,17],axis=1)
    Horas=df_0.iloc[:,0]
    df_0=df_0.iloc[:,2:-2]

    data_dicts = [
        {df_0.iloc[i,j]: float(df_0.iloc[i,j + 1].replace(',','.')) 
                    for j in range(0,df_0.shape[1]-1, 2)} for i in range(df_0.shape[0])
                    ]

    df_0 = pd.DataFrame(data_dicts)
    df_0['Hora']=list(Horas) 

    return df_0