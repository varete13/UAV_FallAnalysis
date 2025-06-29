# %%
from typing import Dict,List

# %%
def log_attitude(fichero:str=None,Registro:Dict[str,List[float]]={}) -> Dict[str,List[float]]:
    """
    Le os contidos do ficheiro referentes a actitude no .txt extraido de ardupilot

    :param Fichero:
    Enderezo do fichero a leer
    
    :param Registro:
    Diccionario contentendo Fecha,hora, actitude e velocidades correspondentes medidas

    :return: Rexistro das actitudes e velocidades angulares


    """

    with open(fichero,'r') as file:
        for line in file:
            if 'mavlink_attitude_t' in line:
                Fecha=line.strip().split()[0]
                Hora=line.strip().split()[1]

                lectura=line.strip().split()[11:-5]
                
                if not Registro:
                    Registro['Fecha']=[]
                    Registro['Hora']=[]
                    for i in range(0,len(lectura),2):
                        Registro[lectura[i]]=[]
                Registro['Fecha'].append(Fecha)
                Registro['Hora'].append(Hora)
                for j in range(0,len(lectura),2):
                    Registro[str(lectura[j])].append(float(lectura[j+1].replace(',','.')))
    return Registro

# %%
def log_altitude(fichero:str=None,Registro:Dict[str,List[float]]={}) -> Dict[str,List[float]]:
    """
    Le os contidos do ficheiro referentes a altitude no .txt extraido de ardupilot

    :param Fichero:
    Enderezo do fichero a leer
    
    :param Registro:
    Diccionario contentendo Fecha,hora, altitude e medidas asociadas

    :return: Rexistro das altitudes e diversas vairables


    """

    with open(fichero,'r') as file:
        for line in file:
            if 'mavlink_ahrs2_t' in line:
                Fecha=line.strip().split()[0]
                Hora=line.strip().split()[1]

                lectura=line.strip().split()[11:-5]
                
                if not Registro:
                    Registro['Fecha']=[]
                    Registro['Hora']=[]
                    for i in range(0,len(lectura),2):
                        Registro[lectura[i]]=[]
                Registro['Fecha'].append(Fecha)
                Registro['Hora'].append(Hora)
                for j in range(0,len(lectura),2):
                    Registro[str(lectura[j])].append(float(lectura[j+1].replace(',','.')))

    return Registro

# %%
def log_speed(fichero:str=None,Registro:Dict[str,List[float]]={}) -> Dict[str,List[float]]:
    """
    Le os contidos do ficheiro referentes a velocidad na hud no .txt extraido de ardupilot

    :param Fichero:
    Enderezo do fichero a leer
    
    :param Registro:
    Diccionario contentendo Fecha,hora, medicions de velocidade

    :return: Rexistro de velocidades


    """

    with open(fichero,'r') as file:
        for line in file:
            if 'mavlink_vfr_hud_t' in line:
                Fecha=line.strip().split()[0]
                Hora=line.strip().split()[1]

                lectura=line.strip().split()[11:-5]
                
                if not Registro:
                    Registro['Fecha']=[]
                    Registro['Hora']=[]
                    for i in range(0,len(lectura),2):
                        Registro[lectura[i]]=[]
                Registro['Fecha'].append(Fecha)
                Registro['Hora'].append(Hora)
                for j in range(0,len(lectura),2):
                    Registro[str(lectura[j])].append(float(lectura[j+1].replace(',','.')))

    return Registro

# %%
if __name__=='__main__':
    Attitudes=log_attitude("2024-10-07 17-20-41.txt")
    Altitudes=log_altitude("2024-10-07 17-20-41.txt")
    Speeds=log_speed("2024-10-07 17-20-41.txt")