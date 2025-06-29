from typing import Dict,List
from collections import defaultdict
import matplotlib.pyplot as plt

def log_imus(fichero):
    """
    Le os contidos do ficheiro referentes a altitude no .txt extraido de ardupilot

    :param Fichero:
    Enderezo do ficheiro a leer
    
    :param Registro:
    Diccionario contentendo Fecha,hora, altitude e medidas asociadas

    :return: Rexistro das altitudes e diversas vairables


    """
    Registro_raw=defaultdict(list)
    Registro_imu2=defaultdict(list)
    Registro_imu3=defaultdict(list)

    with open(fichero,'r') as file:
        for line in file:
            
            if 'mavlink_raw_imu_t' in line:
                lectura = line.strip().split('mavlink_raw_imu_t')[-1].split()[:-5]
                for j in range(0,len(lectura),2):
                    key = lectura[j]
                    value = float(lectura[j+1].replace(',','.'))
                    # print(key,' : ',value)
                    Registro_raw[key].append(value)
            
            if 'mavlink_scaled_imu2_t' in line:
                lectura = line.strip().split('mavlink_scaled_imu2_t')[-1].split()[:-5]
                for j in range(0,len(lectura),2):
                    key = lectura[j]
                    value = float(lectura[j+1].replace(',','.'))
                    # print(key,' : ',value)
                    Registro_imu2[key].append(value)
            
            if 'mavlink_scaled_imu3_t' in line:
                lectura = line.strip().split('mavlink_scaled_imu3_t')[-1].split()[:-5]
                for j in range(0,len(lectura),2):
                    key = lectura[j]
                    value = float(lectura[j+1].replace(',','.'))
                    # print(key,' : ',value)
                    Registro_imu3[key].append(value)
    
    return Registro_raw, Registro_imu2, Registro_imu3

def log_attitude(fichero):
    """
    Le os contidos do ficheiro referentes a attitude no .txt extraido de ardupilot

    :param Fichero:
    Enderezo do ficheiro a leer
    
    :param Registro:
    Diccionario contentendo Fecha,hora, attitude e medidas asociadas

    :return: Rexistro das actitudes e diversas vairables


    """
    Registro_attitude=defaultdict(list)

    with open(fichero,'r') as file:
        for line in file:
            if 'mavlink_attitude_t' in line:
                lectura = line.strip().split('mavlink_attitude_t')[-1].split()[:-5]
                for j in range(0,len(lectura),2):
                    Registro_attitude[lectura[j]].append(float(lectura[j+1].replace(',','.')))
    
    return Registro_attitude

def log_maneuver(fichero):
    Registro_maneuver=defaultdict(list)
    with open(fichero,'r') as file:
        for line in file:
            if 'mavlink_vfr_hud_t' in line:
                lectura = line.strip().split('mavlink_vfr_hud_t')[-1].split()[:-5]
                for j in range(0,len(lectura),2):
                    key = lectura[j]
                    value = float(lectura[j+1].replace(',','.'))
                    # print(key,' : ',value)
                    Registro_maneuver[key].append(value)
    
    return Registro_maneuver

def log_position(fichero):
    Registro_position=defaultdict(list)
    
    with open(fichero,'r') as file:
        for line in file:
            if 'mavlink_global_position_int_t' in line:
                lectura = line.strip().split('mavlink_global_position_int_t')[-1].split()[:-5]
                for j in range(0,len(lectura),2):
                    key = lectura[j]
                    value = float(lectura[j+1].replace(',','.'))
                    # print(key,' : ',value)
                    Registro_position[key].append(value)
    
    return Registro_position


if __name__ == '__main__':

    alfredo=log_maneuver('data_bruta/log_alt_4_v_2_21_50_16.txt')

    print(alfredo.keys())

    fig,ax =plt.subplots(2,3,figsize=(15,5))
    ax[0].plot(alfredo['time_boot_ms'],alfredo['relative_alt'],label='Altura relativa')

    plt.show()