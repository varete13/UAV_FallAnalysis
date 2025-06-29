import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt 
from scipy.signal import argrelextrema

from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List,Literal
from itertools import cycle


def log_entries(fichero:str=None) -> Dict:
    """
    Le os contidos do ficheiro referentes a entradas no .txt extraido de ardupilot

    :param Fichero:
    Enderezo do fichero a leer
    
    :param Registro:
    Diccionario contentendo Fecha,hora, medicions de entradas

    :return: Rexistro de entradas


    """
    Registro_ARM=defaultdict(list)
    Registro_AHR2=defaultdict(list)
    Registro_ATT=defaultdict(list)
    Registro_GPS=defaultdict(list)

    Registro_BAROS=[defaultdict(list),defaultdict(list)]
    Registro_IMUS=[defaultdict(list),defaultdict(list),defaultdict(list)]

    with open(fichero,'r') as file:

        for raw_line in file:

            line=raw_line.strip().split(',')

            match line[0]:

                case 'AHR2':
                    Registro_AHR2['Time'].append(float(line[1]))
                    Registro_AHR2['Roll'].append(float(line[2]))
                    Registro_AHR2['Pitch'].append(float(line[3]))
                    Registro_AHR2['Yaw'].append(float(line[4]))
                    Registro_AHR2['Alt'].append(float(line[5]))
                    Registro_AHR2['Lat'].append(float(line[6]))
                    Registro_AHR2['Lon'].append(float(line[7]))

                case 'ATT':
                    Registro_ATT['Time'].append(float(line[1]))
                    Registro_ATT['DesRoll'].append(float(line[2]))
                    Registro_ATT['Roll'].append(float(line[3]))
                    Registro_ATT['DesPitch'].append(float(line[4]))
                    Registro_ATT['Pitch'].append(float(line[5]))
                    Registro_ATT['DesYaw'].append(float(line[6]))
                    Registro_ATT['Yaw'].append(float(line[7]))


                case 'GPS':
                    Registro_GPS['Time'].append(float(line[1]))
                    Registro_GPS['Status'].append(int(line[3]))
                    Registro_GPS['NSats'].append(int(line[6]))

                    Registro_GPS['Lat'].append(float(line[8]))
                    Registro_GPS['Lon'].append(float(line[9]))
                    Registro_GPS['Alt'].append(float(line[10]))
                    Registro_GPS['Speed'].append(float(line[11]))
                    Registro_GPS['VZ'].append(float(line[13]))
                    Registro_GPS['Yaw'].append(float(line[14]))

                case 'BARO':
                    index=int(line[2])

                    Registro_BAROS[index]['Time'].append(float(line[1]))
                    Registro_BAROS[index]['Alt'].append(float(line[3]))
                    Registro_BAROS[index]['Pressure'].append(float(line[4]))
                    Registro_BAROS[index]['Temperature'].append(float(line[5]))
                
                case 'IMU':
                    index=int(line[2])

                    Registro_IMUS[index]['Time'].append(float(line[1]))
                    Registro_IMUS[index]['GyroX'].append(float(line[3]))
                    Registro_IMUS[index]['GyroY'].append(float(line[4]))
                    Registro_IMUS[index]['GyroZ'].append(float(line[5]))
                    Registro_IMUS[index]['AccX'].append(float(line[6]))
                    Registro_IMUS[index]['AccY'].append(float(line[7]))
                    Registro_IMUS[index]['AccZ'].append(float(line[8]))

                case 'ARM':
                    Registro_ARM['Time'].append(float(line[1]))
                    Registro_ARM['ArmState'].append(float(line[2]))
                    

    return {'ARM':Registro_ARM,'AHR2':Registro_AHR2, 'ATT':Registro_ATT, 'BARO':Registro_BAROS, 'GPS':Registro_GPS, 'IMU':Registro_IMUS}

def filter_time(MegaDict:Dict[str,list[defaultdict]],centro):
    
    Filteres_MegaDict={}

    radio=2e6

    antes=centro-radio
    despues=centro +radio

    for Sensor, Mediciones in MegaDict.items():

        if isinstance(Mediciones,List):
            
            List_sensores=[]

            for sensor in Mediciones:
                mini_dict=defaultdict(list)
                indices=[i for i,t in enumerate(sensor['Time']) if antes<= t and t<=despues ]
                
                for key,value in sensor.items():
                    mini_dict[key]=value[indices[0]:indices[-1]]
                
                List_sensores.append(mini_dict)
            
            Filteres_MegaDict[Sensor]=List_sensores

        else:
            mini_dict=defaultdict(list)
            indices=[i for i,t in enumerate(Mediciones['Time']) if antes<= t and t<=despues ]
           
            if not Sensor =='ARM':

                for key,value in Mediciones.items():
                    if len(value)>1:
                        mini_dict[key]=value[indices[0]:indices[-1]]


            else:
                mini_dict['Time']=[Mediciones['Time'][indices[0]]]
                mini_dict['ArmState']=[Mediciones['ArmState'][indices[0]]]
            
            Filteres_MegaDict[Sensor]=mini_dict
    
    return Filteres_MegaDict

count_saves=0
def save_log(FilteredDict,altitude:int,velocity:int,mode:Literal['alt','climb']='alt'):
    
    global count_saves

    count_saves+=1

    with open(f'sample_{mode}_{altitude}_v_{velocity}_number_{count_saves}.log','w') as file:
                
        for Sensor,Mediciones in FilteredDict.items():
            if isinstance(Mediciones,list):
            
                for index,sensor in enumerate(Mediciones):
                    n_samples=len(sensor['Time'])

                    # t0=sensor['Time'][0]
                    t0=0

                    for n_entry in range(n_samples):
                        entry=f'{Sensor},'
                        time=sensor['Time'][n_entry]
                        entry+=f'{time-t0},'
                        entry+=f'{index},'
                        for jk,key in enumerate(sensor.keys()):
                            if jk==0:
                                continue
                            entry+=f'{sensor[key][n_entry]},'

                        file.write(entry[:-1]+'\n')
            
            else:
                n_samples=len(Mediciones['Time'])

                # t0=Mediciones['Time'][0]
                t0=0

                for n_entry in range(n_samples):
                    entry=f'{Sensor},'
                    for key in Mediciones.keys():
                        if key=='Time':
                            entry+=f'{Mediciones[key][n_entry]-t0},'
                        else:
                            entry+=f'{Mediciones[key][n_entry]},'


                    file.write(entry[:-1]+'\n')

def accZ_time_peaks(MegaDict:Dict[str,list[defaultdict]]):
    Acelero=MegaDict['IMU'][0]
    time=Acelero['Time']
    accz=Acelero['AccZ']
    # print(accz)
    max_index=argrelextrema(np.array(accz),np.greater)
    return [time[int(i)] for i in max_index[0] if accz[int(i)]>0 ]

class SoloPlot:
    def __init__(self,MegaDict):
        
        self.color_cycle=cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'])
        
        
        self.MG=MegaDict
        
        self.ARM=MegaDict['ARM']
        self.AHR2=MegaDict['AHR2']
        self.ATT=MegaDict['ATT']
        self.GPS=MegaDict['GPS']
        self.Baros=MegaDict['BARO']
        self.IMUS=MegaDict['IMU']

    def reset_colors(self):
        self.color_cycle=cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'])

    @staticmethod
    def rotation_matrix(yaw, pitch, roll):
    # Asume ángulos en radianes
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
    
        # Orden típico: Yaw (Z) → Pitch (Y) → Roll (X)
        R = Rz @ Ry @ Rx
        return R
    
    @staticmethod
    def global_values(x,y,z,yaw,pitch,roll,mode:Literal['deg','rad']='deg'):
        def rotation_matrix(yaw, pitch, roll):
        # Asume ángulos en radianes
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll),  np.cos(roll)]
            ])
            
            Ry = np.array([
                [ np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0, 0, 1]
            ])
        
            # Orden típico: Yaw (Z) → Pitch (Y) → Roll (X)
            R = Rz @ Ry @ Rx
            return R
        
        v_rot = np.array([x, y, z])
        
        if mode=='deg':
            yaw=np.deg2rad(yaw)
            pitch=np.deg2rad(pitch)
            roll=np.deg2rad(roll)

        R = rotation_matrix(yaw, pitch, roll)

        # Invertir rotación
        v_orig = R.T @ v_rot

        return v_orig    

    def coherent_ahr2_imus(self)->list[defaultdict[str,list]]:
        result=[]

        for i in range(len(self.IMUS)):

            mini_dict=defaultdict(list)

            imus_times=self.IMUS[i]['Time']
            instrumento_indices=[np.argmin(np.abs(np.array(imus_times)-t_ahr2)) for t_ahr2 in self.AHR2['Time']]

            mini_dict['Time_ahr2']=self.AHR2['Time']
            mini_dict['Time_imu']=[self.IMUS[i]['Time'][id_i] for id_i in instrumento_indices]

            for key1 in ['Yaw','Pitch','Roll']:
                mini_dict[key1]=self.AHR2[key1]

            for key in ['AccX','AccY','AccZ'] :
                sensor_reading=[self.IMUS[i][key][id_i] for id_i in instrumento_indices]
                mini_dict[key].append(sensor_reading)
            
            result.append(mini_dict)
        
        return result

    @staticmethod
    def relativize(data):
        t0=data[0]
        return [point-t0 for point in data]
    @property
    def global_coherent_Acc(self):
        study:list[defaultdict[str,list]]=self.coherent_ahr2_imus()

        for n_imu in range(len(self.IMUS)):
            
            seleccion=[study[n_imu]['AccX'][0],study[n_imu]['AccY'][0],study[n_imu]['AccZ'][0],
                       study[n_imu]['Yaw'],study[n_imu]['Pitch'],study[n_imu]['Roll']]
            
            armas=np.array([self.global_values(x,y,z,yaw,pitch,roll) for x,y,z,yaw,pitch,roll in zip(*seleccion)])

            new_tag='global_Acc'

            for i_component,tag in enumerate(['X','Y','Z']):
                study[n_imu][new_tag+tag]=list(armas[:,i_component])
        
        return study

    def plot_box_IMUS(self,mode:Literal['x','y','z','x_global','y_global','z_global']='z'):
        self.reset_colors()
        match mode:

            case 'x'|'y'|'z':
                fig,ax=plt.subplots(figsize=(5,5))
                Letter=mode.upper()
                
                raw_entries=[self.IMUS[i]['Acc'+ Letter] for i in range(3)]
                entries=[(min(i),np.mean(i),max(i)) for i in raw_entries]

                bplot=ax.boxplot(raw_entries,tick_labels=['Raw','Imu 1','Imu 2'],patch_artist=True)


                for patch, color in zip(bplot['boxes'], self.color_cycle):
                    patch.set_facecolor(color)


                ax.set_ylabel(r'$\left[ \frac{m}{s^2} \right] $',rotation=0)

                fig.suptitle('Aceleración en '+ Letter)
                fig.legend()
            
            case 'x_global'|'y_global'|'z_global':
                pass

    def plot_IMUS(self,mode:Literal['x','y','z','x_global','y_global','z_global']='z'):
        self.reset_colors()
        match mode:

            case 'x'|'y'|'z':
                fig,ax=plt.subplots(1,3,figsize=(15,5),sharey=True)
                fig.subplots_adjust(top=0.85)

                for i,tag in enumerate(['Raw','Imu 1','Imu 2']):
                    color=next(self.color_cycle)
                    ax[i].set_title(tag)


                    Letter=mode.upper()

                    relative_time=SoloPlot.relativize(self.IMUS[i]['Time'])

                    ax[i].plot(list(map(lambda x: x*1e-6,relative_time)),self.IMUS[i]['Acc'+ Letter],color=color)
                ax[0].set_ylabel(r'$\left[ \frac{m}{s^2} \right] $',rotation=0)

                fig.suptitle('Aceleración en '+ Letter)
            
            case 'x_global'|'y_global'|'z_global':
                pass

    def plot_GPS(self,mode:Literal['NSats','Lat','Lon','Alt','Speed','VZ','Yaw','Ground_Position','3D_Position']):
        self.reset_colors()
        match mode:

            case 'Ground_Position':
                fig,ax=plt.subplots()
                ax.axis('equal')
                ax.scatter(self.GPS['Lon'],self.GPS['Lat'])

            case '3D_Position':
                fig,ax=plt.subplots(subplot_kw={'projection': '3d'})
                ax.scatter(self.GPS['Lon'],self.GPS['Lat'],self.GPS['Alt'])

            case _:
                relative_time=SoloPlot.relativize(self.GPS['Time'])

                fig,ax=plt.subplots(figsize=(15,5))
                ax.plot(relative_time,self.GPS[mode])