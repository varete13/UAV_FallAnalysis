import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
import pandas
import re

from pathlib import Path
from typing import Literal
from itertools import cycle,islice
from collections import defaultdict
from matplotlib import patches

def nombre_valido(archivo):
    """
    Verifica si el nombre del archivo (sin extensión) sigue el patrón:
    sample_alt_<n>_v_<n>_number_<n>
    sample_climb_<n>_v_<n>_number_<n>
    """
    patron = re.compile(r'^sample_alt_\d+_v_\d+_number_\d+$')
    patron2 = re.compile(r'^sample_climb_\d+_v_\d+_number_\d+$')

    nombre_sin_extension = Path(archivo).stem  # elimina .log o cualquier extensión
    return bool(patron.match(nombre_sin_extension) or patron2.match(nombre_sin_extension) )



class MultiCompar:
    
    color_palette=plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle=cycle(color_palette)

    def __init__(self,carpeta_caso):

        self.ARM=[]
        self.AHR2=[]
        self.IMU=[]

        self.CO_AHRW_IMU=[]

        self.ATT=[]
        self.GPS=[]
        self.BARO=[]

        self.REF={'ARM':self.ARM,'AHR2': self.AHR2,'ATT': self.ATT,'GPS': self.GPS,'BARO': self.BARO,'IMU': self.IMU,'CO_AHR"_IMU': self.CO_AHRW_IMU}

        self.carpeta=carpeta_caso

        _,self._altitude,_,self._groundspeed=self.carpeta.split('/')[-1].split('_')

        self.leer_carpeta()
        self.integrar_archivos()
        self.conversion_dataframe()

        self.Disarm_ref_times=dict(zip(self.ARM['Ensayo'],self.ARM['Time']))
    
    def leer_carpeta(self):
        """
        Devuelve una lista de rutas completas de archivos .log en la carpeta dada.

        Parámetro:
        - carpeta: ruta a la carpeta (str o Path)

        Retorna:
        - Lista de objetos Path con extensión .log
        """
        carpeta = Path(self.carpeta)
        self.archivos_lectura=[f for f in carpeta.iterdir() if f.is_file() and f.suffix == '.log']

    def integrar_archivos(self):

        for archivo in self.archivos_lectura:
            if not nombre_valido(archivo.stem):
                return
            
            ensayo=int(archivo.stem.split('_')[-1])

            imu_ref={0:'Raw',1:'Scaled 1',2:'Scaled 2'}


            with open(archivo,'r') as file:
                for raw_line in file:

                    line=raw_line.strip().split(',')

                    iter_dir={'Ensayo':ensayo}
                    
                    match line[0]:
                        
                        case 'AHR2':

                            for n_place,key in enumerate(['Time','Roll','Pitch','Yaw','Alt','Lat','Lon']):
                                iter_dir.update({key:float(line[n_place+1])})

                        case 'ATT':

                            for n_place,key in enumerate(['Time','DesRoll','Roll','DesPitch','Pitch','DesYaw','Yaw']):
                                iter_dir.update({key:float(line[n_place+1])})

                        case 'GPS':
                            
                            for n_place,key in enumerate(['Time','Status','NSats','Lat','Lon','Alt','Speed','VZ','Yaw']):
                                iter_dir.update({key:float(line[n_place+1])})

                        case 'BARO':

                            baro_index=int(line[2])

                            iter_dir.update({'Instrumento':baro_index})
                            for n_place,key in enumerate(['Time','Alt','Pressure','Temperature']):
                                if key=='Time':
                                    iter_dir.update({key:float(line[n_place+1])})
                                else:
                                    iter_dir.update({key:float(line[n_place+2])})
                        
                        case 'IMU':

                            imu_index=int(line[2])

                            iter_dir.update({'Instrumento':imu_ref[imu_index]})

                            for n_place,key in enumerate(['Time','GyroX','GyroY','GyroZ','AccX','AccY','AccZ']):
                                if key=='Time':
                                    iter_dir.update({key:float(line[n_place+1])})
                                else:
                                    iter_dir.update({key:float(line[n_place+2])})
                        
                        case 'ARM':
                            iter_dir.update({'Time':float(line[1]),'ArmState':float(line[2])})
                    
                    self.REF[line[0]].append(iter_dir)
    
    def conversion_dataframe(self):
        for key in self.REF.keys():
            setattr(self,key, pandas.DataFrame(self.REF[key]))
            self.REF[key]=pandas.DataFrame(self.REF[key])

    
    def global_accs(self):

        def match_nearest_times(df1, df2, time_col, columns_to_add):
            """
            Para cada tiempo en df1[time_col], encuentra el tiempo más cercano en df2[time_col]
            y trae solo las columnas especificadas en columns_to_add.
            
            Parámetros:
            - df1: DataFrame base
            - df2: DataFrame con el que se compara
            - time_col: nombre de la columna de tiempo (string)
            - columns_to_add: lista de nombres de columnas de df2 a traer
            
            Devuelve:
            - Un nuevo DataFrame combinando df1 + columnas seleccionadas de df2
            """
            # Asegurar que ambas columnas de tiempo estén en formato datetime
            df1 = df1.copy()
            df2 = df2.copy()
            df1[time_col] = pd.to_datetime(df1[time_col])
            df2[time_col] = pd.to_datetime(df2[time_col])
            
            # Función interna para encontrar fila más cercana
            def closest_selected(t):
                idx = (df2[time_col] - t).abs().idxmin()
                return df2.loc[idx, columns_to_add]
            
            # Aplicar para cada fila de df1
            closest_values = df1[time_col].apply(closest_selected)
            
            # Convertir resultado a DataFrame
            closest_values_df = pd.DataFrame(list(closest_values))
            
            # Concatenar con df1
            result = pd.concat([df1.reset_index(drop=True), closest_values_df.reset_index(drop=True)], axis=1)
            
            return result
    
        return match_nearest_times(self.IMU,self.AHR2,'Time',['Pitch','Roll','Yaw'])

    @classmethod
    def reset_colors(cls):
        cls.color_cycle=cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def line_acc_plot(self):
        data=self.IMU[self.IMU['Instrumento']=='Raw']


        fig=plt.figure(figsize=(10,6))
        print(data)
        ax=sns.lineplot(
            data=data,
            x='Time',
            y='AccZ',
            hue='Ensayo',
            marker='o',
        )

        ax.set_title(f'Aceleración en el eje Z')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Aceleración (m/s²)')
        
        ax.grid(True)
        return ax
    
    def mediana(valores,bin_width = 0.1):
        # 1. Crear los bins (rangos) de 0.1        
        min_val = np.floor(min(valores) * 10) / 10
        max_val = np.ceil(max(valores) * 10) / 10 + bin_width  # extender un poco el rango
        bins = np.arange(min_val, max_val + bin_width, bin_width)

        # 2. Agrupar los datos en esos rangos
        frecuencias, edges = np.histogram(valores, bins=bins)

        # 3. Crear DataFrame con los intervalos
        df = pd.DataFrame({
            'lim_inf': edges[:-1],
            'lim_sup': edges[1:],
            'frecuencia': frecuencias
        })

        # 4. Calcular frecuencia acumulada
        df['freq_acum'] = df['frecuencia'].cumsum()
        N = df['frecuencia'].sum()
        N2 = N / 2

        # 5. Localizar la clase de la mediana
        fila_mediana = df[df['freq_acum'] >= N2].iloc[0]
        i = df.index.get_loc(fila_mediana.name)
        L = fila_mediana['lim_inf']
        F = df.iloc[i - 1]['freq_acum'] if i > 0 else 0
        f = fila_mediana['frecuencia']
        h = bin_width

        # 6. Calcular mediana agrupada
        mediana = L + ((N2 - F) / f) * h
        return mediana

    @staticmethod
    def relativize(data):
        t0=data[0]
        return [point-t0 for point in data]
    
    @staticmethod
    def single_boxplot(ax,data,input,y=1,width=0.65,subvalue=0.5,second_min=False,dy=0.5):
        """
        solo acepta os modulos individuales, no acepta listas de modulos

        """
        width/=2
        unique_instrumentos = np.unique(data['Instrumento'])
    
        def simple_single(ax,data,yi,witdth_i,second_min_i=False):

            color_i=next(MultiCompar.color_cycle)

            min_d=min(data)
            max_d=max(data)
            mean=float(MultiCompar.mediana(data))

            dis_text=0.05*(max_d-min_d)
            dis_text=max(dis_text,10)

            square = patches.Rectangle((min_d,yi-witdth_i/2),max_d-min_d , witdth_i, facecolor=color_i,edgecolor='black',zorder=1)
            ax.annotate(f'{min_d:.2f}', xy=(min_d*1.0, yi), xytext=(min_d-2*dis_text, yi),
                        fontsize=10, ha='center', va='center')
            
            sign_text=1 if max_d-mean>mean-min_d else -1

            # ax.annotate(f'{mean:.2f}', xy=(mean, yi), xytext=(mean+sign_text*dis_text, yi),
            #             fontsize=10, ha='center', va='center')
            ax.annotate(f'{max_d:.2f}', xy=(max_d, yi), xytext=(max_d+dis_text+5, yi),
                        fontsize=10, ha='center', va='center')

            ax.add_patch(square)
            ax.plot([mean,mean],[yi-witdth_i/2,yi+witdth_i/2], color='black',alpha=0.8,zorder=2)
            
            if second_min:
                "Procedimiento exclusivo a este caso particular, donde con un muestreo de 10 Hz, tres indices son 0.3 segundos"
                min_index=np.where(data == min_d)[0]
                # print(int(min_index-3),int(min_index+3))
                
                if min_index-3>0 and min_index+3<len(data):
                    antes=int(min_index-3) 
                    despues=int(min_index+3)

                    values=np.concatenate((data[:antes],data[despues:]))
                    min_d2=min(values)
                    if not min_d2<mean:
                        return
                else:
                    return

                ax.plot([min_d2,min_d2],[yi-witdth_i/2,yi+witdth_i/2], linestyle='-',linewidth=2,alpha=0.8,color='white',zorder=2)
                # ax.annotate(f'{min_d2:.2f}', xy=(min_d2, yi), xytext=(min_d2-dis_text, yi),fontsize=10, ha='center', va='center')



        inter_sub=width/((len(unique_instrumentos)+subvalue*(len(unique_instrumentos)-1)))
        width_i=inter_sub/subvalue

        positions=[y+width/2+width_i/2-(inter_sub+width_i)*i for i in range(len(unique_instrumentos))]

        for i,inst in enumerate(unique_instrumentos):

            data_i=data[data['Instrumento']==inst]
            data_i=data_i[input].values

            simple_single(ax,data_i,positions[i],width_i,second_min_i=second_min)


        ax.axhline(y=y+dy, color='grey', linestyle='--', alpha=0.2)
        
        MultiCompar.color_cycle=cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'])

        ax.set_title('Comparativa Multi-Ensayo Extremos y Mediana')
        ax.set_ylabel('Ensayos')
        ax.set_xlabel(input)
        ax.margins(x=0.23)

        handles=[patches.Patch(color=next(MultiCompar.color_cycle),label=inst) for inst in unique_instrumentos]
        ax.legend(handles=handles, title='Instrumento', loc='center left')
        MultiCompar.color_cycle=cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'])
    
    def multi_ensayo_imu(self,ax=None,mode:Literal['AccZ','AccX','AccY']='AccZ'):

        if ax is None:
            fig,ax=plt.subplots(figsize=(10,7))
        # data=Zeta.IMU[Zeta.IMU['Instrumento']=='Raw']

        n_ensayos = len(set(self.IMU['Ensayo']))

        for i in range(1,n_ensayos+1):
            data= self.IMU[self.IMU['Ensayo']==i]
            MultiCompar.single_boxplot(ax,data,mode,y=i,width=0.7,subvalue=0.5,second_min=True)

        ax.set_title(f'Altitud {self._altitude} m y GS {self._groundspeed} m/s')
        ax.set_xlabel(f'{mode} $[m/s^2]$')
        return ax

    def free_fall_times(self,ensayo:int):
        def detectar_inicio_crecimiento(df,Y,t0):
            """
            Devuelve el primer valor de col_time donde col_valor empieza a crecer,
            sin modificar el DataFrame original.
            
            Retorna:
            - valor de col_time (o None si no hay crecimiento)
            """
            col_time='Time' 
            col_valor='Alt'
            # Ordenar por tiempo por seguridad
            df_sorted = df.sort_values(col_time)
            
            # Obtener la serie de valores
            valores = df_sorted[col_valor].values
            
            # Recorrer diferencias directamente
            for i in range(1, len(valores)):
                if valores[i] <= (Y - float(self._altitude)):
                    return df_sorted.iloc[i][col_time]
                

            return t0+1e6  # No se encontró crecimiento

        t0=self.Disarm_ref_times[ensayo]

        rin=self.AHR2[self.AHR2['Ensayo']==ensayo]
        y_max=rin['Alt'].max()
        tf=detectar_inicio_crecimiento(rin,y_max,t0=t0)
        return t0,tf

    @property
    def fall_data(self):
        filtrados = {}

        for clave, df in self.REF.items():
            if 'Time' in df.columns and 'Ensayo' in df.columns:
                # Obtener los distintos valores de ensayo en ese DataFrame
                ensayos_unicos = df['Ensayo'].unique()
                
                lista_filtrada = []

                for ensayo in ensayos_unicos:
                    try:
                        t0, tf = self.free_fall_times(ensayo)  

                        filtrado = df[
                            (df['Ensayo'] == ensayo) &
                            (df['Time'] >= t0) &
                            (df['Time'] <= tf)
                        ]

                        lista_filtrada.append(filtrado)

                    except Exception as e:
                        print(f"No se pudo obtener t0, tf para ensayo {ensayo}: {e}")
                
                # Concatenamos todos los fragmentos filtrados
                if lista_filtrada:
                    filtrados[clave] = pd.concat(lista_filtrada, ignore_index=True)
        
        return filtrados


    def ajustar_y_comparar(self,ensayo,vy=0):

        # t0,tf=self.free_fall_times(ensayo)

        x_baro=self.fall_data['AHR2']
        x_gps=self.fall_data['GPS']
    
        x_baro_i=x_baro[x_baro['Ensayo']==ensayo]
        x_gps_i=x_gps[x_gps['Ensayo']==ensayo]
        # x_baro_i=self.AHR2[self.AHR2['Ensayo']==ensayo]
        # x_baro_i=x_baro_i[x_baro_i['Time']>=t0]
        # x_baro_i=x_baro_i[x_baro_i['Time']<=tf]

        vy=np.clip(vy,0,4*np.sin(np.deg2rad(25)))  # limitado a 4 m/s e inclinación de 25º, pues es la velocidad de climb fijada y la maxima inclinacion del dron


        x_baro=x_baro_i['Time'].values
        x_gps=x_gps_i['Time'].values

        if x_baro[0]>x_gps[0]:
            # Aseguramos que ambos inicien al mismo tiempo
            x_gps=x_gps-x_gps[0]+x_baro[0]
        else:
            x_baro=x_baro-x_baro[0]+x_gps[0]

        x_baro=self.relativize(x_baro)
        x_gps=self.relativize(x_gps)

        x_baro=[i*1e-6 for i in x_baro]
        x_gps=[i*1e-6 for i in x_gps]


        y_baro=x_baro_i['Alt'].values
        y_baro=self.relativize(y_baro)

        y_gps=x_gps_i['Alt'].values
        y_gps=self.relativize(y_gps) 
        
        coeffs_ref= [-9.81/2, vy, 0]
        """
        Ajusta un polinomio a los datos (x_baro, y_baro), calcula R² y_baro compara con un polinomio concreto.
        
        Parámetros:
        - x_baro, y_baro: datos
        - degree: grado del ajuste polinómico
        - coeffs_ref: lista de coeficientes del polinomio concreto (ordenados como np.polyfit)
        
        Devuelve:
        - Un diccionario con:
            'Experimental': {
                'coeffs_baro': coeficientes ajustados,
                'y_pred': valores ajustados sobre x_baro,
                'r2': R² del ajuste
            },
            'Teorico': {
                'coeffs_baro': coeficientes concretos,
                'y_pred': valores del polinomio concreto sobre x_baro,
                'r2': R² del polinomio concreto frente a los datos
            }
        """
        def polyfit_no_constante(x, y, degree):
            X = np.vander(x, N=degree + 1)[:, :-1]  # quitamos constante
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return beta
        # Ajuste polinómico BArometro
        coeffs_baro = polyfit_no_constante(x_baro, y_baro,2)
        coeffs_baro = np.append(coeffs_baro,0)
        y_baro_pred = np.polyval(coeffs_baro, x_baro)
        
        # Ajuste polinómico GPS
        coeffs_gps = polyfit_no_constante(x_gps, y_gps,2)
        coeffs_gps = np.append(coeffs_gps,0)
        y_gps_pred = np.polyval(coeffs_gps, x_gps)

        # R² del ajuste mediciom experimenta del barometro
        ss_res_baro = np.sum((y_baro - y_baro_pred)**2)
        ss_tot_baro = np.sum((y_baro - np.mean(y_baro))**2)
        r2_baro = 1 - ss_res_baro / ss_tot_baro
        
        # R² del ajuste mediciom experimenta del gps
        ss_res_gps = np.sum((y_gps - y_gps_pred)**2)
        ss_tot_gps = np.sum((y_gps - np.mean(y_gps))**2)
        r2_gps = 1 - ss_res_gps / ss_tot_gps

        # Evaluar polinomio concreto en barometro
        y_ref = np.polyval(coeffs_ref, x_baro)
        ss_res_ref = np.sum((y_baro - y_ref)**2)
        r2_ref_baro = 1 - ss_res_ref / ss_tot_baro

        # Evaluar polinomio concreto en barometro
        y_ref_gps = np.polyval(coeffs_ref, x_gps)
        ss_res_ref_gps = np.sum((y_gps - y_ref_gps)**2)
        r2_ref_gps = 1 - ss_res_ref_gps / ss_tot_gps

        
        result = {
            'Experimental': {
                'coeffs_baro': coeffs_baro,
                'coeffs_gps':coeffs_gps,
                'climb_angle_º':np.rad2deg(np.arcsin(coeffs_gps[1]/4)),

                't_baro':x_baro,
                't_gps':x_gps,

                'y_baro_pred': y_baro_pred,
                'y_baro':y_baro,

                'y_gps':y_gps,
                'y_gps_pred':'',

                'r2_baro': r2_baro,
                'r2_gps':r2_gps
            },
            'Teorico': {
                'coeffs_': coeffs_ref,

                't':x_baro,

                'y_pred': y_ref,

                'r2_ref_baro': r2_ref_baro,
                'r2_ref_gps':r2_ref_gps,
            }
        }
        
        return result
    
    def single_plot_fit_cuadratic(self,ensayo:int,ax=None,show_baro=False):
        if ax is None:
            fig,ax=plt.subplots()
        
        mini_data=self.ajustar_y_comparar(ensayo)

        if show_baro:
            ax.scatter(mini_data['Experimental']['t_baro'],mini_data['Experimental']['y_baro'],label='Barometro', marker='X', alpha=.8)
        ax.scatter(mini_data['Experimental']['t_gps'],mini_data['Experimental']['y_gps'],label='GPS', marker='D', alpha=.8)

        # ax.plot(mini_data['Experimental']['t_baro'],mini_data['Experimental']['y_baro_pred'],label='Baro Prediccion',linestyle='--')
        ax.plot(mini_data['Teorico']['t'],mini_data['Teorico']['y_pred'],label='Parabola',linestyle='--',linewidth=2,color=self.color_palette[2])
        
        ax.set_title(f'Caida Libre - Ensayo {ensayo}')
        
        ax.set_ylabel('y [m]')
        ax.set_xlabel('t [s]')



        plt.legend()


    def parobolic_gps_r2(self,vy=0):
        lista= []
        for ensayo in set(self.GPS['Ensayo']):
            r2_i=self.ajustar_y_comparar(ensayo,vy=vy)['Teorico']['r2_ref_gps']
            if np.isnan(r2_i) :
                print(f'No se pudo calcular R² para el ensayo {ensayo}.')
                r2_i=0
            lista.append(r2_i)
        return lista        

class MetaComparLineal:

    color_palette=plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle=cycle(color_palette)


    def __init__(self,altura,carpeta_base='FinishedSamples'):
        self.altura=altura
        self.carpeta_base=carpeta_base
        self.detect_alt_v()

    
    @classmethod
    def reset_colors(cls):
        cls.color_cycle=cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def detect_alt_v(self):
        ruta_principal = Path(self.carpeta_base)

        # Expresión regular para extraer X e Y
        patron = re.compile(r'^Alt_(\d+)_V_(\d+)$')

        # Conjuntos para mantener únicos
        valores_X = set()
        valores_Y = set()
        combinaciones_XY = set()

        # Opcional: diccionario para mapear (X, Y) → ruta
        rutas_por_xy = {}

        for carpeta in ruta_principal.iterdir():
            if carpeta.is_dir():
                coincidencia = patron.match(carpeta.name)
                if coincidencia:
                    x = int(coincidencia.group(1))
                    y = int(coincidencia.group(2))

                    valores_X.add(x)
                    valores_Y.add(y)
                    combinaciones_XY.add((x, y))
                    rutas_por_xy[(x, y)] = carpeta

        # Ordenar si se desea
        self.alturas = sorted(valores_X)
        self.velocidades = sorted(valores_Y)
        self.combinaciones_alt_vs = sorted(combinaciones_XY)
        
    def plot_alt_cte_vs_gs_box(self,ax=None):

        altura= self.altura
        
        if altura not in self.alturas:
            raise ValueError(f"La altura {altura} no está disponible en las carpetas.")

        velocidades=sorted({y for (x, y) in self.combinaciones_alt_vs if x == altura})
        if not velocidades:
            raise ValueError(f"No hay velocidades disponibles para la altura {altura}.")
        

        RS=[]
        for velocidad in velocidades:
            ruta =f'{self.carpeta_base}/Alt_{altura}_V_{velocidad}'
            comparativa_i = MultiCompar(ruta)
            rs_i=comparativa_i.parobolic_gps_r2()
            RS.append(rs_i)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(RS, labels=velocidades, showfliers=False)

        self.reset_colors()
        for i, group in enumerate(RS):
            x_pos = i+1
            jitter = np.linspace(-0.15,0.15,len(group)) # un jitter independiente por cada punto
            colores_repetidos = list(islice(MetaComparLineal.color_cycle, len(group)))
            ax.scatter(x_pos+jitter, group,color=colores_repetidos,marker='X',s=80)
            self.reset_colors()


            # plt.scatter(x_pos + jitter, group, color='blue', zorder=3, s=60, edgecolors='black')

        num_max_colors=max([ len(group) for group in RS])
        colores_repetidos = list(islice(MetaComparLineal.color_cycle, num_max_colors))
        parches = [patches.Patch(color=color, label=nombre+1) for nombre, color in enumerate(colores_repetidos)]
        ax.legend(handles=parches, title='Ensayos',ncol=2)

        ax.set_title(f'Ajuste parabólico (GPS) a {altura} m')

        ax.set_xlabel('GS [m/s]')
        ax.set_ylabel('R²')

        return ax

    def plot_accz(self, ax=None,entry=None,common:Literal['gs','alt']='alt',mode:Literal['AccZ','AccX','AccY']='AccZ'):
        """
        Plotea la aceleración en Z contra la altura para cada combinación de altura y velocidad.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))


        if common == 'alt':

            altura=entry
            
            for velocidad in self.velocidades:
                ruta = f'{self.carpeta_base}/Alt_{altura}_V_{velocidad}'
                comparativa_i = MultiCompar(ruta)
                MultiCompar.single_boxplot(ax,comparativa_i.IMU,mode,y=velocidad,width=0.7,subvalue=0.5,dy=1)


            ax.set_title(f'Medicion Global a {entry} m')
            ax.set_ylabel('GS [m/s]')

        elif common=='gs':

            velocidad=entry

            for altura in self.alturas:
                ruta = f'{self.carpeta_base}/Alt_{altura}_V_{velocidad}'
                comparativa_i = MultiCompar(ruta)
                MultiCompar.single_boxplot(ax,comparativa_i.IMU,mode,y=altura,width=0.7,subvalue=0.5,dy=1)


        ax.set_xlabel('AccZ [m/s²]')
        
        return ax

        
class MetaComparClimb:

    color_palette=plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle=cycle(color_palette)


    def __init__(self,velocidad,carpeta_base='FinishedSamples'):
        self.velocidad=velocidad
        self.carpeta_base=carpeta_base
        self.detect_climb_v()


    @classmethod
    def reset_colors(cls):
        cls.color_cycle=cycle( plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def detect_climb_v(self):
        ruta_principal = Path(self.carpeta_base)

        # Expresión regular para extraer X e Y
        patron = re.compile(r'^Climb_(\d+)_V_(\d+)$')

        # Conjuntos para mantener únicos
        valores_X = set()
        valores_Y = set()
        combinaciones_XY = set()

        # Opcional: diccionario para mapear (X, Y) → ruta
        rutas_por_xy = {}

        for carpeta in ruta_principal.iterdir():
            if carpeta.is_dir():
                coincidencia = patron.match(carpeta.name)
                if coincidencia:
                    x = int(coincidencia.group(1))
                    y = int(coincidencia.group(2))

                    valores_X.add(x)
                    valores_Y.add(y)
                    combinaciones_XY.add((x, y))
                    rutas_por_xy[(x, y)] = carpeta

        # Ordenar si se desea
        self.climbs = sorted(valores_X)
        self.velocidades = sorted(valores_Y)
        self.combinaciones_alt_vs = sorted(combinaciones_XY)
        
    def plot_gs_cte_vs_climb_box(self,ax=None):

        velocidad= self.velocidad
        
        if velocidad not in self.velocidades:
            raise ValueError(f"La velocidad {velocidad} no está disponible en las carpetas.")


        if not self.climbs:
            raise ValueError(f"No hay angulso disponibles para la esa velocidad {velocidad}.")
        

        RS=[]
        for climb in self.climbs:

            ruta =f'{self.carpeta_base}/Climb_{climb}_V_{velocidad}'
            comparativa_i = MultiCompar(ruta)
            # rs_i=comparativa_i.parobolic_gps_r2(vy=velocidad*np.sin(np.deg2rad(climb)))
            rs_i=comparativa_i.parobolic_gps_r2(velocidad*np.sin(np.deg2rad(climb)))
            RS.append(rs_i)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.boxplot(RS, labels=self.climbs, showfliers=False)

        self.reset_colors()
        for i, group in enumerate(RS):
            x_pos = i+1
            jitter = np.linspace(-0.15,0.15,len(group)) # un jitter independiente por cada punto
            colores_repetidos = list(islice(MetaComparClimb.color_cycle, len(group)))
            ax.scatter(x_pos+jitter, group,color=colores_repetidos,marker='X',s=80)
            self.reset_colors()

        num_max_colors=max([ len(group) for group in RS])
        colores_repetidos = list(islice(MetaComparClimb.color_cycle, num_max_colors))
        parches = [patches.Patch(color=color, label=nombre+1) for nombre, color in enumerate(colores_repetidos)]
        ax.legend(handles=parches, title='Ensayos',ncol=2)

        ax.set_title(f'Ajuste parabólico (GPS) a {velocidad} m/s')

        ax.set_xlabel('Climb [º]')
        ax.set_ylabel('R²')

        return ax

    def plot_accz(self, ax=None,entry=None,common:Literal['gs','climb']='gs',mode:Literal['AccZ','AccX','AccY']='AccZ'):
        """
        Plotea la aceleración en Z contra la climb para cada combinación de climb y velocidad.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))


        if common == 'gs':

            velocidad=entry
            
            for climb in self.climbs:

                ruta = f'{self.carpeta_base}/Climb_{climb}_V_{velocidad}'
                comparativa_i = MultiCompar(ruta)
                MultiCompar.single_boxplot(ax,comparativa_i.IMU,mode,y=climb,width=0.7,subvalue=0.5,dy=7.5)


            ax.set_title(f'Medicion Global a {entry} m/s')
            ax.set_ylabel('Climb [º]')

        elif common=='climb':

            climb=entry

            for velocidad in self.velocidades:
                ruta = f'{self.carpeta_base}/Climb_{climb}_V_{velocidad}'
                comparativa_i = MultiCompar(ruta)
                MultiCompar.single_boxplot(ax,comparativa_i.IMU,mode,y=climb,width=0.7,subvalue=0.5,dy=1)


        ax.set_xlabel('AccZ [m/s²]')
        
        return ax
