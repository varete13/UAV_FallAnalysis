import math
import time
from pymavlink import mavutil
import asyncio
import threading

finished=False

mav = mavutil.mavlink_connection('COM5', baud=57600)
  # Velocidad máxima en m/s

def check_groundspeed(max_speed = 0.3):
    # Esperar mensaje
    while True:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:    
            # Calcular groundspeed a partir de la velocidad en el plano XY
            vx = msg.vx / 100  # Convertir de cm/s a m/s
            vy = msg.vy / 100  # Convertir de cm/s a m/s
            
            groundspeed_m_s_2 = vx**2 + vy**2  # Calcular la groundspeed
            
            # Si la groundspeed supera un umbral, imprimir un mensaje
            if groundspeed_m_s_2 > max_speed**2:
                print(f"El dron ha alcanzado una velocidad de groundspeed superior a {max_speed:.2f} m/s")
                desarmar()
                finished=True
                break

def armar():
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # Comando de armado/desarmado
        0,  # Confirmación
        1,  # 1 para armar, 0 para desarmar
        0, 0, 0, 0, 0, 0  # Parámetros vacíos
    )
    print("Armando el dron...")

def desarmar():
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # Comando de armado/desarmado
        0,  # Confirmación
        0,  # 1 para armar, 0 para desarmar
        0, 0, 0, 0, 0, 0  # Parámetros vacíos
    )
    print("Desarmando el dron...")

async def subir_a(altura_objetivo):
    print(f"Subiendo a {altura_objetivo} metros...")
    while not finished:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        alt_actual = msg.relative_alt / 1000.0  # Convertir mm a metros
        # print(alt_actual,altura_objetivo,alt_actual - altura_objetivo)
        set_throttle(3, 1650)  # Establecer el canal 3 (throttle) 
        print(f"Altura actual: {alt_actual:.2f} m")
        if alt_actual > altura_objetivo:  
            break

    print(f"{'ANUNCIO':#^100}\n")
    print(f"Altura alcanzada: {alt_actual:.2f} metros \n")
    print(f"{'ANUNCIO':#^100}")    

def set_throttle(channel, pwm_value):
    # Enviar el comando de override al canal 3 (throttle)
    mav.mav.rc_channels_override_send(
        mav.target_system,  # Target system ID
        mav.target_component,  # Target component ID
        0, 0, pwm_value, 0, 0, 0, 0, 0  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
    )

def simple_forward():
    # print("forward")
    while not finished:
        set_throttle(3, 1600)
        set_throttle(2, 1600)

async def up_and_forward_fall(altura_objetivo=1,max_speed=0.3):
    # Subir a la altura deseada
    await subir_a(altura_objetivo)
    
    t1=threading.Thread(target=check_groundspeed,args=(max_speed,))
    t2=threading.Thread(target=simple_forward)
    
    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Comenzar a verificar la velocidad de groundspeed en segundo plano
    # asyncio.gather(check_groundspeed(max_speed),simple_forward())
    # check_groundspeed_task=asyncio.create_task(check_groundspeed(max_speed))
    # simple_forward_task=asyncio.create_task(simple_forward())

    # await check_groundspeed_task
    # await simple_forward_task


armar()  # Desarmar el dron
time.sleep(5)  # Esperar 5 segundos antes de comenzar
asyncio.run(up_and_forward_fall(2,4))  # Subir a 5 metros de altura
# asyncio.run(subir_a(2))
# asyncio.run(check_groundspeed(4))

