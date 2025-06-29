import math
import time
import asyncio
import threading
import datetime

from pymavlink import mavutil

mav = mavutil.mavlink_connection('COM5', baud=57600)

def set_throttle(channel, pwm_value):
    # Enviar el comando de override al canal 3 (throttle)
    orden=[0,0,0,0,0,0,0,0]
    orden[int(channel-1)]=pwm_value

    mav.mav.rc_channels_override_send(
        mav.target_system,  # Target system ID
        mav.target_component,  # Target component ID
        *orden  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
    )

def armar():
    mav.mav.command_long_send(
    mav.target_system, mav.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # Comando de armado/desarmado
    0,  # Confirmación
    1,  # 1 para armar, 0 para desarmar
    0, 0, 0, 0, 0, 0  # Parámetros vacíos
    )

    set_throttle(3, 1100)  # Establecer el canal 3 (throttle) a un valor seguro para el despegue

def desarmar():
    # mav.mav.command_long_send(
    #     mav.target_system,
    #     mav.target_component,
    #     mavutil.mavlink.MAV_CMD_DO_FLIGHTTERMINATION,
    #     0,  # Confirmation
    #     1,  # 1 to terminate
    #     0, 0, 0, 0, 0, 0  # Empty parameters
    # )

    mav.mav.command_long_send(
    mav.target_system, mav.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # Comando de armado/desarmado
    0,  # Confirmación
    1,  # 1 para armar, 0 para desarmar
    0, 0, 0, 0, 0, 0  # Parámetros vacíos
    )
    print("Desarmando el dron...")

def subir_a(event,altura_objetivo):
    print(f"Subiendo a {altura_objetivo} metros...")
    while not event.is_set():
        # msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True,timeout=0.2)
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True,timeout=0.2)
        if msg:
            alt_actual = msg.relative_alt / 1000.0  # Convertir mm a metros
            # print(alt_actual,altura_objetivo,alt_actual - altura_objetivo)
            # set_throttle(2,1400)
            # print(f"Altura actual: {alt_actual:.2f} m")
            if alt_actual > altura_objetivo:  
                print(f"{'ANUNCIO':-^100}\n")
                print(f"Altura alcanzada: {alt_actual:.2f} metros \n")
                print(f"{'ANUNCIO':-^100}") 
                break
        set_throttle(3, 1750) # Establecer el canal 3 (throttle) 
    
    return

def check_groundspeed(event,max_speed = 0.3):
    # Esperar mensaje
    print(f"Velocidad Umbral: {max_speed:.2f} m/s")
    while not event.is_set():
        # Esperar un mensaje de posición global
        msg = mav.recv_match(type='VFR_HUD', blocking=True)
        if msg:    
            # Si la groundspeed supera un umbral, imprimir un mensaje
            if msg.groundspeed > max_speed:
                print(f"{'ANUNCIO':-^100}\n")
                print(f"Velocidad {msg.groundspeed:.2f} m/s \n")
                print(f"{'ANUNCIO':-^100}")
                event.set()  # Establecer el evento de finalización
                desarmar()
                break
    
    return

def wait_for_fail(event):
    while not event.is_set():
        msg = mav.recv_match(type='RC_CHANNELS', blocking=True)
        # set_throttle(3, 1590)  # Establecer el canal 3 (throttle)
        if msg:
            if msg.chan8_raw < 1500:
                event.set()  # Establecer el evento de finalización
                desarmar()
                break
    
    return

def save_log(event,altura_max,velocidad_max):
    # Guardar el log en un archivo
    fecha_simple=str(fecha).split( " ")[-1].replace("-","_").split(".")[0]
    fecha_simple=fecha_simple.replace(":","_")


    with open(f"data_bruta/log_alt_{altura_max:.0f}_v_{velocidad_max:.0f}_{fecha_simple}.tlog", "wb") as log_file:
        while not event.is_set():
            msg = mav.recv_match(blocking=True)
            if msg is None:
                continue

            log_file.write(msg.get_msgbuf())

    print("Guardando log...")
    return


def forward_hover(event):

    pitch = 1700  # Valor de pitch para avanzar
    throttle=1790

    while not event.is_set():
        # print("Avanzando...")
        mav.mav.rc_channels_override_send(
            mav.target_system,  # Target system ID
            mav.target_component,  # Target component ID
            0,pitch,throttle,0,0,0,0,0  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
        )
        time.sleep(0.1)  # Esperar un poco antes de enviar el siguiente comando

def simple_opertaion(altura,max_speed=0.3):

    altura_max=altura
    velocidad_max=max_speed

    finish_event = threading.Event()
    log_event = threading.Event()

    print("# Esperando Inicio de vuelo #")
    while True:
        msg = mav.recv_match(type='RC_CHANNELS', blocking=True,timeout=0.2)
        if msg:
            if msg.chan8_raw > 1500:
                break
    
    armar()
    time.sleep(0.3)  # Esperar un momento antes de iniciar la subida
    t_log=threading.Thread(target=save_log, args=(log_event,altura,max_speed))
    t_trust = threading.Thread(target=wait_for_fail, args=(finish_event,))
    t_log.start()
    t_trust.start()

    subir_a(finish_event,altura)  # Subir a la altura deseada
    # time.sleep(0.1)  # Esperar un momento antes de iniciar el avance

    t_ground = threading.Thread(target=check_groundspeed, args=(finish_event,max_speed))
    t_ground.start()
    
    forward_hover(finish_event)  # Mantenerse en el aire

    time.sleep(5)
    log_event.set()  # Esperar un momento antes de iniciar el avance
    print("  M. Cumplida  ")

    t_trust.join()
    t_log.join()
    t_ground.join()
    




mav.wait_heartbeat()


fecha=datetime.datetime.now()
print(f"{str(fecha)::^100} \n")	

simple_opertaion(2,2)

fecha=datetime.datetime.now()
print(f"\n{str(fecha)::^100}")	
# while True:
#     msg = mav.recv_match(type='VFR_HUD',blocking=True)
#     if msg: 
#         print(msg.groundspeed)