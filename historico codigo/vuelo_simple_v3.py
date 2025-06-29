import math
import time
import asyncio
import threading

from pymavlink import mavutil

mav = mavutil.mavlink_connection('COM5', baud=57600)

def set_throttle(channel, pwm_value):
    # Enviar el comando de override al canal 3 (throttle)
    mav.mav.rc_channels_override_send(
        mav.target_system,  # Target system ID
        mav.target_component,  # Target component ID
        0, 0, pwm_value, 0, 0, 0, 0, 0  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
    )

def hover_forward(event1,event2,pwm_pitch=1600):
    while not event1.is_set() and not event2.is_set():

        mav.mav.rc_channels_override_send(
            mav.target_system,  # Target system ID
            mav.target_component,  # Target component ID
            0, pwm_pitch, 1700, 0, 0, 0, 0, 0  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
        )
    desarmar()

def wiat_for_fail(event):
    while not event.is_set():
        msg = mav.recv_match(type='RC_CHANNELS', blocking=True)
        if msg:
            if msg.chan8_raw < 1500:
                print(f"CH8: {msg.chan8_raw}")
                event.set()  # Establecer el evento de finalización
                break
    
    return False

def subir_a(event,altura_objetivo):
    print(f"Subiendo a {altura_objetivo} metros...")
    while not event.is_set():
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True,timeout=0.2)
        if msg:
            alt_actual = msg.relative_alt / 1000.0  # Convertir mm a metros
            # print(alt_actual,altura_objetivo,alt_actual - altura_objetivo)
            set_throttle(3, 1700)  # Establecer el canal 3 (throttle) 
            print(f"Altura actual: {alt_actual:.2f} m")
            if alt_actual > altura_objetivo:  
                print(f"{'ANUNCIO':-^100}\n")
                print(f"Altura alcanzada: {alt_actual:.2f} metros \n")
                print(f"{'ANUNCIO':-^100}") 
                break
    return False


def esta_armado():
    """Devuelve True si el dron está armado, False si está desarmado."""
    msg = mav.recv_match(type='HEARTBEAT', blocking=True)
    if msg:
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        return bool(armed)  # Devuelve True si está armado, False si no
    return False


def armar():
    if esta_armado():
        return False
    
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,  # Comando de armado/desarmado
        0,  # Confirmación
        1,  # 1 para armar, 0 para desarmar
        0, 0, 0, 0, 0, 0  # Parámetros vacíos
    )
    print("Armando el dron...")
    while True:
        msg = mav.recv_match(type="HEARTBEAT", blocking=True,timeout=1)
        if msg:
            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if armed:
                print("¡El dron está armado!")
                break
        time.sleep(0.2)  # Pequeña pausa para evitar saturar la CPU
    set_throttle(3, 1400)  # Establecer el canal 3 (throttle) a un valor seguro para el despegue

def desarmar():

    # set_throttle(7,2006)
    while esta_armado():
        print("Desarmando el dron...")
        mav.mav.command_long_send(
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_FLIGHTTERMINATION,
            0,  # Confirmation
            1,  # 1 to terminate
            0, 0, 0, 0, 0, 0  # Empty parameters
            )

        

def check_groundspeed(event,max_speed = 0.3):
    # Esperar mensaje
    while not event.is_set():
        # Esperar un mensaje de posición global
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True,timeout=0.1)
        if msg:    
            # Calcular groundspeed a partir de la velocidad en el plano XY
            vx = msg.vx / 100.0  # Convertir de cm/s a m/s
            vy = msg.vy / 100.0  # Convertir de cm/s a m/s
            
            groundspeed_m_s_2 = vx**2 + vy**2  # Calcular la groundspeed
            print("Groundspeed: {:.5f} m/s".format(math.sqrt(groundspeed_m_s_2)))
            # Si la groundspeed supera un umbral, imprimir un mensaje
            if groundspeed_m_s_2 > max_speed**2:
                print(f"El dron ha alcanzado una velocidad de groundspeed superior a {max_speed:.2f} m/s")
                event.set()  # Establecer el evento de finalización
                desarmar()
                break
    

def operation_up_forward(altura=1,max_speed=2):
    # Eventos de Interes: 
        # Mision Completada
    completed_event = threading.Event()
        # Fallo de Seguridad
    fail_event = threading.Event()

    while True:
        msg = mav.recv_match(type='RC_CHANNELS', blocking=True,timeout=0.2)
        print("# Esperando Inicio de vuelo #")
        if msg:
            if msg.chan8_raw > 1500:
                print(f"CH8: {msg.chan8_raw}")
                break


    armar()

    # Hilo Seguridad
    t_trust = threading.Thread(target=wiat_for_fail, args=(fail_event,))
    t_trust.start()

    subir_a(fail_event,altura)  # Subir a la altura deseada

    # Hilos Principales
    t_moves = threading.Thread(target=hover_forward, args=(completed_event,fail_event,1300))
    t_ground = threading.Thread(target=check_groundspeed, args=(completed_event,max_speed))

    # Inicia ambos hilos
    t_moves.start()
    t_ground.start()

    # Espera a que ambos hilos terminen
    t_moves.join()
    t_ground.join()
    t_trust.join()


if __name__=="__main__":
    operation_up_forward(0,1)