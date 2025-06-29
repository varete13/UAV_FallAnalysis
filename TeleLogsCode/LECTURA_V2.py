import math
import time
from pymavlink import mavutil
import asyncio

# Configurar conexión MAVLink
mav = mavutil.mavlink_connection('COM5', baud=57600)

def leer_canales_rc():
    """
    Lee los canales RC en tiempo real y muestra los valores.
    """
    # mav = mavutil.mavlink_connection(puerto, baud=baudrate)
    mav.wait_heartbeat()
    print("Conectado al PX4. Esperando datos de RC...")

    n=1
    dt=0

    last_time = time.time()
    rc_channels = None
    position = None
    vfr = None
    
    with open("log_simple.tlog","wb") as log_file:
        while True:
            msg = mav.recv_match(blocking=True, timeout=1)
            if msg is None:
                print("Esperando datos...")
                continue

            log_file.write(msg.get_msgbuf())

            msg_type = msg.get_type()

            if msg_type == 'RC_CHANNELS':
                rc_channels = msg
            elif msg_type == 'GLOBAL_POSITION_INT':
                position = msg
            elif msg_type == 'VFR_HUD':
                vfr = msg

            if rc_channels and position and vfr:

                now = time.time()
                dt = round(now - last_time, 3)
                last_time = now
                
                print(f"{' Iteración '+ str(n) +' ':-^100}")
                print(f"{' Dt = '+ str(dt) +' segundos ':-^100}")
                print(f"Altitud relativa: {position.relative_alt / 1000.0:.2f} m")
                print(f"Velocidad terrestre: {vfr.groundspeed:.2f} m/s")
                print(f"Pitch (canal 2): {rc_channels.chan2_raw}, Throttle (canal 3): {rc_channels.chan3_raw}")


                # Reiniciar para próxima iteración
                rc_channels = None
                position = None
                vfr = None
                n += 1
            

leer_canales_rc()