import math
import time
from pymavlink import mavutil
import asyncio

# Configurar conexión MAVLink
mav = mavutil.mavlink_connection('COM5', baud=57600)

# Función para verificar groundspeed

max_speed = 0.3  # Velocidad máxima en m/s

async def check_groundspeed():
    # Esperar mensaje
    while True:
        msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:    
            # Calcular groundspeed a partir de la velocidad en el plano XY
            vx = msg.vx / 100  # Convertir de cm/s a m/s
            vy = msg.vy / 100  # Convertir de cm/s a m/s
            
            groundspeed_m_s = math.sqrt(vx**2 + vy**2)  # Calcular la groundspeed
            
            print(f"Groundspeed: {groundspeed_m_s:.5f} m/s")
            
            # Si la groundspeed supera un umbral, imprimir un mensaje
            if groundspeed_m_s > max_speed:
                print(f"El dron ha alcanzado una velocidad de groundspeed superior a {max_speed:.5f} m/s")
                disarm_drone()
                break

# Función para verificar el modo de vuelo
def get_flight_mode():
    # Recibir el mensaje HEARTBEAT
    msg = mav.recv_match(type='HEARTBEAT', blocking=True)
    mode = msg.custom_mode
    print(f"Modo de vuelo actual: {mode}")
    
    # Puedes verificar si el modo es adecuado para el armado
    # Ejemplo: GUIDED o STABILIZE son modos adecuados
    if mode == 4:  # Modo GUIDED
        return True
    elif mode == 0:  # Modo STABILIZE
        return True
    else:
        return False

# Función para verificar si el sistema está listo para ser armado
def is_system_ready():
    # Comprobar el estado del sistema (usando SYS_STATUS o HEARTBEAT)
    msg = mav.recv_match(type='SYS_STATUS', blocking=True)
    if msg:
        print(msg)
        # print(f"Nivel de batería: {msg.battery_voltage / 1000} V")
        # print(f"Estado de los sensores: {msg.sensor_health}")
        # # Aquí puedes verificar si los sensores son saludables y la batería tiene suficiente carga
        return True  # Puedes agregar condiciones adicionales aquí
    else:
        print("No se pudo obtener el estado del sistema.")
        return False

# Función principal para verificar si está listo para ser armado
def check_ready_to_arm():
    if get_flight_mode() and is_system_ready():
        print("El dron está listo para ser armado.")
        return True
    else:
        print("El dron no está listo para ser armado.")
        return False

# Si todo está bien, armar el dron
def arm_drone():
    # if check_ready_to_arm():
    # Enviar el comando de armado
    # Enviar el comando de override al canal 7 (arm)
    mav.mav.rc_channels_override_send(
        mav.target_system,  # Target system ID
        mav.target_component,  # Target component ID
        0, 0, 0, 0, 0, 0, 2006, 0  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
    )
    print("Comando de armado enviado.")
    # else:
    #     print("No se pudo armar el dron, ya que no está listo.")


def disarm_drone():
    # Enviar el comando MAV_CMD_COMPONENT_ARM_DISARM
    mav.mav.rc_channels_override_send(
        mav.target_system,  # Target system ID
        mav.target_component,  # Target component ID
        0, 0, 0, 0, 0, 0, 982, 0  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
    )
    print("Comando de desarmado enviado.")

async def set_throttle(channel, pwm_value, duration=2):
    print(f"Configurando el canal {channel} a {pwm_value} PWM durante {duration} segundos...")

    # Enviar el comando de override al canal 3 (throttle)
    mav.mav.rc_channels_override_send(
        mav.target_system,  # Target system ID
        mav.target_component,  # Target component ID
        0, 0, pwm_value, 0, 0, 0, 0, 0  # CH1, CH2, CH3 (Throttle), CH4, CH5, CH6, CH7, CH8
    )

    # Mantener el throttle durante el tiempo deseado
    await asyncio.sleep(duration)

    # Detener el override para devolver el control al piloto
    mav.mav.rc_channels_override_send(
        mav.target_system,  # Target system ID
        mav.target_component,  # Target component ID
        0, 0, 0, 0, 0, 0, 0, 0  # Restablecer todos los canales
    )
    print("Throttle restablecido.")

def leer_canales_rc(puerto='COM5', baudrate=57600):
    """
    Lee los canales RC en tiempo real y muestra los valores.
    """
    # mav = mavutil.mavlink_connection(puerto, baud=baudrate)
    mav.wait_heartbeat()
    print("Conectado al PX4. Esperando datos de RC...")

    while True:
        msg = mav.recv_match(type='RC_CHANNELS', blocking=True)
        if msg:
            print(f"CH1: {msg.chan1_raw}, CH2: {msg.chan2_raw}, CH3: {msg.chan3_raw}, CH4: {msg.chan4_raw}, "
                  f"CH5: {msg.chan5_raw}, CH6: {msg.chan6_raw}, CH7: {msg.chan7_raw}, CH8: {msg.chan8_raw}")
            # Normalmente, el CH3 es el throttle en configuraciones estándar
        else:
            print("Esperando datos RC...")

# Usa el puerto correspondiente (ejemplo: 'COM5' en Windows o '/dev/ttyUSB0' en Linux)
leer_canales_rc()
def esperar_armado():
    """Espera hasta que el dron esté armado"""
    print("Esperando a que el dron se arme...")
    
    while True:
        msg = mav.recv_match(type='HEARTBEAT', blocking=True)
        if msg:
            armado = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if armado:
                print("¡Dron armado!")
                break

# arm_drone()
# esperar_armado()
async def main():
    await asyncio.gather(
        set_throttle(channel=3, pwm_value=1800, duration=20),
        check_groundspeed()
    )

# asyncio.run(main())

# # arm_drone()
# set_throttle(channel=3, pwm_value=1500, duration=10)
# # Llamar a la función
# check_groundspeed()
