from pymavlink import mavutil

# Conéctate a tu dron (ajusta esto)
master = mavutil.mavlink_connection('COM5', baud=57600)
master.wait_heartbeat()
print("Conectado al dron")

# Pide todos los parámetros
master.mav.param_request_list_send(master.target_system, master.target_component)

# Almacena resultados
sr_params = {}

print("Leyendo parámetros SRx_...")

while True:
    msg = master.recv_match(type='PARAM_VALUE', blocking=True, timeout=1)
    if msg is None:
        break
    param_id = msg.param_id.decode("utf-8") if isinstance(msg.param_id, bytes) else msg.param_id
    if param_id.startswith("SR"):
        sr_params[param_id] = msg.param_value

# Mostrar los resultados
for name, value in sorted(sr_params.items()):
    print(f"{name}: {value}")