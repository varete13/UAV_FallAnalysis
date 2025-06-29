import clr
import MissionPlanner 
import time

# print(dir(MAV))

print(dir(Script))

while True:
    if cs.alt>3:
        print("Altura maior que 3")
    elif cs.alt<3:
        print("Altura menor que 3")
    if cs.groundspeed<1:
        print("Velocidade menor que 1")
    
    if cs.groundspeed>1:
        print("Velocidade maior que 1")

# print(MAV.mavlink_altitude_t.altitude_relative)
# MAV.send_manual_control(0,0,-100,0)
# time.sleep(1)
# MAV.send_manual_control(0,0,0,0)