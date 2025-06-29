import math
import time
import asyncio
import threading
import logging
from typing import Optional

from pymavlink import mavutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("drone_controller")

class DroneController:
    """Class to handle all drone control operations via MAVLink"""
    
    def __init__(self, port='COM5', baud=57600):
        """Initialize the drone controller with communication parameters"""
        self.mav = None
        self.port = port
        self.baud = baud
        self.connect()
    
    def connect(self) -> bool:
        """Establish connection to the drone"""
        try:
            self.mav = mavutil.mavlink_connection(self.port, baud=self.baud)
            logger.info(f"Connected to drone on {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to drone: {e}")
            return False
    
    def set_rc_channel(self, channel_values: dict) -> None:
        """
        Set RC channel values
        
        Args:
            channel_values: Dictionary with channel numbers as keys and PWM values as values
        """
        if not self.mav:
            logger.error("No MAVLink connection established")
            return
            
        # Default all channels to 0 (no override)
        channels = [0] * 8
        
        # Set the specified channels
        for channel, value in channel_values.items():
            if 1 <= channel <= 8:
                channels[channel-1] = value
        
        # Send the RC override command
        try:
            self.mav.mav.rc_channels_override_send(
                self.mav.target_system,
                self.mav.target_component,
                *channels
            )
        except Exception as e:
            logger.error(f"Failed to set RC channels: {e}")
    
    def arm(self) -> bool:
        """Arm the drone"""
        if not self.mav:
            logger.error("No MAVLink connection established")
            return False
            
        try:
            self.mav.mav.command_long_send(
                self.mav.target_system, 
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # Confirmation
                1,  # 1 to arm
                0, 0, 0, 0, 0, 0  # Empty parameters
            )
            logger.info("Arming drone...")
            return True
        except Exception as e:
            logger.error(f"Failed to arm drone: {e}")
            return False
    
    def disarm(self) -> bool:
        """Disarm the drone"""
        if not self.mav:
            logger.error("No MAVLink connection established")
            return False
            
        try:
            self.mav.mav.command_long_send(
                self.mav.target_system, 
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # Confirmation
                0,  # 0 to disarm
                0, 0, 0, 0, 0, 0  # Empty parameters
            )
            logger.info("Disarming drone...")
            return True
        except Exception as e:
            logger.error(f"Failed to disarm drone: {e}")
            return False
    
    def get_altitude(self) -> Optional[float]:
        """Get the current relative altitude in meters"""
        if not self.mav:
            logger.error("No MAVLink connection established")
            return None
            
        try:
            msg = self.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                return msg.relative_alt / 1000.0  # Convert mm to meters
            return None
        except Exception as e:
            logger.error(f"Failed to get altitude: {e}")
            return None
    
    def get_ground_speed(self) -> Optional[float]:
        """Calculate the ground speed in m/s from velocity components"""
        if not self.mav:
            logger.error("No MAVLink connection established")
            return None
            
        try:
            msg = self.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                vx = msg.vx / 100  # Convert cm/s to m/s
                vy = msg.vy / 100  # Convert cm/s to m/s
                return math.sqrt(vx**2 + vy**2)
            return None
        except Exception as e:
            logger.error(f"Failed to get ground speed: {e}")
            return None
    
    def get_rc_channel(self, channel_number: int) -> Optional[int]:
        """Get the current value of a specific RC channel"""
        if not self.mav:
            logger.error("No MAVLink connection established")
            return None
            
        try:
            msg = self.mav.recv_match(type='RC_CHANNELS', blocking=True, timeout=1)
            if msg:
                # RC channels are accessed as chan1_raw, chan2_raw, etc.
                channel_attr = f'chan{channel_number}_raw'
                if hasattr(msg, channel_attr):
                    return getattr(msg, channel_attr)
            return None
        except Exception as e:
            logger.error(f"Failed to get RC channel {channel_number}: {e}")
            return None


class MissionController:
    """Class to handle mission control operations"""
    
    def __init__(self, drone: DroneController):
        """Initialize with a drone controller"""
        self.drone = drone
        self.stop_event = threading.Event()
        self.fail_event = threading.Event()
    
    def reset_events(self):
        """Reset all control events"""
        self.stop_event.clear()
        self.fail_event.clear()
    
    def monitor_emergency_switch(self):
        """Monitor the emergency switch (channel 8)"""
        logger.info("Starting emergency switch monitor")
        while not self.stop_event.is_set() and not self.fail_event.is_set():
            ch8_value = self.drone.get_rc_channel(8)
            if ch8_value is not None and ch8_value < 1500:
                logger.warning(f"Emergency switch activated! CH8: {ch8_value}")
                self.fail_event.set()
                break
            time.sleep(0.1)  # Reduce CPU usage
    
    def hover_forward(self, pitch_pwm=1600):
        """Control the drone to hover and move forward"""
        logger.info(f"Starting forward hover with pitch PWM: {pitch_pwm}")
        
        while not self.stop_event.is_set() and not self.fail_event.is_set():
            try:
                self.drone.set_rc_channel({
                    2: pitch_pwm,  # Pitch (forward)
                    3: 1590        # Throttle (maintain altitude)
                })
                time.sleep(0.1)  # Update rate limit
            except Exception as e:
                logger.error(f"Error in hover_forward: {e}")
                break
        
        if self.fail_event.is_set():
            logger.warning("Hover forward terminated due to fail event")
            self.drone.disarm()
    
    async def climb_to_altitude(self, target_altitude: float):
        """Ascend to the specified altitude"""
        logger.info(f"Climbing to {target_altitude} meters...")
        
        while not self.fail_event.is_set():
            current_alt = self.drone.get_altitude()
            
            if current_alt is None:
                logger.warning("Failed to get altitude reading")
                await asyncio.sleep(0.1)
                continue
                
            # Set hover throttle
            self.drone.set_rc_channel({3: 1500})
            
            logger.info(f"Current altitude: {current_alt:.2f} m")
            
            if current_alt >= target_altitude:
                logger.info(f"{'-'*20} TARGET ALTITUDE REACHED {'-'*20}")
                logger.info(f"Altitude reached: {current_alt:.2f} meters")
                logger.info(f"{'-'*60}")
                break
                
            await asyncio.sleep(0.2)  # Update rate limit
        
        if self.fail_event.is_set():
            logger.warning("Climb terminated due to fail event")
            self.drone.disarm()
    
    def monitor_ground_speed(self, max_speed: float):
        """Monitor ground speed and trigger stop if threshold exceeded"""
        logger.info(f"Monitoring ground speed (max: {max_speed} m/s)")
        
        while not self.stop_event.is_set() and not self.fail_event.is_set():
            try:
                speed = self.drone.get_ground_speed()
                
                if speed is None:
                    time.sleep(0.1)
                    continue
                    
                if speed > max_speed:
                    logger.warning(f"Ground speed threshold exceeded: {speed:.2f} m/s > {max_speed:.2f} m/s")
                    self.stop_event.set()
                    break
                    
                time.sleep(0.2)  # Update rate limit
            except Exception as e:
                logger.error(f"Error in monitor_ground_speed: {e}")
                time.sleep(0.1)
    
    def wait_for_flight_start(self):
        """Wait for flight start signal on channel 8"""
        logger.info("Waiting for flight start signal (CH8 > 1500)")
        
        while True:
            ch8_value = self.drone.get_rc_channel(8)
            if ch8_value is not None and ch8_value > 1500:
                logger.info(f"Flight start signal received! CH8: {ch8_value}")
                return True
            time.sleep(0.1)  # Reduce CPU usage
    
    async def execute_mission(self, target_altitude: float = 0.5, max_speed: float = 2.0):
        """Execute the complete mission sequence"""
        self.reset_events()
        
        # Wait for start signal
        self.wait_for_flight_start()
        
        # Arm the drone
        if not self.drone.arm():
            logger.error("Failed to arm drone. Mission aborted.")
            return
            
        logger.info("Waiting for arm confirmation...")
        await asyncio.sleep(2)
        
        # Start emergency switch monitor
        emergency_thread = threading.Thread(
            target=self.monitor_emergency_switch, 
            daemon=True
        )
        emergency_thread.start()
        
        # Climb to target altitude
        await self.climb_to_altitude(target_altitude)
        
        # If emergency was triggered during climb, abort
        if self.fail_event.is_set():
            logger.warning("Mission aborted during climb")
            self.drone.disarm()
            return
        
        # Start hover and forward motion
        hover_thread = threading.Thread(
            target=self.hover_forward,
            args=(1600,),
            daemon=True
        )
        
        # Start speed monitor
        speed_thread = threading.Thread(
            target=self.monitor_ground_speed,
            args=(max_speed,),
            daemon=True
        )
        
        # Start both threads
        hover_thread.start()
        speed_thread.start()
        
        # Wait for completion or failure
        while not self.stop_event.is_set() and not self.fail_event.is_set():
            await asyncio.sleep(0.1)
        
        # Ensure proper shutdown
        self.stop_event.set()  # Signal all threads to stop
        
        # Wait for threads to finish
        hover_thread.join(timeout=2)
        speed_thread.join(timeout=2)
        emergency_thread.join(timeout=2)
        
        # Disarm the drone
        self.drone.disarm()
        logger.info("Mission completed")


async def main():
    """Main entry point for the drone mission"""
    # Initialize controllers
    drone = DroneController(port='COM5', baud=57600)
    mission = MissionController(drone)
    
    try:
        # Execute the mission
        await mission.execute_mission(target_altitude=0.5, max_speed=2.0)
    except KeyboardInterrupt:
        logger.info("Mission interrupted by user")
        drone.disarm()
    except Exception as e:
        logger.error(f"Mission failed with error: {e}")
        drone.disarm()


if __name__ == "__main__":
    asyncio.run(main())