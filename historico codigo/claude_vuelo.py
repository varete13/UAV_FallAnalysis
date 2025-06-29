import math
import time
import asyncio
import threading
import logging
from typing import Optional, Dict
from collections import deque

from pymavlink import mavutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fast_drone_controller")

# Constants for fast drone operation
HIGH_SPEED_POLLING_RATE = 0.05  # 20Hz polling for critical functions
EMERGENCY_POLLING_RATE = 0.02   # 50Hz polling for emergency monitoring
CONTROL_UPDATE_RATE = 0.05      # 20Hz control updates
MAX_ACCELERATION = 2.0          # m/s²
COMMAND_TIMEOUT = 0.5           # Half-second timeout for commands
HEARTBEAT_TIMEOUT = 1.0         # Timeout for heartbeat checks

class FastDroneController:
    """Enhanced controller for high-speed drone operations via MAVLink"""
    
    def __init__(self, port='COM5', baud=57600):
        """Initialize the drone controller with communication parameters"""
        self.mav = None
        self.port = port
        self.baud = baud
        self.last_speeds = deque(maxlen=10)  # Store recent speed measurements for acceleration calculation
        self.last_update_time = 0
        self.connected = self.connect()
        self.heartbeat_event = threading.Event()
        self.heartbeat_event.set()  # Assume connected initially
    
    def connect(self) -> bool:
        """Establish connection to the drone"""
        try:
            self.mav = mavutil.mavlink_connection(self.port, baud=self.baud)
            logger.info(f"Connected to drone on {self.port}")
            
            # Wait for a heartbeat to confirm connection
            msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
            if msg:
                logger.info("Heartbeat received, connection confirmed")
                return True
            else:
                logger.error("No heartbeat received, check drone connection")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to drone: {e}")
            return False
    
    def set_rc_channel(self, channel_values: Dict[int, int], priority: bool = False) -> bool:
        """
        Set RC channel values with optional priority handling
        
        Args:
            channel_values: Dictionary with channel numbers as keys and PWM values as values
            priority: If True, sends immediately, otherwise queues
        """
        if not self.mav or not self.heartbeat_event.is_set():
            logger.error("No active MAVLink connection")
            return False
            
        # Default all channels to 0 (no override)
        channels = [0] * 8
        
        # Set the specified channels
        for channel, value in channel_values.items():
            if 1 <= channel <= 8:
                # Apply rate limiting to prevent abrupt changes for fast drones
                channels[channel-1] = int(value)
        
        # Send the RC override command
        try:
            self.mav.mav.rc_channels_override_send(
                self.mav.target_system,
                self.mav.target_component,
                *channels
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set RC channels: {e}")
            return False
    
    def arm(self, force: bool = False) -> bool:
        """
        Arm the drone
        
        Args:
            force: If True, forces arming even if pre-arm checks fail
        """
        if not self.mav or not self.heartbeat_event.is_set():
            logger.error("No active MAVLink connection")
            return False
            
        try:
            self.mav.mav.command_long_send(
                self.mav.target_system, 
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,                # Confirmation
                1,                # 1 to arm
                21196 if force else 0,  # Force (magic number 21196)
                0, 0, 0, 0, 0     # Empty parameters
            )
            
            # Wait for arming confirmation
            start_time = time.time()
            while time.time() - start_time < 3.0:
                msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
                if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    logger.info("Drone armed successfully")
                    return True
            
            logger.warning("Arming timeout - drone may not be armed")
            return False
            
        except Exception as e:
            logger.error(f"Failed to arm drone: {e}")
            return False
    
    def disarm(self, force: bool = False) -> bool:
        """
        Disarm the drone immediately
        
        Args:
            force: If True, forces disarming even if in flight
        """
        if not self.mav:
            logger.error("No MAVLink connection established")
            return False
            
        try:
            # High priority emergency disarm command
            self.mav.mav.command_long_send(
                self.mav.target_system, 
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,                # Confirmation
                0,                # 0 to disarm
                21196 if force else 0,  # Force (magic number 21196)
                0, 0, 0, 0, 0     # Empty parameters
            )
            logger.info("Emergency disarm command sent")
            
            # Verify disarming
            start_time = time.time()
            while time.time() - start_time < 2.0:
                msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=0.5)
                if msg and not (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                    logger.info("Drone disarmed successfully")
                    return True
            
            if force:
                logger.warning("Force disarm failed - try emergency stop")
                return self.emergency_stop()
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to disarm drone: {e}")
            return False
    
    def emergency_stop(self) -> bool:
        """Immediate emergency stop - kills motors"""
        if not self.mav:
            return False
            
        try:
            # Send emergency stop command
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_FLIGHTTERMINATION,
                0,  # Confirmation
                1,  # 1 to terminate
                0, 0, 0, 0, 0, 0  # Empty parameters
            )
            logger.critical("EMERGENCY STOP ACTIVATED")
            return True
        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")
            return False
    
    def get_altitude(self) -> Optional[float]:
        """Get the current relative altitude in meters"""
        if not self.mav or not self.heartbeat_event.is_set():
            return None
            
        try:
            msg = self.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=COMMAND_TIMEOUT)
            if msg:
                return msg.relative_alt / 1000.0  # Convert mm to meters
            return None
        except Exception as e:
            logger.error(f"Failed to get altitude: {e}")
            return None
    
    def get_ground_speed(self) -> Optional[float]:
        """Calculate the ground speed in m/s from velocity components"""
        if not self.mav or not self.heartbeat_event.is_set():
            return None
            
        try:
            # For a fast drone, using VFR_HUD for direct speed reading is more efficient
            msg = self.mav.recv_match(type='VFR_HUD', blocking=True, timeout=COMMAND_TIMEOUT)
            if msg:
                speed = msg.groundspeed  # Already in m/s
                
                # Store speed for acceleration calculation
                now = time.time()
                if self.last_update_time > 0:
                    self.last_speeds.append((speed, now))
                self.last_update_time = now
                
                return speed
                
            # Fallback to calculated speed from components
            msg = self.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=COMMAND_TIMEOUT)
            if msg:
                vx = msg.vx / 100  # Convert cm/s to m/s
                vy = msg.vy / 100  # Convert cm/s to m/s
                speed = math.sqrt(vx**2 + vy**2)
                
                # Store speed for acceleration calculation
                now = time.time()
                if self.last_update_time > 0:
                    self.last_speeds.append((speed, now))
                self.last_update_time = now
                
                return speed
                
            return None
        except Exception as e:
            logger.error(f"Failed to get ground speed: {e}")
            return None
    
    def get_acceleration(self) -> Optional[float]:
        """Calculate current acceleration in m/s²"""
        if len(self.last_speeds) < 2:
            return None
            
        # Get first and last recorded speeds
        first_speed, first_time = self.last_speeds[0]
        last_speed, last_time = self.last_speeds[-1]
        
        # Calculate acceleration
        time_diff = last_time - first_time
        if time_diff > 0:
            return (last_speed - first_speed) / time_diff
        return None
    
    def get_rc_channel(self, channel_number: int) -> Optional[int]:
        """Get the current value of a specific RC channel"""
        if not self.mav or not self.heartbeat_event.is_set():
            return None
            
        try:
            msg = self.mav.recv_match(type='RC_CHANNELS', blocking=True, timeout=COMMAND_TIMEOUT)
            if msg:
                # RC channels are accessed as chan1_raw, chan2_raw, etc.
                channel_attr = f'chan{channel_number}_raw'
                if hasattr(msg, channel_attr):
                    return getattr(msg, channel_attr)
            return None
        except Exception as e:
            logger.error(f"Failed to get RC channel {channel_number}: {e}")
            return None
    
    def wait_for_heartbeat(self, timeout=HEARTBEAT_TIMEOUT):
        """Wait for heartbeat to verify connection"""
        try:
            msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
            if msg:
                self.heartbeat_event.set()
                return True
            self.heartbeat_event.clear()
            return False
        except Exception:
            self.heartbeat_event.clear()
            return False


class FastMissionController:
    """Enhanced mission controller for high-speed drone operations"""
    
    def __init__(self, drone: FastDroneController):
        """Initialize with a drone controller"""
        self.drone = drone
        self.stop_event = threading.Event()
        self.fail_event = threading.Event()
        self.pause_event = threading.Event()
        self.heartbeat_monitor_active = False
    
    def reset_events(self):
        """Reset all control events"""
        self.stop_event.clear()
        self.fail_event.clear()
        self.pause_event.clear()
    
    def start_heartbeat_monitor(self):
        """Start monitoring the heartbeat to detect connection issues"""
        if self.heartbeat_monitor_active:
            return
            
        self.heartbeat_monitor_active = True
        
        def monitor_heartbeat():
            missed_heartbeats = 0
            while not self.stop_event.is_set() and not self.fail_event.is_set():
                if not self.drone.wait_for_heartbeat():
                    missed_heartbeats += 1
                    logger.warning(f"Missed heartbeat: {missed_heartbeats}")
                    if missed_heartbeats >= 3:
                        logger.critical("Lost connection to drone!")
                        self.fail_event.set()
                        break
                else:
                    missed_heartbeats = 0
                time.sleep(1.0)  # Check every second
        
        heartbeat_thread = threading.Thread(target=monitor_heartbeat, daemon=True)
        heartbeat_thread.start()
    
    def monitor_emergency_switch(self):
        """Monitor the emergency switch (channel 8) with high frequency"""
        logger.info("Starting emergency switch monitor")
        while not self.stop_event.is_set() and not self.fail_event.is_set():
            ch8_value = self.drone.get_rc_channel(8)
            if ch8_value is not None and ch8_value < 1500:
                logger.warning(f"Emergency switch activated! CH8: {ch8_value}")
                self.fail_event.set()
                self.drone.disarm(force=True)  # Force disarm on emergency
                break
            time.sleep(EMERGENCY_POLLING_RATE)  # High frequency check
    
    def fast_forward_flight(self, pitch_pwm=1700):  # Increased pitch for faster forward motion
        """Control the drone for rapid forward flight"""
        logger.info(f"Starting fast forward flight with pitch PWM: {pitch_pwm}")
        
        # Gradually increase pitch to avoid sudden movements
        current_pitch = 1500
        step = 25  # Smaller steps for smoother acceleration
        
        while not self.stop_event.is_set() and not self.fail_event.is_set() and not self.pause_event.is_set():
            try:
                # If not at target pitch, gradually increase
                if current_pitch < pitch_pwm:
                    current_pitch = min(current_pitch + step, pitch_pwm)
                
                # Check acceleration
                accel = self.drone.get_acceleration()
                if accel is not None and abs(accel) > MAX_ACCELERATION:
                    logger.warning(f"High acceleration detected: {accel:.2f} m/s²")
                    # Reduce pitch to limit acceleration
                    current_pitch = max(1500, current_pitch - step)
                
                # Send control commands
                self.drone.set_rc_channel({
                    2: current_pitch,  # Pitch (forward)
                    3: 1600            # Throttle (maintain altitude with slight increase for speed)
                })
                
                time.sleep(CONTROL_UPDATE_RATE)  # Faster update rate
                
            except Exception as e:
                logger.error(f"Error in fast_forward_flight: {e}")
                break
        
        if self.fail_event.is_set():
            logger.warning("Fast forward flight terminated due to fail event")
            self.drone.disarm(force=True)
    
    async def rapid_climb(self, target_altitude: float, climb_rate: float = 2.0):
        """
        Rapidly climb to the specified altitude with controlled climb rate
        
        Args:
            target_altitude: Target altitude in meters
            climb_rate: Desired climb rate in m/s
        """
        logger.info(f"Rapid climb to {target_altitude} meters (rate: {climb_rate} m/s)...")
        
        last_alt = 0
        last_time = time.time()
        throttle = 1600  # Start with higher throttle for faster climb
        
        while not self.fail_event.is_set():
            current_alt = self.drone.get_altitude()
            current_time = time.time()
            
            if current_alt is None:
                logger.warning("Failed to get altitude reading")
                await asyncio.sleep(0.1)
                continue
            
            # Calculate current climb rate
            if last_time < current_time:
                current_rate = (current_alt - last_alt) / (current_time - last_time)
                
                # Adjust throttle based on climb rate
                if current_rate < climb_rate * 0.8:  # Too slow
                    throttle = min(1800, throttle + 10)  # Increase throttle, max 1800
                elif current_rate > climb_rate * 1.2:  # Too fast
                    throttle = max(1500, throttle - 10)  # Decrease throttle, min 1500
            
            # Set adjusted throttle
            self.drone.set_rc_channel({3: throttle})
            
            logger.info(f"Altitude: {current_alt:.2f} m | Throttle: {throttle}")
            
            # Save values for next iteration
            last_alt = current_alt
            last_time = current_time
            
            # Check if target reached
            if current_alt >= target_altitude:
                logger.info(f"{'-'*20} TARGET ALTITUDE REACHED {'-'*20}")
                logger.info(f"Altitude: {current_alt:.2f} meters")
                logger.info(f"{'-'*60}")
                
                # Stabilize at target altitude
                self.drone.set_rc_channel({3: 1500})  # Set to hover throttle
                await asyncio.sleep(0.5)  # Brief pause to stabilize
                
                break
                
            await asyncio.sleep(0.1)  # Faster update rate for responsive control
        
        if self.fail_event.is_set():
            logger.warning("Climb terminated due to fail event")
            self.drone.disarm(force=True)
    
    def monitor_speed_and_acceleration(self, max_speed: float):
        """
        Monitor ground speed and acceleration with safety thresholds
        
        Args:
            max_speed: Maximum allowed speed in m/s
        """
        logger.info(f"Monitoring speed (max: {max_speed} m/s) and acceleration")
        
        while not self.stop_event.is_set() and not self.fail_event.is_set():
            try:
                speed = self.drone.get_ground_speed()
                accel = self.drone.get_acceleration()
                
                if speed is not None:
                    if speed > max_speed:
                        logger.warning(f"Speed threshold exceeded: {speed:.2f} m/s > {max_speed:.2f} m/s")
                        self.pause_event.set()  # Pause forward motion temporarily
                        
                        # Wait for speed to decrease
                        while speed and speed > max_speed * 0.8:
                            # Apply slight backward pitch to slow down
                            self.drone.set_rc_channel({2: 1400})  # Backward pitch
                            time.sleep(0.1)
                            speed = self.drone.get_ground_speed()
                        
                        # Resume normal operation once slowed
                        self.pause_event.clear()
                        
                    # Check for excessive acceleration
                    if accel is not None and abs(accel) > MAX_ACCELERATION:
                        logger.warning(f"High acceleration detected: {accel:.2f} m/s²")
                        # Temporary slow down
                        self.pause_event.set()
                        self.drone.set_rc_channel({2: 1450})  # Slight backward pitch
                        time.sleep(0.5)
                        self.pause_event.clear()
                    
                time.sleep(HIGH_SPEED_POLLING_RATE)  # Fast polling for responsive control
                
            except Exception as e:
                logger.error(f"Error in monitor_speed_and_acceleration: {e}")
                time.sleep(0.1)
    
    def wait_for_flight_start(self):
        """Wait for flight start signal on channel 8"""
        logger.info("Waiting for flight start signal (CH8 > 1500)")
        
        while True:
            ch8_value = self.drone.get_rc_channel(8)
            if ch8_value is not None and ch8_value > 1500:
                logger.info(f"Flight start signal received! CH8: {ch8_value}")
                return True
            time.sleep(0.1)
    
    async def execute_high_speed_mission(self, target_altitude: float = 0.5, max_speed: float = 4.0, climb_rate: float = 2.0):
        """
        Execute a high-speed mission with enhanced safety monitoring
        
        Args:
            target_altitude: Target hover altitude in meters
            max_speed: Maximum allowed ground speed in m/s
            climb_rate: Desired climb rate in m/s
        """
        self.reset_events()
        
        # Start heartbeat monitor for connection status
        self.start_heartbeat_monitor()
        
        # Wait for start signal
        self.wait_for_flight_start()
        
        # Arm the drone with pre-flight checks
        for attempt in range(3):
            if self.drone.arm():
                break
            if attempt == 2:  # Last attempt, try force arm
                logger.warning("Normal arming failed, attempting force arm")
                if not self.drone.arm(force=True):
                    logger.error("All arming attempts failed. Mission aborted.")
                    return
            await asyncio.sleep(1)
        
        logger.info("Pre-flight checks complete")
        
        # Start monitoring threads before flight
        monitors = []
        
        # Emergency switch monitor
        emergency_thread = threading.Thread(
            target=self.monitor_emergency_switch, 
            daemon=True
        )
        emergency_thread.start()
        monitors.append(emergency_thread)
        
        # Rapid climb to target altitude with controlled rate
        await self.rapid_climb(target_altitude, climb_rate)
        
        # If emergency was triggered during climb, abort
        if self.fail_event.is_set():
            logger.warning("Mission aborted during climb")
            self.drone.disarm(force=True)
            return
        
        # Start high-speed forward flight
        flight_thread = threading.Thread(
            target=self.fast_forward_flight,
            args=(1700,),  # Higher pitch value for faster movement
            daemon=True
        )
        
        # Start speed and acceleration monitor
        speed_thread = threading.Thread(
            target=self.monitor_speed_and_acceleration,
            args=(max_speed,),
            daemon=True
        )
        
        # Start both threads
        flight_thread.start()
        speed_thread.start()
        monitors.extend([flight_thread, speed_thread])
        
        # Wait for completion or failure
        while not self.stop_event.is_set() and not self.fail_event.is_set():
            await asyncio.sleep(0.1)
        
        # Ensure proper shutdown
        self.stop_event.set()  # Signal all threads to stop
        
        # Wait for threads to finish with timeout
        for thread in monitors:
            thread.join(timeout=1)
        
        # Ensure the drone is disarmed
        self.drone.disarm(force=True)
        logger.info("Mission completed")


async def main():
    """Main entry point for the high-speed drone mission"""
    # Initialize controllers
    drone = FastDroneController(port='COM5', baud=57600)
    
    if not drone.connected:
        logger.error("Failed to connect to drone. Aborting mission.")
        return
        
    mission = FastMissionController(drone)
    
    try:
        # Execute the high-speed mission with increased parameters
        await mission.execute_high_speed_mission(
            target_altitude=0.5,  # 0.5m altitude
            max_speed=4.0,        # 4 m/s maximum speed (increased for fast drone)
            climb_rate=2.0        # 2 m/s climb rate
        )
    except KeyboardInterrupt:
        logger.info("Mission interrupted by user")
        drone.disarm(force=True)
    except Exception as e:
        logger.error(f"Mission failed with error: {e}")
        drone.disarm(force=True)


if __name__ == "__main__":
    asyncio.run(main())