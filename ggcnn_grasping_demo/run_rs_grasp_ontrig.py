import sys
import time
import signal
from threading import Thread, Event
import logging
from networktables import NetworkTables

# Import the core grasping functionality
from run_rs_grasp import perform_grasp

# Setup logging for NetworkTables
logging.basicConfig(level=logging.DEBUG)

class NetworkTablesGraspTrigger:
    def __init__(self, raspi_ip='172.16.0.12', arm_ip='172.16.0.13'):
        self.raspi_ip = raspi_ip
        self.arm_ip = arm_ip
        self.stop_event = Event()
        self.monitor_thread = None
        self.sd = None
        self.grasp_triggered = False
        self.grasping = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        print("Sigint recv...")
        self.disable_nt()
        sys.exit(0)

    def disable_nt(self):
        print("Shutting down...")
        self.stop_event.set()
        NetworkTables.shutdown()   
    
    def initialize_networktables(self):
        """Initialize connection to NetworkTables"""
        NetworkTables.initialize(server=self.raspi_ip)
        self.sd = NetworkTables.getTable("PiJetson")
        
        # Wait for connection
        while not NetworkTables.isConnected() and not self.stop_event.is_set():
            print("Waiting for NetworkTables connection...")
            time.sleep(1)
            
        if NetworkTables.isConnected():
            print("Connected to NetworkTables!")
            return True
        return False
        
    def background_monitor(self):
        """Monitor NetworkTables for navigation status and trigger grasping"""
        if not self.initialize_networktables():
            print("Failed to connect to NetworkTables")
            return
            
        while not self.stop_event.is_set():
            try:
                nav_status = self.sd.getNumber("NavCmplt", -1)
                print("Navigation Status:", nav_status)
                
                # Set manipulator status based on navigation status
                if nav_status == 0 or nav_status == -1:  # navigation is not complete/NT unreachable
                    manip_status = 0
                elif nav_status == 1:
                    manip_status = 1
                    
                self.sd.putNumber("ManStatus", manip_status)
                
                if manip_status == 1:  # Navigation is complete, start grasping
                    # Give time for the navigation system to stabilize
                    time.sleep(1)
                    self.grasp_triggered = True
                    break
                # Check status periodically
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Exception in monitoring thread: {e}")
                time.sleep(1)  # Continue the loop even after an exception
        
        self.disable_nt() # stop the network tables code
        self.start_grasping()  # Trigger the grasping process
        
    
    def start_grasping(self):
        """Trigger the grasping process"""
        if self.grasp_triggered:
            print("Grasping triggered!")
            perform_grasp(self.arm_ip)
        else:
            print("Grasping already in progress.")

    def start_monitoring(self):
        """Start the background monitoring thread"""
        self.monitor_thread = Thread(target=self.background_monitor, daemon=True)
        self.monitor_thread.start()
        
        # Keep the main thread alive
        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting program")
            self.stop_event.set()
            self.monitor_thread.join(timeout=2)  # Wait for thread to exit

def main():
    if len(sys.argv) < 2:
        print('Usage: {} {{arm_ip}} [raspi_ip]'.format(sys.argv[0]))
        return 1

    arm_ip = sys.argv[1]
    raspi_ip = sys.argv[2] if len(sys.argv) > 2 else '172.16.0.12'
    
    trigger = NetworkTablesGraspTrigger(raspi_ip=raspi_ip, arm_ip=arm_ip)
    trigger.start_monitoring()
    return 0

if __name__ == '__main__':
    sys.exit(main())