# code to implement pub sub architecture using nettables4 
# currently used between pi and jetson orin#!/usr/bin/env python3

import ntcore
import time
import argparse
import subprocess
import threading
import signal
import sys

class Subscriber:
    def __init__(self, server_mode=True, server_addr=None, topic_name="navigation_status"):
        # Create an instance for NetworkTables
        self.instance = ntcore.NetworkTableInstance.GetDefault()
        
        # Configure as server or client
        if server_mode:
            # Start a NetworkTables server
            self.instance.startServer()
            print(f"Started NetworkTables server on port {self.instance.getPort()}")
        else:
            # Connect to a NetworkTables server
            self.instance.setServer(server_addr)
            self.instance.startClient4("SubscriberClient")
            print(f"Started NetworkTables client connecting to {server_addr}")
        
        # Get a reference to the topic
        self.topic = self.instance.getTopic(topic_name)
        
        # Subscribe to the topic
        self.sub = self.topic.genericSubscribe("string")
        
        # Set up a listener
        self.listener_thread = None
        self.running = True
        
    def wait_for_connection(self, timeout=10):
        """Wait for connection to NT server or client"""
        start_time = time.time()
        while not self.instance.isConnected():
            if time.time() - start_time > timeout:
                print("Timed out waiting for NT connection")
                return False
            time.sleep(0.1)
        print("NetworkTables connection established!")
        return True
        
    def start_listening(self):
        """Start listening for messages in a separate thread"""
        self.listener_thread = threading.Thread(target=self._listen_for_messages)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        print("Started listening for messages")
        
    def _listen_for_messages(self):
        """Continuously listen for messages"""
        last_message = None
        
        while self.running:
            try:
                # Check for new messages
                message = self.sub.getString("")
                
                # Only process new messages
                if message and message != last_message:
                    print(f"Received message: {message}")
                    last_message = message
                    self._execute_task(message)
                    
                # Sleep a short time to avoid excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error while listening: {e}")
                time.sleep(1)
    
    def _execute_task(self, message):
        """Execute a Python script when a message is received"""
        print(f"Executing task in response to message: {message}")
        
        try:
            # Example: Run a Python script with the message as an argument
            result = subprocess.run(
                ["python3", "task_script.py", message],
                capture_output=True,
                text=True
            )
            
            print(f"Task execution output: {result.stdout}")
            if result.stderr:
                print(f"Task execution error: {result.stderr}")
                
        except Exception as e:
            print(f"Failed to execute task: {e}")
    
    def close(self):
        """Close the subscriber"""
        self.running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1)
        
        self.sub.close()
        if self.instance.isServer():
            self.instance.stopServer()
        else:
            self.instance.stopClient()
        print("Subscriber closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NetworkTables Subscriber')
    parser.add_argument('--mode', choices=['server', 'client'], default='server',
                        help='Run as server or client')
    parser.add_argument('--server', default='127.0.0.1', 
                        help='NetworkTables server address (when in client mode)')
    parser.add_argument('--topic', default='navigation_status', 
                        help='Topic name to subscribe to')
    args = parser.parse_args()
    
    # Create a subscriber based on mode
    server_mode = args.mode == 'server'
    subscriber = Subscriber(
        server_mode=server_mode,
        server_addr=args.server if not server_mode else None,
        topic_name=args.topic
    )
    
    # Setup signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        subscriber.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Wait for connection
        if subscriber.wait_for_connection():
            # Start listening for messages
            subscriber.start_listening()
            
            print("Subscriber is running. Press Ctrl+C to exit.")
            # Keep the main thread alive
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        subscriber.close()