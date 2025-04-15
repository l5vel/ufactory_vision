#!/usr/bin/env python3

import socket
import json
import time
import argparse
import subprocess
import threading

class Subscriber:
    def __init__(self, publisher_host, publisher_port, topic_filter=None):
        self.publisher_host = publisher_host
        self.publisher_port = publisher_port
        self.topic_filter = topic_filter
        self.running = True
        self.connect_thread = None
    
    def start_listening(self):
        """Start listening for messages in a separate thread"""
        self.connect_thread = threading.Thread(target=self._listen_for_messages)
        self.connect_thread.daemon = True
        self.connect_thread.start()
        
    def _listen_for_messages(self):
        """Continuously listen for messages from the publisher"""
        while self.running:
            try:
                # Create a new connection to the publisher
                socket_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                socket_conn.connect((self.publisher_host, self.publisher_port))
                
                # Receive data
                data = b""
                while self.running:
                    chunk = socket_conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                
                # Process the received data
                if data:
                    try:
                        message = json.loads(data.decode())
                        topic = message.get("topic", "")
                        
                        # Check if we should process this topic
                        if self.topic_filter is None or topic == self.topic_filter:
                            print(f"Received message on topic '{topic}': {message}")
                            self._handle_message(message)
                    except json.JSONDecodeError:
                        print("Received invalid JSON data")
                
            except ConnectionRefusedError:
                print(f"Connection to {self.publisher_host}:{self.publisher_port} refused. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"Error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            finally:
                try:
                    socket_conn.close()
                except:
                    pass
    
    def _handle_message(self, message):
        """Handle the received message by executing the appropriate code"""
        topic = message.get("topic", "")
        
        if topic == "function_completed":
            # Execute the task when the function completed message is received
            self._execute_task(message)
    
    def _execute_task(self, message):
        """Execute a Python script or task in response to a message"""
        print(f"Executing task in response to message: {message}")
        
        # Example: Run a Python script when message is received
        try:
            result = subprocess.run(
                ["python3", "task_script.py", json.dumps(message)],
                capture_output=True,
                text=True
            )
            print(f"Task execution output: {result.stdout}")
            if result.stderr:
                print(f"Task execution error: {result.stderr}")
        except Exception as e:
            print(f"Failed to execute task: {e}")
    
    def stop(self):
        """Stop the subscriber"""
        self.running = False
        if self.connect_thread:
            self.connect_thread.join(timeout=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Message Subscriber')
    parser.add_argument('--publisher-host', type = str, default = '192.168.68.52', help='Publisher\'s IP address')
    parser.add_argument('--publisher-port', type=int, default=5555, help='Publisher\'s port')
    parser.add_argument('--topic', help='Topic to filter messages (optional)')
    args = parser.parse_args()
    
    subscriber = Subscriber(
        publisher_host=args.publisher_host,
        publisher_port=args.publisher_port,
        topic_filter=args.topic
    )
    
    try:
        print(f"Starting subscriber, connecting to {args.publisher_host}:{args.publisher_port}")
        subscriber.start_listening()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping subscriber...")
    finally:
        subscriber.stop()