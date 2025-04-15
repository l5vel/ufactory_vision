#!/usr/bin/env python3
#
# This is a NetworkTables client (eg, the DriverStation/coprocessor side).
# You need to tell it the IP address of the NetworkTables server (the
# robot or simulator).
#
# When running, this will continue incrementing the value 'dsTime', and the
# value should be visible to other networktables clients and the robot.
#

import sys
import time
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print("Attempting connection with default IP: 172.16.0.12")
    ip = '172.16.0.12'
else:
    ip = sys.argv[1]

NetworkTables.initialize(server=ip)

sd = NetworkTables.getTable("PiJetson")

manip_status = 0
while True:
    nav_status = sd.getNumber("NavCmplt",-1)
    print("Navigation Status:", sd.getNumber("NavCmplt", -1))
    if nav_status == 0: # navigation is not complete
        manip_status = 1
    else:
        manip_status = 0

    sd.putNumber("ManStatus: ", manip_status)

    time.sleep(1)