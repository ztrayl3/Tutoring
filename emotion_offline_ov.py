#!/usr/bin/env python3

import moviepy.editor
import numpy as np
import random
import pygame
import socket
import time


def sendOVstim(ID, sock, t=None, f=4):
    # create the three pieces of the tag: [uint64 flags ; uint64 stimulation_identifier ; uint64 timestamp]
    # flags can be 1 (using fixed point time), 2 (client bakes, e.g. StimulusSender class), 4 (server bakes timestamp)
    # note also that time must be 32:32 fixed point time since boot in milliseconds (hence use of fxp)
    flags = bytearray(f.to_bytes(8, 'little'))
    event_id = bytearray(ID.to_bytes(8, 'little'))

    if t:  # if we have a timestamp, use it!
        timestamp = bytearray.fromhex(t[2:])  # trim the 0x from the front of our timestamp
        timestamp.reverse()  # reverse it to maintain little endianness
    else:  # if we have no timestamp, set to 0 (server will bake)
        timestamp = bytearray((0).to_bytes(8, 'little'))

    sock.sendall(flags)
    sock.sendall(event_id)
    sock.sendall(timestamp)


""" OpenViBE TCP Connections:
This includes both receiving packets from OV (using TCP writer box)
and sending packets to OV (using Acquisition Server TCP tagging).
Note, we might not use both, but they're easy and good to have.
"""
HOST = "127.0.0.1"

# connect to output port
sPORT = 15361

print("Waiting for OpenViBE ACQ...")
out = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # open TCP socket
out.connect((HOST, sPORT))  # connect to port
print("Socket connected (OpenViBE ACQ)")

# connect to input port
rPORT = 5678

print("Waiting for OpenViBE Scenario...")
inn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connected = False
while not connected:
    try:
        inn.connect((HOST, rPORT))
        connected = True
    except Exception as e:
        pass  # Try again
print("Socket connected (OpenViBE Scenario)")

# read the global header before receiving any streams
# all header values are uint32 - do not read 32 bytes at once, just 4 bytes at a time
# variable names sourced from documentation:
# http://openvibe.inria.fr//documentation/3.1.0/Doc_BoxAlgorithm_TCPWriter.html
header = dict(
    Version=np.frombuffer(inn.recv(4), np.uint32)[0],
    Endianness=np.frombuffer(inn.recv(4), np.uint32)[0],
    Frequency=np.frombuffer(inn.recv(4), np.uint32)[0],
    Channels=np.frombuffer(inn.recv(4), np.uint32)[0],
    Samples_per_chunk=np.frombuffer(inn.recv(4), np.uint32)[0]
)
Reserved0 = np.frombuffer(inn.recv(4), np.uint32)[0]
Reserved1 = np.frombuffer(inn.recv(4), np.uint32)[0]
Reserved2 = np.frombuffer(inn.recv(4), np.uint32)[0]
# NOTE: header packet may not work, but values still must be read to clear the pipe
sendOVstim(32769, out, None, 4)  # give us a value to offset later analyses with if we need (Experiment Start = 32769)
start = time.time()  # calculate a start time for internal timestamp logging
inn.close()  # TCP port must be closed here, for some reason causes OV to lag at ~32s


def fixation_cross(screen, duration):
    bg = (0, 0, 0)
    cross = (255, 255, 255)

    center = (screen.get_width() // 2, screen.get_height() // 2)
    screen.fill(bg)
    pygame.draw.line(screen, cross, (center[0] - 400, center[1]), (center[0] + 400, center[1]), 10)
    pygame.draw.line(screen, cross, (center[0], center[1] - 400), (center[0], center[1] + 400), 10)

    pygame.display.update()
    time.sleep(duration)


# Begin actual video presentation:
# Constants:
black = (0, 0, 0)
baseline_start = 32775
baseline_end = 32776
video_end = 800  # End_of_Trial
vids = dict([
    (33025, "Data/v1.mp4"),  # Stim 1
    (33026, "Data/v2.mp4")  # Stim 2
])

keys = list(vids.keys())
random.shuffle(keys)  # randomize the display order

pygame.init()
pygame.display.init()
for key in keys:
    name = vids[key]  # select the matching file
    code = key
    v = moviepy.editor.VideoFileClip(name)  # load video file
    v = v.resize(height=1080)  # resize to screen resolution for fullscreen video
    screen = pygame.display.set_mode(v.size, pygame.FULLSCREEN)  # create pygame display

    # display fixation cross (and notify OV)
    sendOVstim(baseline_start, out, None, 4)
    fixation_cross(screen, 5)
    sendOVstim(baseline_end, out, None, 4)

    # start video (and notify OV)
    sendOVstim(key, out, None, 4)
    v.preview()
    sendOVstim(video_end, out, None, 4)
pygame.quit()

# signal end of experiment
sendOVstim(32770, out, None, 4)  # send marker to OpenViBE (OVTK_StimulationId_ExperimentStop: 32770)
