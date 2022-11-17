#!/usr/bin/env python3
import moviepy.editor
import numpy as np
import random
import pygame
import pandas
import socket
import time

ID = 0  # SET THE SUBJECT ID FIRST!

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


def blit_text(surface, text, pos, font, color):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.


def wait(key, capture=False):
    while True:
        # gets a single event from the event queue
        event = pygame.event.wait()
        # captures the 'KEYDOWN'
        if event.type == pygame.KEYDOWN:
            # gets the key name
            if key and pygame.key.name(event.key) == key:
                if capture:
                    return pygame.key.name(event.key)
                return True
            elif not key:  # if no key is specified, work anyway
                if capture:
                    return pygame.key.name(event.key)
                return True


def off_center(surface, xoff, yoff):
    center = (surface.get_width() / 2, surface.get_height() / 2)
    return center[0] + xoff, center[1] + yoff


# Constants:
black = pygame.Color("black")
white = pygame.Color("white")
baseline_start = 32775
baseline_end = 32776
video_end = 800  # End_of_Trial
ITI = 3  # inter-trial interval in seconds
vids = dict([
    (33025, "Data/v1.mp4"),  # Stim 1
    (33026, "Data/v2.mp4"),  # Stim 2
    (33027, "Data/v3.mp4"),  # Stim 3
    (33028, "Data/v4.mp4"),  # Stim 4
    (33029, "Data/v5.mov")   # Stim 5
])

keys = list(vids.keys())
random.shuffle(keys)  # randomize the display order

# Responses:
fill = np.zeros((2, len(keys)))
results = pandas.DataFrame(data=fill.T, index=keys, columns=["Valence", "Arousal"])

# setup all pygame variables
pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)  # create pygame display
center = (screen.get_width() // 2, screen.get_height() // 2)  # useful for later
valence = pygame.image.load("../Data/valence.png").convert()  # both our emotional scale values
arousal = pygame.image.load("../Data/arousal.png").convert()

intro = "       In this task, you will watch a series of videos\n" \
        "                 that evoke different emotions.\n\n" \
        "             Each video will be preceded by a short\n" \
        "      fixation cross (to relax) before the video starts.\n\n" \
        "       Each video will be followed by two questions\n" \
        "              about how the video made you feel.\n\n" \
        "                Answer to the best of your ability!\n" \
        "                 There will be 5 videos in total.\n\n" \
        "                    Press any key to continue..."
myfont = pygame.font.SysFont('Arial', 30)
blit_text(screen, intro, off_center(screen, -300, -225), myfont, white)
pygame.display.update()
wait('')

for key in keys:
    name = vids[key]  # select the matching file
    code = key
    v = moviepy.editor.VideoFileClip(name)  # load video file
    v = v.resize(height=1080)  # resize to screen resolution for fullscreen video

    # display fixation cross (and notify OV)
    sendOVstim(baseline_start, out, None, 4)
    fixation_cross(screen, 2)
    sendOVstim(baseline_end, out, None, 4)

    # start video (and notify OV)
    sendOVstim(key, out, None, 4)
    v.preview()
    sendOVstim(video_end, out, None, 4)

    # present valence question
    screen.fill(black)
    screen.blit(valence, off_center(screen, -valence.get_width()/2, -valence.get_height()/2))
    pygame.display.update()
    results["Valence"][key] = (wait('', capture=True))

    # present arousal question
    screen.fill(black)
    screen.blit(arousal, off_center(screen, -arousal.get_width()/2, -arousal.get_height()/2))
    pygame.display.update()
    results["Arousal"][key] = (wait('', capture=True))

    # wait for ITI seconds before next video
    screen.fill(black)
    pygame.display.update()
    time.sleep(ITI)

pygame.quit()

# signal end of experiment
sendOVstim(32770, out, None, 4)  # send marker to OpenViBE (OVTK_StimulationId_ExperimentStop: 32770)
results.to_csv("{}_offline.csv".format(ID))
