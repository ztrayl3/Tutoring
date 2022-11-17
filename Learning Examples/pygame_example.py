import pygame
import time
import pygame.gfxdraw
import math

# The following is some example code from a memory/imagination fMRI study
# Participants were lying in the MRI machine and asked to recall a memory
# While they recalled the memory (for 30 seconds), brain activity was recorded
# I have removed the MRI-specific code, since it isn't relevant here, but the rest is the same

# Function for filling in the arc of a circle with lines
def fill_arc(center, radius, theta0, theta1, color):
    ndiv = 100
    d_theta = (theta1 - theta0) / ndiv

    for i in range(ndiv):
        x = center[0] + radius * math.cos(theta0 + i * d_theta)
        y = center[1] + radius * math.sin(theta0 + i * d_theta)

        pygame.draw.line(screen, color, center, (x, y), 8)


# Function to wait until a key is pressed, then return the key
def wait(key):
    while True:
        # gets a single event from the event queue
        event = pygame.event.wait()

        # captures the 'KEYDOWN'
        if event.type == pygame.KEYDOWN:
            # gets the key name
            if key and pygame.key.name(event.key) == key:
                return True
            elif not key and key != '5':
                return True


# Function to allow large amounts of text on a screen
def blit_text(surface, text, pos, font, color=pygame.Color('black')):
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


def off_center(surface, xoff, yoff):
    center = (surface.get_width() / 2, surface.get_height() / 2)
    return center[0] + xoff, center[1] + yoff


# Constants needed throughout the experiment
pygame.display.init()
pygame.font.init()
screen = pygame.display.set_mode((800, 600))#, pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
myfont = pygame.font.SysFont('Arial', 30)
black = (0, 0, 0)
gray = (112, 112, 112)
white = (255, 255, 255)
center = (screen.get_width() / 2, screen.get_height() / 2)
# establish instructions
instructions =  "       In this task, you will be asked to imagine scenes\n" \
                "           from events that happened in your own life.\n\n" \
                "              It is important that you imagine the\n" \
                "           scenes as vividly and detailed as possible.\n\n" \
                "                   There will be directions as for\n" \
                "                  what type of memory to imagine.\n\n" \
                "                 Try your best to imagine the event\n" \
                "                     in proper chronological order.\n\n" \
                "                      Press any key to continue..."
# establish cue
cue =   "-Remember an event in your life\n" \
        " that made you very anxious\n\n" \
        "-In the following task, imagine that event\n" \
        " over again from start to finish.\n\n" \
        "-Remember to be as vivid as possible\n" \
        " in your imagination and to stay\n" \
        " in chronological order.\n\n" \
        "   Press any key when you're ready"


# Begin with instructions screen
screen.fill(gray)
blit_text(screen, instructions, off_center(screen, -380, -225), myfont)
pygame.display.update()
wait('')

screen.fill(gray)
blit_text(screen, cue, off_center(screen, -250, -200), myfont)
pygame.display.update()
wait('')

start = time.time()
angle = 0
end = 30  # 30 seconds

while time.time() - start < end:
    darkgray = (105, 105, 105)
    # begin timer for recall
    screen.fill(gray)
    # draw circle timer
    pygame.draw.arc(screen, darkgray, (center[0] - 55, center[1] - 55, 110, 110), 0, 360, 7)
    fill_arc((center[0], center[1]), 50, math.radians(270), math.radians(angle - 90), darkgray)
    pygame.display.update()
    angle = angle + 360 / end
    time.sleep(1)

pygame.display.quit()
pygame.quit()
