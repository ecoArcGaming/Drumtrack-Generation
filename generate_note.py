import random
import numpy as np
import os 
from midiutil import MIDIFile

class Song:
    def __init__(self, bpm, length) -> None:
        self.bpm = bpm
        self.length = length #number of measures
    

    def get_bpm(self) -> int:
        return self.bpm #number of 16th notes per minute
    
    def get_length(self) -> int:
        return self.length #number of 16th notes total

def generate_note():
    pass

def play(Song):
    bpm = Song.get_bpm()
    tn = Song.get_length()
    file = MIDIFile(1) #one track
    file.addTempo(track = 0, time = 0, tempo = bpm)
    for i in range(tn):
        file.addNote(track = 0, channel = 0, time = float(i)/4, pitch = 36, volume = 127, duration=0.25) #36 is C2, the kick
    with open('song.mid', "wb") as f:
        file.writeFile(f)
    return os.path.abspath('song.mid')





    

