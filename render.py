import generate_note
import reapy as rp
from reapy import reascript_api as RPR
from pathlib import Path

s = generate_note.Song(200, 50)
path = generate_note.play(s)

with rp.inside_reaper(): 
    project = rp.Project(index=0)
    project.bpm = 200 
    track = None
    if len(project.tracks) < 1:   
        track = project.add_track(index=0, name='drums')
        track.add_fx('Addictive Drums 2 (XLN Audio)')

    RPR.InsertMedia(path,0)
    if project.cursor_position != 0:
        project.cursor_position = 0
    project.play()
