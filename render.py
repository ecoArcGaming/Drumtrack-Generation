import rnn
import reapy as rp
from reapy import reascript_api as RPR
import pandas as pd
import numpy as np 

num_prediction = 120
vocab_size = 128

sample_notes = np.stack([rnn.all_notes[key] for key in ['pitch', 'step','duration']], axis = 1)
init_seq = (sample_notes[:rnn.sequence_length] / np.array([vocab_size, 1, 1]))

generated_note = []
note_start = 0

for _ in range(num_prediction):
    pitch, step, duration = rnn.predict_next_note(init_seq, rnn.model)
    start = note_start + step
    end = start + duration 
    new_note = (pitch, step, duration)
    generated_note.append((*new_note, start, end))
    init_seq = np.delete(init_seq, 0, axis = 0)
    init_seq = np.append(init_seq, np.expand_dims(new_note, 0), axis = 0)
    note_start = start

generated_note = pd.DataFrame(generated_note, columns = (*['pitch','step','duration'], 'start','end'))

song = rnn.note_to_midi(generated_note, file='output.mid',instrument_name='Synth Drum')

with rp.inside_reaper(): 
    project = rp.Project(index=0)
    project.bpm = 200 
    track = None
    if len(project.tracks) < 1:   
        track = project.add_track(index=0, name='drums')
        track.add_fx('Addictive Drums 2 (XLN Audio)')

    RPR.InsertMedia(song,0)
    if project.cursor_position != 0:
        project.cursor_position = 0
    project.play()
