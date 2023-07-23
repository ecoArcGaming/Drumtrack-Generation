import collections
import datetime
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import pretty_midi
from typing import Optional, Sequence
import os


'''
for i, note in enumerate(instrument.notes[:20]):
    note_name = pretty_midi.note_number_to_drum_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch = {note.pitch}, note_name = {note_name}, duration = {duration:.4f}')

 '''

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    s = pretty_midi.PrettyMIDI(midi_file)
    instrument = s.instruments[0]
    notes = collections.defaultdict(list) #dictionary with values as lists
    sorted_notes = sorted(instrument.notes, key = lambda note: note.start)
    start = sorted_notes[0].start

    for note in sorted_notes:
        note_start = note.start
        note_end = note.end
        notes['pitch']. append(note.pitch)
        notes['start'].append(note_start)
        notes['end'].append(note_end)
        notes['step'].append(note_start - start) #time since song starts
        notes['duration'].append(note_end - note_start)
        start = note_start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def plot_midi(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20,4))
    plot_pitch = np.stack([notes['pitch'],notes['pitch']], axis=0)
    print(plot_pitch)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:,:count],plot_pitch[:,:count], color='b', marker= '.'
    )
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    _ = plt.title(title)
    plt.show()

def note_to_midi(
        notes: pd.DataFrame, 
        file: str, instrument_name: str, 
        velocity = 100
        ) -> pretty_midi.PrettyMIDI:
    s = pretty_midi.PrettyMIDI() #create MIDI
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)) #create intrument
    note_start = 0
    for i, note in notes.iterrows(): #index of row, and row as a series
        start = float(note_start + note['step'])
        end = float(start+note['duration'])
        note = pretty_midi.Note(
            velocity = velocity,
            pitch = int(note['pitch']),
            start = start,
            end = end)
        instrument.notes.append(note)
        note_start = start
    s.instruments.append(instrument)
    s.write(file)
    return os.path.abspath(file)


raw_notes = midi_to_notes(r"C:\Users\erik3\Downloads\Music Generation\data\bleed.mid")
plot_midi(raw_notes, count = 150)