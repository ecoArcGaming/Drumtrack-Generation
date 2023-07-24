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
from pathlib import Path


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


# raw_notes = midi_to_notes(r"C:\Users\erik3\Downloads\Music Generation\data\bleed.mid")
# plot_midi(raw_notes, count = 150)

files = glob.glob(r"C:\Users\erik3\Downloads\Music Generation\data\*")
all_notes = []
for file in files:
    notes = midi_to_notes(file)
    all_notes.append(notes)

all_notes = pd.concat(all_notes)
print('number of notes', len(all_notes))
train = np.stack([all_notes[key] for key in ['pitch', 'step','duration']], axis = 1)
midi_dataset = tf.data.Dataset.from_tensor_slices(train)


def generate(   #supervised, sequence of note = inpute, next note = label
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size = 128
) -> tf.data.Dataset: 
    seq_length += 1
    #window slices the whole sequence into smaller sequences
    windows = dataset.window(seq_length, shift = 1, stride = 1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder = True)
    sequence = windows.flat_map(flatten)

    def normalize_pitch(x):
        x = x/[vocab_size, 1.0, 1.0]
        return x
    def split_labels(sequence): #takes a sequence of notes
        inputs = sequence[:-1] #removes the last element, keeps the input
        labels_dense = sequence[-1] #last element, aka the target
        labels = {key:labels_dense[i] for i, 
                  key in enumerate(['pitch','step','duration'])}
        #creates a dict with pitch, step, and duration values of the target
        return normalize_pitch(inputs), labels
    
    return sequence.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

seq = generate(midi_dataset, seq_length = 50, vocab_size=128)
