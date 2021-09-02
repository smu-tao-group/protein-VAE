#!/usr/bin/env python

import pickle
import numpy as np
import mdtraj as md

# extract xyz
# load aligned trajs
open_state_1 = md.load("../trajs/4ake-01-align.dcd", top="../trajs/4ake.psf")
open_state_2 = md.load("../trajs/4ake-02-align.dcd", top="../trajs/4ake.psf")
open_state_3 = md.load("../trajs/4ake-03-align.dcd", top="../trajs/4ake.psf")

close_state_1 = md.load("../trajs/1ake-01-align.dcd", top="../trajs/1ake.psf")
close_state_2 = md.load("../trajs/1ake-02-align.dcd", top="../trajs/1ake.psf")
close_state_3 = md.load("../trajs/1ake-03-align.dcd", top="../trajs/1ake.psf")

# extract heavy atom indices
selected_atoms = open_state_1.topology.select_atom_indices("heavy")

# combine all trajs
total_trajs = [open_state_1, open_state_2, open_state_3,
               close_state_1, close_state_2, close_state_3]

# store xyz data
maps = []

for traj in total_trajs:
    cur_coor = traj.atom_slice(selected_atoms).xyz
    maps += cur_coor.reshape(cur_coor.shape[0], -1).tolist()

maps = np.array(maps)
maps = maps.reshape(maps.shape[0], -1)

pickle.dump(maps, open("../results/maps.pkl", "wb"))
