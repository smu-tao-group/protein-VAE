#!/usr/bin/env python

"""
align simulation trajs
"""

import MDAnalysis as mda
from MDAnalysis.analysis import align

# load trajs
open_state_1 = mda.Universe("../trajs/4ake.psf", "../trajs/4ake-01.dcd")
open_state_2 = mda.Universe("../trajs/4ake.psf", "../trajs/4ake-02.dcd")
open_state_3 = mda.Universe("../trajs/4ake.psf", "../trajs/4ake-03.dcd")

close_state_1 = mda.Universe("../trajs/1ake.psf", "../trajs/1ake-01.dcd")
close_state_2 = mda.Universe("../trajs/1ake.psf", "../trajs/1ake-02.dcd")
close_state_3 = mda.Universe("../trajs/1ake.psf", "../trajs/1ake-03.dcd")

# combine all trajs
total_trajs = [open_state_1, open_state_2, open_state_3,
               close_state_1, close_state_2, close_state_3]

names = ["4ake-01-align", "4ake-02-align", "4ake-03-align",
         "1ake-01-align", "1ake-02-align", "1ake-03-align"]

for i in range(len(names)):
    align.AlignTraj(total_trajs[i], total_trajs[0], select='name CA',
                    filename="../trajs/" + names[i] + ".dcd",
                    match_atoms=True).run()
