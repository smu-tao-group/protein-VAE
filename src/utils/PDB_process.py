#!/usr/bin/env python

"""
PDB_process class
"""

class Protein():
    """Extract a PDB template and write a set of new coordinates
    into the template PDB.
    """

    def __init__(self):
        """
        Examples
        --------
        Example code to show how to extract a PDB template,
        load new coordinates and write to a new PDB file.

            from PDB_process import Protein


            protein = Protein()
            protein.extract_template("/path/to/template.pdb")
            protein.load_coor(coors)
            protein.write_file("/path/to/new.pdb")
        """

        self.template = []
        self.cur_cor = None

    def extract_template(self, pdb_file):
        """Extract PDB template.

        Parameters
        ----------
        pdb_file : string
            direction to template PDB file
        """

        file = open(pdb_file).readlines()

        for line in file:
            # leave coordinate columns empty
            self.template.append(line[:31] + " " * 23 + line[54:])

    def load_coor(self, coordinates):
        """Load coordinates into the PDB template.

        Parameters
        ----------
        coordinates : list, np.array
            xyz of new coordinates
        """

        # copy template
        self.cur_cor = self.template[:]

        # atom index
        index = 0

        for i in range(len(coordinates) // 3):
            x, y, z = coordinates[3 * i: 3 * (i + 1)]
            x, y, z = "%.3f" % x, "%.3f" % y, "%.3f" % z

            for cor, end in zip([x, y, z], [38, 46, 54]):
                length = len(cor)
                self.cur_cor[index] = self.cur_cor[index][:end - length] \
                    + cor + self.cur_cor[index][end:]

            index += 1

    def write_file(self, file_direction):
        """Write template with new coordinates to a PDB file.

        Parameters
        ----------
        file_direction : string
            direction to write a PDB file
        """

        file = open(file_direction, "w")

        for line in self.cur_cor:
            file.write(line)

        file.close()
