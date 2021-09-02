#!/usr/bin/env python

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr, pearsonr
import os
from utils.PDB_process import Protein
from utils.DOPE_score import DOPE


class Evaluation():
    def __init__(self, encoder, decoder, data_scale, scaler):
        '''
        data: testing data
        '''

        self.data_scale = data_scale
        self.encoder = encoder
        self.decoder = decoder
        self.scaler = scaler
        self.spearman = None
        self.pearson = None
        self.dope = None

        # temporary file
        self.temp_file = "./temp.pdb"
        self._process()

    def _process(self):
        '''
        data processing
        '''

        # inverse transform the scaled data back to original data
        self.data = self.scaler.inverse_transform(self.data_scale)

        # encode structure
        self.data_latent = self.encoder.predict(self.data_scale)

        # VAE outputs
        if type(self.data_latent) == list:
            self.data_latent = self.data_latent[0]

        # decode structure
        decoded_structure = self.decoder(self.data_latent).numpy()
        decoded_structure = decoded_structure.reshape(
            self.data_latent.shape[0], -1)

        # scale back to coordinates
        self.decoded_structure = self.scaler.inverse_transform(
            decoded_structure)

        # calculate distances
        self.dist_ori = np.square(
            euclidean_distances(self.data, self.data)
            ).flatten()

        self.dist_encoded = np.square(
            euclidean_distances(self.data_latent, self.data_latent)
            ).flatten()

    def cal_spearman(self, recalculation=False):
        if self.spearman is None or recalculation:
            self.spearman = spearmanr(self.dist_ori, self.dist_encoded)

        return self.spearman

    def cal_pearson(self, recalculation=False):
        if self.pearson is None or recalculation:
            self.pearson = pearsonr(self.dist_ori, self.dist_encoded)

        return self.pearson

    def cal_rmsd(self):
        rmsd = np.sqrt(np.sum(np.square(
            self.decoded_structure * 10 - self.data * 10), axis=1
            ) / (self.data.shape[1] // 3))

        return np.mean(rmsd), np.std(rmsd)

    def cal_dope(self, template_file, recalculation=False, stride=1):
        if self.dope is None or recalculation:
            # protein template
            protein = Protein()
            protein.extract_template(template_file)

            # store all dope scores
            self.dope = []

            combined_data = np.vstack(
                (self.data[::stride], self.decoded_structure[::stride])
                )

            for frame in combined_data:
                protein.load_coor(frame * 10)
                protein.write_file(self.temp_file)
                self.dope.append(DOPE(self.temp_file))

            # remove this temporary file
            os.system("rm %s" % self.temp_file)

        # calculate differences
        dopes_diff = []
        size = len(self.dope) // 2

        for i in range(size):
            # decoded - real
            cur_diff = self.dope[i + size] - self.dope[i]
            dopes_diff.append([cur_diff, cur_diff / self.dope[i] * 100])

        mean_vals = np.mean(dopes_diff, axis=0)

        return mean_vals
