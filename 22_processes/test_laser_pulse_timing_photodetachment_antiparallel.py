# importing relevant packages
import matplotlib.pyplot as plt
import numpy as np
import os
import pybdsim
import pytest
import time

from sklearn.decomposition import PCA
from pathlib import Path

# defining generic plotting and calculation functions
def plot_histogram(t, z, particle, bins=20, cmap='viridis'):
    '''
    '''
    plt.hist2d(t, z, bins=bins, cmap=cmap)

    plt.xlabel(f'Time at production $t$ [ns]')
    plt.ylabel(f'Position at production $z$ [m]')

    plt.colorbar(label=f'Number of {particle}s')

    plt.plot()
    plt.show()  

def calc_gradient(t, z):
    '''
    '''
    # linear regression
    z = np.array(z)
    t = np.array(t)*1e-9 # units: [s]

    m_lr, _ = np.polyfit(t, z, 1)

    # principal component analysis
    data = np.column_stack((t, z))

    pca = PCA(n_components=2)
    pca.fit(data)

    vt, vz = pca.components_[0]
    m_pca = vz/vt

    return m_lr, m_pca

def test():
    '''
    testing laser pulse timings for photodetachment laser antiparallel to the particle beam
    '''
    # changing directory to the directory of the script
    os.chdir(Path(__file__).resolve().parent)

    # defining the relevant filenames
    base_name = 'laser_pulse_timing_photodetachment_antiparallel'
    template_name = f'{base_name}.tpl'
    gmad_name = f'{base_name}.gmad'
    root_name = f'{base_name}.root'

    # defining the relevant data
    data_antiparallel = {
        'LASER_ENERGY': '154.0',
        'BEAM_ENERGY': '0.942022 + 0.003',
        'LASER_OFFSET_THETA': '0'
    }

    # defining the number of primaries to generate
    nprimary = 1000

    # # rendering the gmad template and running the corresponding beam simulation
    # pybdsim.Run.RenderGmadJinjaTemplate(template_name, gmad_name, data_antiparallel)
    # sproc, timing = pybdsim.Run.Bdsim(gmad_name, base_name, ngenerate=nprimary, seed=1, returnTiming=True)

    # load data from generated root file
    d = pybdsim.Data.Load(f'{root_name}')
    samplerData = pybdsim.Data.SamplerData(d, 'laser1')

    # loading the relevant sampler data
    partid = samplerData.data.get('partID')
    charge = samplerData.data.get('charge')

    # initiallising relevant variables
    n_ele, n_h0 = 0, 0
    z_ele, z_h0 = [], []
    t_ele, t_h0 = [], []

    # for every event in the simulation, obtain the event tree from the data file
    for event in d.GetEventTree():

        # retrieve particle trajectory for a given event
        traj = event.Trajectory

        # for each particle trajectory
        for i in range(traj.n):

            # if the particle is an electron
            if traj.partID[i] == 11:

                n_ele += 1

                # obtain parent index and parent step index that produced the secondary
                parent_index = traj.parentIndex[i]
                parent_step = traj.parentStepIndex[i]

                # retrieve the z and t coordinates
                z = traj.XYZ[parent_index][parent_step].Z()
                t = traj.T[parent_index][parent_step]

                z_ele.append(z)
                t_ele.append(t)

            # if the particle is a netural hydrogen atom
            if (traj.partID[i] == 2212) and (charge[i] == 0):

                n_h0 += 1

                parent_index = traj.parentIndex[i]
                parent_step = traj.parentStepIndex[i]

                z = traj.XYZ[parent_index][parent_step].Z()
                t = traj.T[parent_index][parent_step]

                z_h0.append(z)
                t_h0.append(t)

    plot_histogram(t_ele, z_ele, particle='electron')
    plot_histogram(t_h0, z_h0, particle='negative hydrogen ion')

    m_lr_ele, m_pca_ele = calc_gradient(t_ele, z_ele)
    m_lr_h0, m_pca_h0 = calc_gradient(t_h0, z_h0)

    print(m_lr_ele, m_pca_ele)
    print(m_lr_h0, m_pca_h0)

test()
