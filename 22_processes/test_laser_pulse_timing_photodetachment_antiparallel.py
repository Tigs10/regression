import numpy as np
import os
import pybdsim
import pytest

from sklearn.decomposition import PCA
from pathlib import Path

def calc_gradient(t, z):
    '''calculate gradient of particle position vs time using linear regression and principal component analysis'''
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

def test(test_length, testlength_primaries, testdata_store):
    '''test laser pulse timings for photodetachment laser antiparallel to the particle beam'''

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
    nprimary = testlength_primaries.get_nprimary(__file__, test_length)

    # # rendering the gmad template and running the corresponding beam simulation
    pybdsim.Run.RenderGmadJinjaTemplate(template_name, gmad_name, data_antiparallel)
    sproc, timing = pybdsim.Run.Bdsim(gmad_name, base_name, ngenerate=nprimary, seed=1, returnTiming=True)

    # load data from generated root file
    d = pybdsim.Data.Load(f'{root_name}')
    samplerData = pybdsim.Data.SamplerData(d, 'laser1')

    # loading the charge data from sampler
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
            if partid[i] == 11:

                n_ele += 1

                # obtain parent index and parent step index that produced the secondary
                parent_index = traj.parentIndex[i]
                parent_step = traj.parentStepIndex[i]

                # retrieve the z and t coordinates
                z = traj.XYZ[parent_index][parent_step].Z()
                t = traj.T[parent_index][parent_step]

                z_ele.append(z)
                t_ele.append(t)

            # if the particle is a neutral hydrogen atom
            if (traj.partID[i] == 2212) and (charge[i] == 0):

                n_h0 += 1

                parent_index = traj.parentIndex[i]
                parent_step = traj.parentStepIndex[i]

                z = traj.XYZ[parent_index][parent_step].Z()
                t = traj.T[parent_index][parent_step]

                z_h0.append(z)
                t_h0.append(t)

    # calculate gradients
    m_lr_ele, m_pca_ele = calc_gradient(t_ele, z_ele)
    m_lr_h0, m_pca_h0 = calc_gradient(t_h0, z_h0)

    # reference gradients
    ref_m_lr_ele, ref_m_pca_ele = -257589161.82326686, -306558144.2939847
    ref_m_lr_h0, ref_m_pca_h0 = -254264354.88906184, -306845014.2525596

    # storing test entry with gradients, and asserting.
    te = testdata_store.new_test_entry("22_processes", __file__, nprimary, timing)

    te.add_output_parameter("m_lr_ele", m_lr_ele)
    te.add_output_parameter("m_pca_ele", m_pca_ele)
    te.add_output_parameter("m_lr_h0", m_lr_h0)
    te.add_output_parameter("m_pca_h0", m_pca_h0)

    assert (m_lr_ele==ref_m_lr_ele)
    assert (m_pca_ele==ref_m_pca_ele)
    assert (m_lr_h0==ref_m_lr_h0)
    assert (m_pca_h0==ref_m_pca_h0)
