import maxlab
import time
from mea1k_config_utils import create_stim_sine_sequence
from mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df, turn_off_stimulation_units
from mea1k_config_utils import start_saving, stop_saving, try_routing
from glob import glob

import os
def origial_characterize():
    array = maxlab.chip.Array()
    c = maxlab.characterize.StimulationUnitCharacterizer()

    stim_unit_offsets = {}
    for i in range(32):
        o = c.characterize(i)
        stim_unit_offsets[i] = o
        print(f"Stimulation unit {i}, Offset: {o}")
    print(stim_unit_offsets)

def alternative_characterize():
    array = maxlab.chip.Array()
    reset_MEA1K(gain=7, enable_stimulation_power=True)
    array.reset()
    array.clear_selected_electrodes()
    # array.connect_all_floating_amplifiers()
    array.download()
    
    s = maxlab.Saving()

    c = maxlab.chip.Core()\
        
    seq = create_stim_sine_sequence(dac_id=0, amplitude=100, f=1000, ncycles=400, nreps=1)
    
    for ampl_id in range(1024):
        array.connect_amplifier_to_stimulation(ampl_id)
        # assert result == 'OK', f"Failed to connect amplifier {ampl_id} to stimulation unit"
        stim_unit = array.query_stimulation_at_amplifier(ampl_id)
        array.connect_amplifier_to_ringnode(int(ampl_id))

        c.use_external_port(True)
        maxlab.send(c.use_external_port(True))
        array.download()
        print(f"Connected stim unit {stim_unit} to amplifier {ampl_id}, "
              f"connected to ringnode ampl_id:{array.query_amplifier_at_ringnode()}")
        start_saving(s, dir_name=f"{os.path.dirname(__file__)}/testrec", fname=f"config_ampl_{ampl_id:04d}_stimunit_{int(stim_unit):02d}")
        time.sleep(0.1)
        turn_on_stimulation_units([stim_unit])
        
        print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ on stimUnit {stim_unit}, amplifier {ampl_id}")
        seq.send()
        time.sleep(.4)
        
        turn_off_stimulation_units([stim_unit])
        array.disconnect_amplifier_from_stimulation(ampl_id)
        array.disconnect_amplifier_from_ringnode(ampl_id)

        time.sleep(.1)
        stop_saving(s)

from mea1k_raw_preproc import read_raw_data
import matplotlib.pyplot as plt

def simple_analysis():
    dirname = f"{os.path.dirname(__file__)}/testrec"
    for fname in os.listdir(dirname):
        if not fname.endswith(".raw.h5"):
            continue
        ampl_id = int(fname.split("_")[-1].replace(".raw.h5", ""))
        print(f"Processing {fname} ", ampl_id)
        data = read_raw_data(dirname, fname, convert2uV=True,
                             subtract_dc_offset=False)
        print(data.shape)
        
        plt.figure(figsize=(20, 8))
        plt.plot(data[:1024].T, alpha=0.3)
        plt.plot(data[ampl_id], linewidth=1.5, color='red', linestyle='--', alpha=0.4)
        
        plt.title(f"Raw data from {fname}")
        plt.show()
        
        
    
alternative_characterize()

# simple_analysis()