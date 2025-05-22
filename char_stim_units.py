import maxlab
import time
from mea1k_config_utils import create_stim_sine_sequence
from mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df, turn_off_stimulation_units
from mea1k_config_utils import start_saving, stop_saving, try_routing
from glob import glob
from signal_helpers import estimate_frequency_power
from mea1k_raw_preproc import read_stim_DAC
import numpy as np
import pandas as pd
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

def alternative_characterize(rec_name):
    array = maxlab.chip.Array()
    reset_MEA1K(gain=7, enable_stimulation_power=True)
    array.reset()
    array.clear_selected_electrodes()
    # array.connect_all_floating_amplifiers()
    array.download()
    
    s = maxlab.Saving()

    c = maxlab.chip.Core()
        
    seq = create_stim_sine_sequence(dac_id=0, amplitude=1, f=1000, ncycles=400, nreps=1)
    
    seen_stim_units = []
    for ampl_id in range(206, 1024):
        
        
        array.connect_amplifier_to_stimulation(ampl_id)
        # assert result == 'OK', f"Failed to connect amplifier {ampl_id} to stimulation unit"
        stim_unit = array.query_stimulation_at_amplifier(ampl_id)
        if stim_unit == '':
            print(f"Amplifier {ampl_id} not connected to stimulation unit. Skipping.")
            continue
        if (np.array(seen_stim_units) == stim_unit).sum() > 3:
            print(f"Seen stim unit 3 times already. Skipping.")
            continue
        seen_stim_units.append(stim_unit)
        
        array.connect_amplifier_to_ringnode(int(ampl_id))
        array.download()

        c.use_external_port(True)
        maxlab.send(c.use_external_port(True))
        print(f"Connected stim unit {stim_unit} to amplifier {ampl_id}, "
              f"connected to ringnode ampl_id:{array.query_amplifier_at_ringnode()}")
        start_saving(s, dir_name=f"{os.path.dirname(__file__)}/{rec_name}", fname=f"config_ampl_{ampl_id:04d}_stimunit_{int(stim_unit):02d}")
        time.sleep(0.1)
        turn_on_stimulation_units([stim_unit], mode='small_current')
        time.sleep(1.5)
        
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

def post_proc_alt_characterization(rec_name, debug=False):
    dirname = f"{os.path.dirname(__file__)}/{rec_name}"
    sine_amplitudes = []
    for fname in sorted(os.listdir(dirname)):
        if not fname.endswith(".raw.h5"):
            continue
        print(fname.split("_"))
        ampl_id = int(fname.split("_")[-3].replace(".raw.h5", ""))
        stimunit_id = int(fname.split("_")[-1].replace(".raw.h5", ""))
        # if ampl_id < 115:
        #     continue
        print(f"Processing {fname} ", ampl_id, stimunit_id)
        data = read_raw_data(dirname, fname, convert2uV=True,
                             subtract_dc_offset=False)
        
        dac = read_stim_DAC(dirname, fname)
        stim_sample_ids = np.where(dac != 512)[0]
        if len(stim_sample_ids) < 2:
            print("No stimulation found on DAC, skipping")
            continue
        _, m_ampl = estimate_frequency_power(data[ampl_id, stim_sample_ids[0]:stim_sample_ids[-1]].astype(float), 
                                                    sampling_rate=20_000, 
                                                    debug=debug, 
                                                    min_band=960, max_band=1040)
        sine_amplitudes.append((ampl_id, stimunit_id, m_ampl))
        
        if debug:
            plt.figure(figsize=(20, 8))
            plt.plot(data[:100].T, alpha=0.3)
            plt.plot(data[ampl_id], linewidth=1.5, color='red', linestyle='--', 
                     alpha=0.4, label='stimulated')
            plt.plot(dac.astype(float)*1024, linewidth=4, color='black', 
                     linestyle='--', alpha=0.4, label='DAC')
        
        
            plt.title(f"Raw data from {fname}")
            plt.legend()
            plt.show()
            
    sine_amplitudes = pd.DataFrame(sine_amplitudes, columns=["ampl_id", "stimunit_id", "sine_amplitude"])
    print(sine_amplitudes)
    sine_amplitudes.to_csv(os.path.join(dirname, "sine_amplitudes.csv"))
        
def vis_characterization(rec_name):
    dirname = f"{os.path.dirname(__file__)}/{rec_name}"
    sine_amplitudes = pd.read_csv(os.path.join(dirname, "sine_amplitudes.csv"), index_col=None)
    print(sine_amplitudes)
    means = sine_amplitudes.groupby("stimunit_id").mean()
    print(means)
    plt.scatter(sine_amplitudes["stimunit_id"], sine_amplitudes["sine_amplitude"])
    plt.scatter(means.index, means["sine_amplitude"], marker='_')
    plt.title(f"Sine amplitude over 3 amplifiers, {rec_name}")
    plt.xlabel("Stimulation unit ID")
    plt.ylabel("Sine amplitude (uV)")
    plt.savefig(f"./{rec_name}_sine_amplitudes.png")
    plt.show()
        
# origial_characterize()  
# alternative_characterize(rec_name="curmode_rec12_10M_ampl1")
# post_proc_alt_characterization(rec_name="curmode_rec12_10M_ampl1", debug=False)
# vis_characterization(rec_name="curmode_rec12_10M_ampl1",)


# alternative_characterize(rec_name="curmode_rec6_10M_ampl10")
post_proc_alt_characterization(rec_name="curmode_rec6_10M_ampl10", debug=True)
vis_characterization(rec_name="curmode_rec6_10M_ampl10",)