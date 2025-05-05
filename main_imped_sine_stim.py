import os
import sys
from glob import glob
import time
import datetime
import maxlab

import pandas as pd
import numpy as np

# import ephys_constants as C
from mea1k_config_utils import start_saving, stop_saving, try_routing
from mea1k_config_utils import attampt_connect_el2stim_unit, create_stim_sine_sequence
from mea1k_config_utils import reset_MEA1K, turn_on_stimulation_units, array_config2df, turn_off_stimulation_units

from mea1k_config_utils import create_stim_pulse_sequence
from mea1k_config_utils import create_stim_onoff_sequence

def process_config(config_fullfname, path, rec_time, post_config_wait_time, s, 
                   with_offset, amplitude, mode):
    array = maxlab.chip.Array()
    array.load_config(config_fullfname)
    
    config_map = pd.read_csv(config_fullfname.replace(".cfg", ".csv"))
    # copy to output dir
    for stim_el in config_map.electrode[config_map.stim].tolist():
        success, stim_units = attampt_connect_el2stim_unit(stim_el, array, with_download=False)
        print(f"{stim_units=}") # turn on
        
        # adjust DAC sine wave with offset calulcated earlier 
        adjust_offset_for_stimunit = stim_units[0] if with_offset else None
        stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000, ncycles=400, 
                                             nreps=1, adjust_offset_for_stimunit=adjust_offset_for_stimunit)
        time.sleep(.2)
        
    array.download()
    fname = os.path.basename(config_fullfname).replace(".cfg", "")
    start_saving(s, dir_name=path, fname=fname)
    time.sleep(post_config_wait_time/1/3)
    
    turn_on_stimulation_units(stim_units, mode=mode)
    print(f"Connected ampl: {array.query_amplifier_at_electrode(stim_el)}")
    print(f"Connected ampl stim unit: {array.query_amplifier_at_stimulation(stim_units[0])}")
    # print(f"Connected stim unit at ampl: {array.query_stimulation_at_amplifier(stim_units[0])}")
    print(f"Connected stim unit at electrode: {array.query_stimulation_at_electrode(stim_el)}")
    
    
    time.sleep(post_config_wait_time* 2/3)
    
    print(f"\nStimulating ~ ~ ~ ~ ~ ~ ~ ~ "
          f"on {stim_units} ")
    stim_seq.send()
    time.sleep(rec_time)
    # stimulation
    
    # turn off
    turn_off_stimulation_units(stim_units)
    array.download()
    time.sleep(.1)
    array.close()
    stop_saving(s)

    # hardcoded for now for device 4983
    offset_map = {0: 383, 1: 351, 2: 375, 3: 0, 4: 767, 5: 767, 6: 538, 7: 560, 8: 555, 9: 0, 10: 1023, 11: 703, 12: 703, 13: 208, 14: 695, 15: 343, 16: 767, 17: 767, 18: 767, 19: 156, 20: 1023, 21: 1023, 22: 0, 23: 703, 24: 1023, 25: 0, 26: 343, 27: 0, 28: 252, 29: 0, 30: 703, 31: 1023}
    # a secpond run after a few hourse gave differnt value Stimulation unit 0, Offset: 351 # Stimulation unit 1, Offset: 703 # Stimulation unit 2, Offset: 17 # Stimulation unit 3, Offset: 0 # Stimulation unit 4, Offset: 383 # Stimulation unit 5, Offset: 332 # Stimulation unit 6, Offset: 439 # Stimulation unit 7, Offset: 447 # Stimulation unit 8, Offset: 471 # Stimulation unit 9, Offset: 471 # Stimulation unit 10, Offset: 293 # Stimulation unit 11, Offset: 431 # Stimulation unit 12, Offset: 703
    config_map['stim_unit_offset'] = offset_map[stim_units[0]]
    print(config_map)
    config_map.to_csv(os.path.join(path, os.path.basename(config_fullfname).replace(".cfg", ".csv")))
    

def main():
    # ======== PARAMETERS ========
    subdir = f"devices/well_devices/4983/recordings"
    # nas_dir = C.device_paths()[0]
    nas_dir = os.path.join(os.path.dirname(__file__), 'nas_imitation') # rel paths don't work with mawell saving
    
    print(f"nas_dir: {nas_dir}")
    
    amplitude = 10
    mode = "small_current" # vs large_current mode effect on offset?
    stimpulse = 'sine'
    with_offset = False
    
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M")
    name = "noCAFAmpls"
    rec_dir = f"{t}_{name}_{mode=}_{with_offset=}_{stimpulse=}_{amplitude=}"
    
    post_config_wait_time = 1
    log2file = False
    rec_time = 1
    gain = 7
    configs_basepath = os.path.join(nas_dir, "devices", "implant_devices", 
                                    "250308_MEA1K07_H1628pad1shankB6", 'bonding', )
    which_configs = "imp_rec_configs"
    
    # # stim
    # if stimpulse == 'sine':
    #     stim_seq = create_stim_sine_sequence(dac_id=0, amplitude=amplitude, f=1000, ncycles=400, 
    #                                          nreps=1, voltage_conversion=mode=='voltage')
    # elif stimpulse == 'bursting':
    #     stim_seq = create_stim_pulse_sequence(dac_id=0, amplitude=amplitude, 
    #                                           pulse_duration=167e-6, 
    #                                           inter_phase_interval=67e-6, 
    #                                           frequency=50, 
    #                                           burst_duration=400e-3, nreps=1,
    #                                           voltage_conversion=mode=='voltage')
    
    # elif stimpulse == 'onoff':
    #     stim_seq = create_stim_onoff_sequence(dac_id=0, amplitude=amplitude,
    #                                            pulse_duration=2_000_000, 
    #                                            voltage_conversion=mode=='voltage')
    # ======== PARAMETERS ========
    
    if log2file:
        log_fname = os.path.join(nas_dir, subdir, rec_dir, "log.txt")
        logfile = open(log_fname, "w")
        sys.stdout = logfile
    
    path = os.path.join(nas_dir, subdir, rec_dir)
    print(f"Recording path exists: {os.path.exists(path)} - ", path)
    os.makedirs(path, exist_ok=True)
    
    s = maxlab.Saving()
    reset_MEA1K(gain=gain, enable_stimulation_power=True)
    
    fnames = glob(os.path.join(configs_basepath, which_configs, "*.cfg"))
    print(f"Found {len(fnames)} configs in {configs_basepath}/{which_configs}")
    for i, config_fullfname in enumerate(sorted(fnames)):
        print(f"\nConfig {i+1}/{len(fnames)}: {config_fullfname}", flush=True)
        process_config(config_fullfname, path, rec_time, post_config_wait_time, 
                       s, with_offset=with_offset, amplitude=amplitude, mode=mode)
        # if i>3:
        #     break
    if log2file:
        logfile.close()
        
if __name__ == "__main__":
    main()