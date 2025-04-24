import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mea1k_raw_preproc import read_raw_data
from mea1k_raw_preproc import read_stim_DAC
from mea1k_raw_preproc import get_raw_implant_mapping
from mea1k_raw_preproc import get_recording_implant_mapping
from signal_helpers import estimate_frequency_power

def get_hdf5_fnames_from_dir(subdir):
    fnames, ids = [], []
    print(subdir)
    for fname in sorted(os.listdir(subdir)):
        if fname.endswith('raw.h5'):
            fnames.append(fname)
            # check 4 digit case...
            pruned_fname = fname.replace('.raw.h5', '')
            id_str = pruned_fname.split('_')[-1]
            ids.append(id_str)
    return fnames, ids

def save_output(subdir, data, fname):
    fullpath = os.path.join(subdir, "processed")
    if not os.path.exists(fullpath):
        print("creating processed output dir: ", fullpath)
        os.makedirs(fullpath)
    print("Saving to: ", os.path.join(fullpath, fname))
    data.to_csv(os.path.join(fullpath, fname))

def extract_impedance(subdir, implant_name, current_ampl_nA, debug=False):
    fnames, ids = get_hdf5_fnames_from_dir(subdir)
    aggr_imp_data = []
    for fname, i in zip(fnames, ids):
        print(f"Config {i},{fname} of {len(fnames)}")
        
        # if int(i[-4:]) != 1200:
        #     continue
        # get the config information about this configuration
        stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))


        dac = read_stim_DAC(subdir, fname)
        stim_sample_ids = np.where(dac != 512)[0]
        # shortcut, since we know the stim samples are between 20500 and 29500
        # stim_sample_ids = (20500, 29500)
        data = read_raw_data(subdir, fname, convert2uV=True,
                            subtract_dc_offset=False,) 
        
        if debug:
            # mapping = get_recording_implant_mapping(subdir, fname, implant_name=implant_name,
            #                                         drop_non_bonded=False)
            # viz_mea1k_config(mapping, stim_mea1k_el=stimulated.electrode[stimulated.stim].item())
            # vis_shank_traces(data, mapping, stim_mea1k_el=stimulated.electrode[stimulated.stim].item(), scaler=1/1_000, uVrange=470_000)
            
            plt.subplot(2, 1, 1)
            plt.plot(data.T)
            plt.subplot(2, 1, 2, sharex=plt.gca())              
            plt.plot(read_stim_DAC(subdir, fname))
            plt.show()
    
        mean_ampl = []
        for j,row in enumerate(data):
            if stimulated.stim[j]:
                pass
            _, m_ampl = estimate_frequency_power(row.astype(float)[stim_sample_ids[0]:stim_sample_ids[-1]], 
                                                    sampling_rate=20_000, 
                                                    debug=debug, 
                                                    min_band=960, max_band=1040)
            mean_ampl.append(m_ampl)
        mean_ampl = np.array(mean_ampl)
        
        stimulated['imp_voltage_uV'] = mean_ampl
        stimulated['imp_kOhm'] = (mean_ampl / (current_ampl_nA * 1e-3)) / 1e3 * stimulated.stim.astype(int)
        stimulated['imp_stim_ratio'] = mean_ampl/ mean_ampl[stimulated.stim].item()
        stimulated.drop(columns=['channel', 'x', 'y', 'stim'], inplace=True)
        stimulated.index = pd.MultiIndex.from_product([[fname.replace(".raw.h5","")],
                                                        stimulated.index], names=['config', 'el'])
        print(stimulated)
        aggr_imp_data.append(stimulated)
    
    aggr_imp_data = pd.concat(aggr_imp_data)
    save_output(subdir, aggr_imp_data, "extracted_imp_voltages.csv")

    
def vis_impedance(subdir, name):
    data = pd.read_csv(os.path.join(subdir, "processed", "extracted_imp_voltages.csv"))
    data = data[data.stim_unit.notna()]
    data = data.sort_values(by='pad_id').reset_index(drop=True)
    print(data.stim_unit)
    plt.scatter(data.stim_unit.values, data.imp_kOhm, alpha=.4, s=5, label=f"Stimulated {name}")
    plt.yscale('log')
    plt.legend()
    plt.xlabel("Stimulated unit")
    plt.ylabel("Impedance (kOhm)")
    plt.savefig(os.path.join(subdir, "processed", f"impedance_{name}.png"))
    print(data)
    
def main(): 
    print("Starting in vivo impedance analysis")
    # nas_dir = C.device_paths()[0]
    nas_dir = './nas_imitation'
    
    implant_name = "250308_MEA1K07_H1628pad1shankB6"
    current_ampl_nA = 200 # amplidute == 100 bits, step == 2nA - not sure though
    
    subdirs = [
        f"devices/well_devices/{4983}/recordings//2025-04-24_15.36_FernandoTest_mode=\'small_current\'_with_offset=False_stimpulse=\'sine\'_amplitude=100/",
        f"devices/well_devices/{4983}/recordings/2025-04-24_13.09_FernandoTest_mode='small_current'_with_offset=True_stimpulse='sine'_amplitude=100",
    ]
    # extract_impedance(os.path.join(nas_dir, subdirs[0]), implant_name=implant_name, 
    #                   current_ampl_nA=current_ampl_nA, debug=False)
    # extract_impedance(os.path.join(nas_dir, subdirs[1]), implant_name=implant_name, 
    #                   current_ampl_nA=current_ampl_nA, debug=False)
    
    vis_impedance(os.path.join(nas_dir, subdirs[0]), name="without_offset")
    vis_impedance(os.path.join(nas_dir, subdirs[1]), name="with_offset")
    plt.show()
    
if __name__ == "__main__":
    main()