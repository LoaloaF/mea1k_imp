import maxlab

array = maxlab.chip.Array()
c = maxlab.characterize.StimulationUnitCharacterizer()

stim_unit_offsets = {}
for i in range(32):
    o = c.characterize(i)
    stim_unit_offsets[i] = o
    print(f"Stimulation unit {i}, Offset: {o}")
print(stim_unit_offsets)