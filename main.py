#!/usr/bin/env python3
import arbor as A

import subprocess as sp
from pathlib import Path
from time import perf_counter as pc
import h5py as h5

here = Path(__file__).parent

def compile(fn, cat):
    fn = fn.resolve()
    cat = cat.resolve()
    recompile = False
    if fn.exists():
        for src in cat.glob('*.mod'):
            src = Path(src).resolve()
            if src.stat().st_mtime > fn.stat().st_mtime:
                recompile = True
                break
    else:
        recompile = True
    if recompile:
        sp.run(f'arbor-build-catalogue local {cat}', shell=True, check=True)
    return A.load_catalogue(fn)

class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)
        self.props = A.neuron_cable_properties()
        cat = compile(here / 'local-catalogue.so', here / 'cat')
        self.props.catalogue.extend(cat, 'local_')
        self.cell_to_morph = {'L5PC': 'morphology_L5PC', }
        self.gid_to_cell = ['L5PC', ]
        self.i_clamps = {'Input_0': (699.999988079071, 2000.0, 0.7929999989997327), }
        self.gid_to_inputs = { 0: [("seg_0_frac_0.5", "Input_0"), ], }
        self.gid_to_labels = { 0: [(0, 0.5), ], }

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, gid):
        mrf, dec, lbl = self.load_cell(gid)
        # add the labels to our loaded data
        if gid in self.gid_to_labels:
            for seg, frac in self.gid_to_labels[gid]:
                lbl[f'seg_{seg}_frac_{frac}'] = f'(on-components {frac} (segment {seg}))'
        # set discretisation
        dec.discretization(A.cv_policy_every_segment())
        # add inputs to loaded data
        if gid in self.gid_to_inputs:
            for tag, inp in self.gid_to_inputs[gid]:
                lag, dur, amp = self.i_clamps[inp]
                dec.place(f'"{tag}"', A.iclamp(lag, dur, amp), f'ic_{inp}@{tag}')
        # build the cell
        return A.cable_cell(mrf, dec, lbl)

    def probes(self, _):
        return [A.cable_probe_membrane_voltage('(join (root) (location 0 1))'),
                A.cable_probe_membrane_voltage_cell()]

    def global_properties(self, kind):
        return self.props

    def load_cell(self, gid):
        cid = self.gid_to_cell[gid]
        mrf = self.cell_to_morph[cid]
        nml = A.neuroml(f'{here}/mrf/{mrf}.nml').morphology(mrf, allow_spherical_root=True)
        dec = A.load_component(f'{here}/acc/{cid}.acc').component
        lbl = A.label_dict()
        lbl.append(nml.segments())
        lbl.append(nml.named_segments())
        lbl.append(nml.groups())
        lbl['all'] = '(all)'
        return nml.morphology, dec, lbl

# Tuning knobs
T = 100      # overall runtime
dt = 0.025   # timestep for simulation
ds = 0.1     # sampling interval. CAUTION: despite a regular schedule,
             # the sampling times *might* not fall on the ds grid
             # depending on spikes and sampling policy

# Setup
ctx = A.context()
mdl = recipe()
ddc = A.partition_load_balance(mdl, ctx)
sim = A.simulation(mdl, ctx, ddc)

# Sample our two probes
sched = A.regular_schedule(ds)
hdls = [(id, 'voltage', sim.sample(id, sched))
        for id in [(0, 0), # gid=0 probe=0 => root & (location 0 1)
                   (0, 1)] # gid=0 probe=1 => all cables in the cell
        ]

print(f'Running simulation for {T}ms with dt={dt}...')
t0 = pc()
sim.run(T, dt)
t1 = pc()
print(f'Simulation done, took: {t1-t0:.3f}s')

# prep a file for dumping
fd = h5.File(f'results.h5', 'w')

# now extract our data
for (gid, pid), quantity, hdl in hdls:
    # Use the recipe to get hold of the morphology
    mrf, _, _ = mdl.load_cell(gid)

    # This object translates abstract descriptions to 3d points. The isometry
    # can be used to add translation/rotation if needed
    pw = A.place_pwlin(mrf, A.isometry())

    # This is more for demonstration's sake than anything else, normal we'd just
    # write to the H5 file directly, but here we copy arrays around
    samples = []
    xs = []
    ys = []
    zs = []
    times = None
    for data, meta in sim.samples(hdl):
        # We have two cases here. Either we have meta as a *single* mlocation, or
        # meta is a *list* of mcables. Awkward.
        if isinstance(meta, list):
            # we got a list of cables and the data is shaped like
            #        (number_of_samples, number_of_cables + 1)
            # each cable corresponds to a CV and we need to turn
            # this into a mlocation at the midpoint
            for ix, m in enumerate(meta):
                assert(isinstance(m, A.cable))
                loc = A.location(m.branch, 0.5*(m.prox + m.dist))
                pt = pw.at(loc)
                xs.append(pt.x)
                ys.append(pt.y)
                zs.append(pt.z)
                samples.append(data[:, ix])
            times = data[:, 0]
        else:
            assert(isinstance(meta, A.location))
            # we got a single mlocation, that means we are (potentially) repeating the loop
            # across sim.samples to see all locations in the sampled probe.
            # The data is now shaped as
            #        (number_of_samples, 2)
            pt = pw.at(meta)
            xs.append(pt.x)
            ys.append(pt.y)
            zs.append(pt.z)
            times = data[:, 0]
            samples.append(data[:, 1])

    # Here, we have
    #  * a list of x, y, and coords
    #  * a list of samples as 1d np.arrays of length N
    #  * the sample time as one np.array of length N
    # Now start dumping all to disk.
    gp = fd.create_group(f'{gid}/{pid}/{quantity}')
    gp.create_dataset('time', data=times)
    gp.create_dataset('x', data=xs)
    gp.create_dataset('y', data=ys)
    gp.create_dataset('z', data=zs)
    sp = gp.create_group('samples')
    for ix, data in enumerate(samples):
        sp.create_dataset(f'{ix}', data=data)
