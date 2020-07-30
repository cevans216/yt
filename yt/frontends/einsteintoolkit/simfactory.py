"""
Code for handling the directory structure of Simfactory simulations.



"""

import weakref

from os import path as osp
from os import listdir
from glob import glob

class SimfactorySimulation:
    def __init__(self, basedir):
        self.basedir = osp.expanduser(basedir)

        self.simfactory_dir = osp.join(self.basedir, 'SIMFACTORY')
        self.parameter_file = glob(osp.join(self.simfactory_dir, 'par', '*.par'))[0]

        self.name = osp.splitext(osp.basename(self.parameter_file))[0]

        self.restarts = list()
        for d in sorted(listdir(self.basedir)):
            if 'output' in d and 'active' not in d and len(listdir(osp.join(self.basedir, d))) > 1:
                self.restarts.append(SimfactoryRestart(self, int(d.split('-')[-1])))

        self.initial_iteration = min([r.initial_iteration for r in self.restarts])
        self.final_iteration   = max([r.final_iteration for r in self.restarts])

        self.initial_time = min([r.initial_time for r in self.restarts])
        self.final_time   = max([r.final_time for r in self.restarts])

        self.timestep = self.restarts[0].timestep

class SimfactoryRestart:
    def __init__(self, simulation, restart_index):
        self.simulation    = weakref.proxy(simulation)
        self.restart_index = restart_index
        self.basedir       = osp.join(simulation.basedir, f'output-{restart_index:04d}')
        self.output_dir    = osp.join(self.basedir, self.simulation.name)

        self.determine_intervals()

        self.h5files = { 2: dict(xy=list(), xz=list(), yz=list()), \
                         3: list() }
        
        for h5f in glob(osp.join(self.output_dir, '*.h5')):
            base = osp.basename(h5f)
            if '.xy.h5' in base:
                self.h5files[2]['xy'].append(h5f)
            elif '.xz.h5' in base:
                self.h5files[2]['xz'].append(h5f)
            elif '.yz.h5' in base:
                self.h5files[2]['yz'].append(h5f)
            else:
                self.h5files[3].append(h5f)

    def determine_intervals(self):
        formaline = osp.join(self.output_dir, 'formaline-jar.txt')
        assert osp.isfile(formaline), 'Must have formaline-jar.txt!'

        with open(formaline, 'r') as fh:
            flines = fh.readlines()

        iteration_lines        = [l for l in flines if 'cctk_iteration=' in l]
        self.initial_iteration = int(iteration_lines[ 0].split('=',1)[-1])
        self.final_iteration   = int(iteration_lines[-1].split('=',1)[-1])

        time_lines        = [l for l in flines if 'cctk_time=' in l]
        self.initial_time = float(time_lines[ 0].split('=',1)[-1])
        self.final_time   = float(time_lines[-1].split('=',1)[-1])

        self.timestep = self.final_time/self.final_iteration if self.final_iteration > 0 else 1.0
