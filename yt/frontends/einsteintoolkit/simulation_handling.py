"""
EinsteinToolkitSimulation class.



"""

from yt.data_objects.time_series import SimulationTimeSeries, DatasetSeries
from yt.units import dimensions
from yt.units.unit_registry import UnitRegistry
from yt.funcs import mylog as log_handler

from .simfactory import SimfactorySimulation
from .parameter_file import ParameterFile, CarpetIOHDF5OutputHandler, CoordBaseDomain

class EinsteinToolkitSimulation(SimulationTimeSeries):
    def __init__(self, simulation_dir, find_outputs=False):
        self.simulation_basedir      = simulation_dir
        self.simulation_type         = "grid"
        self.cosmological_simulation = False

        self._code_mass = 1.0

        super(EinsteinToolkitSimulation, self).__init__(simulation_dir, find_outputs=find_outputs)

    def get_time_series(self, slice_plane=None,
                        initial_iteration=None, final_iteration=None,
                        initial_time=None, final_time=None,
                        iterations=None, times=None,
                        tolerance=None, parallel=True):

        outputs = [o for o in self.all_outputs if o['slice_plane'] == slice_plane]

        if not outputs:
            DatasetSeries.__init__(self, outputs=[], parallel=parallel)
            log_handler.info('Loaded 0 outputs into time series.')
            return

        if iterations is not None:
            outputs = [o for o in outputs if any([o['iteration'] == it for it in iterations])]
        elif times is not None:
            raise NotImplementedError
        elif initial_iteration is not None or final_iteration is not None:
            if initial_iteration is None:
                initial_iteration = self.initial_iteration
            if final_iteration is None:
                final_iteration = self.final_iteration

            outputs = [o for o in outputs if o['iteration'] >= initial_iteration and \
                                             o['iteration'] <= final_iteration]
        elif initial_time is not None or final_time is not None:
            raise NotImplementedError
            
        outputs.sort(key=lambda o: o['iteration'])

        kwargs = dict(code_mass_solar=self.code_mass_solar)
        if slice_plane is None:
            kwargs['file_pattern'] = '*.xyz.h5'
        else:
            kwargs['file_pattern'] = f'*.{slice_plane}.h5'

        DatasetSeries.__init__(self, outputs=outputs, parallel=parallel, **kwargs)
        log_handler.info(f'Loaded {len(outputs)} outputs into time series.')

    def get_dataset(self, iteration=None, time=None, **kwargs):
        kwargs['iterations'] = None if iteration is None else [iteration]
        kwargs['times'] = None if time is None else [time]
        self.get_time_series(**kwargs)

        if len(self._pre_outputs) > 1:
            raise ValueError(f'Ambiguous call to get_dataset(iteration={iteration}, time={time}, kwargs={kwargs}')
        return self[0]

    def __iter__(self):
        for output in self._pre_outputs:
            ds = self._load(output['basedir'], iteration=output['iteration'], **self.kwargs)
            yield ds

    def __getitem__(self, index):
        output = self._pre_outputs[index]
        return self._load(output['basedir'], iteration=output['iteration'], **self.kwargs)

    def _parse_parameter_file(self):
        self.simfactory     = SimfactorySimulation(self.simulation_basedir)
        self.parfile        = ParameterFile(self.simfactory.parameter_file)
        self.output_handler = CarpetIOHDF5OutputHandler(self.parfile)

        if self.parfile.get('Carpet::domain_from_coordbase', default=False):
            self.domain = CoordBaseDomain(self.parfile)
            self.domain_dimensions = self.domain.ncells

    @property
    def code_mass_solar(self):
        return self._code_mass

    @code_mass_solar.setter
    def code_mass_solar(self, value):
        self._code_mass = value
        self._set_units()

    def _set_units(self):
        if not hasattr(self, 'unit_registry'):
            self.unit_registry = UnitRegistry()
            self.unit_registry.add('code_mass', 1.0, dimensions.mass)
            self.unit_registry.add('code_length', 1.0, dimensions.length)
            self.unit_registry.add('code_time', 1.0, dimensions.time)

        self.mass_unit   = self.quan(self.code_mass_solar, 'm_geom')
        self.length_unit = self.quan(self.code_mass_solar, 'l_geom')
        self.time_unit   = self.quan(self.code_mass_solar, 't_geom')

        self.unit_registry.modify('code_mass', self.mass_unit)
        self.unit_registry.modify('code_length', self.length_unit)
        self.unit_registry.modify('code_time', self.time_unit)

        if hasattr(self, 'domain'):
            self.domain_left_edge  = self.domain.xmin*self.length_unit.in_units('code_length')
            self.domain_right_edge = self.domain.xmax*self.length_unit.in_units('code_length')

        self.initial_time = self.simfactory.initial_time*self.time_unit.in_units('code_time')
        self.final_time   = self.simfactory.final_time*self.time_unit.in_units('code_time')

    def _calculate_simulation_bounds(self):
        self.initial_iteration = self.simfactory.initial_iteration
        self.final_iteration   = self.simfactory.final_iteration

    def _get_all_outputs(self, find_outputs=False):
        self.all_outputs = list()
        for restart in self.simfactory.restarts:
            basedir  = restart.output_dir

            if len(restart.h5files[3]) > 0:
                iterations = self.output_handler.get_output_iterations(3, restart.initial_iteration, \
                                                                          restart.final_iteration)

                if restart.restart_index > 0:
                    previous_restart = self.simfactory.restarts[restart.restart_index-1]
                    iterations = [it for it in iterations if it > previous_restart.final_iteration]

                for it in iterations:
                    self.all_outputs.append(dict(basedir=basedir, iteration=it, \
                                                 time=self.simfactory.timestep*it, \
                                                 slice_plane=None))

            for sp in ['xy', 'xz', 'yz']:
                if len(restart.h5files[2][sp]) > 0:
                    iterations = self.output_handler.get_output_iterations(2, restart.initial_iteration, \
                                                                              restart.final_iteration)

                    if restart.restart_index > 0:
                        previous_restart = self.simfactory.restarts[restart.restart_index-1]
                        iterations = [it for it in iterations if it > previous_restart.final_iteration]

                    for it in iterations:
                        self.all_outputs.append(dict(basedir=basedir, iteration=it, \
                                                     time=self.simfactory.timestep*it,
                                                     slice_plane=sp))
