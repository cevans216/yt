"""
Utility functions for yt.frontends.einsteintoolkit



"""
import numpy as np

def cast_parameter_value(vstr):
    try:
        return int(vstr)
    except:
        try:
            return float(vstr)
        except:
            if vstr.lower() in ['t', 'true', 'yes']:
                return True
            elif vstr.lower() in ['f', 'false', 'no']:
                return False
            else:
                return vstr

class ParameterFile:
    def __init__(self, parameter_file):
        self.active_thorns = list()
        self.parameters    = dict()

        self.parse_parameter_file(parameter_file)

    def parse_parameter_file(self, path):
        with open(path, 'r') as parfile:
            lines = parfile.readlines()

        for line in (l.strip() for l in lines):
            if '#' in line:
                line = line[0:line.find('#')]
            if len(line) < 2:
                continue

            par, vstr = (p.strip() for p in line.split('=', 1))
            par  = par.lower()
            vstr = vstr.replace('"', '')

            if par == 'activethorns':
                self.active_thorns.extend([t.strip() for t in vstr[1:-1].split()])
            else:
                self.parameters[par] = cast_parameter_value(vstr)

    def __contains__(self, key):
        return key.lower() in self.parameters

    def __getitem__(self, key):
        return self.parameters[key.lower()]
        
    def get(self, key, default=None):
        try:
            return self.parameters[key.lower()]
        except KeyError as ex:
            if default is not None:
                return default
            raise ex

class CarpetIOHDF5OutputHandler:
    def __init__(self, parfile):
        io_out_criterion = parfile.get('IO::out_criterion', default='iteration')
        io_out_every     = parfile.get('IO::out_every'    , default=-1)
        io_out_dt        = parfile.get('IO::out_dt'       , default=-2)

        self.out_criterion = dict()
        self.out_every     = dict()
        self.out_dt        = dict()

        for dim in [2, 3]:
            out = 'out' if dim == 3 else 'out2D'
            self.out_criterion[dim] = parfile.get(f'CarpetIOHDF5::{out}_criterion', default=io_out_criterion)
            self.out_every    [dim] = parfile.get(f'CarpetIOHDF5::{out}_every'    , default=io_out_every)
            self.out_dt       [dim] = parfile.get(f'CarpetIOHDF5::{out}_dt'       , default=io_out_dt)

    def output_enabled(self, dim):
        if self.out_criterion[dim] == 'never':
            return False
        elif self.out_criterion[dim] == 'iteration':
            return self.out_every[dim] >= 1
        elif self.out_criterion[dim] == 'time':
            return self.out_enabled[dim] >= 0
        else:
            raise ValueError(self.out_criterion[dim])

    def get_output_iterations(self, dim, initial, final):
        if not self.output_enabled(dim):
            return list()

        if self.out_criterion[dim] == 'iteration':
            out_every = self.out_every[dim]
            initial = out_every*int(np.ceil(initial/out_every))
            return np.array([initial + x*out_every for x in range(int((final-initial)/out_every)+1)], dtype=int)
        else:
            raise NotImplementedError(self.out_criterion[dim])

class CoordBaseDomain:
    def __init__(self, parfile):
        self.domainsize = parfile.get('CoordBase::domainsize', default='minmax')
        self.spacing    = parfile.get('CoordBase::spacing', default='gridspacing')

        # First get the physical domain
        if self.domainsize == 'minmax':
            self.xmin = np.array([parfile.get(f'CoordBase::{x}min', default=0.0) for x in 'xyz'], dtype=float)
            self.xmax = np.array([parfile.get(f'CoordBase::{x}max', default=1.0) for x in 'xyz'], dtype=float)

            if self.spacing == 'gridspacing':
                self.coarse_dx = np.array([parfile.get(f'CoordBase::d{x}', default=1.0) for x in 'xyz'], dtype=float)
                self.ncells    = np.rint((self.xmax - self.xmin)/self.coarse_dx).astype(int)
            else:
                self.ncells    = np.array([parfile.get(f'CoordBase::ncells_{x}', default=1) for x in 'xyz'], dtype=int)
                self.coarse_dx = (self.xmax - self.xmin)/self.ncells
        elif self.domainsize == 'extent':
            xmin               = np.array([parfile.get(f'CoordBase::{x}min', default=0.0) for x in 'xyz'], dtype=float)
            self.xextent       = np.array([parfile.get(f'CoordBase::{x}extent', default=1.0) for x in 'xyz'], dtype=float)
            self.zero_origin_x = np.array([parfile.get(f'CoordBase::zero_origin_{x}', default=False) for x in 'xyz'], dtype=bool)

            self.xmin = -0.5*self.xextent
            self.xmax =  0.5*self.xextent

            self.xmin[self.zero_origin_x] = xmin[self.zero_origin_x]
            self.xmax[self.zero_origin_x] = xmin[self.zero_origin_x] + self.xextent[self.zero_origin_x]

            if self.spacing == 'gridspacing':
                self.coarse_dx = np.array([parfile.get(f'CoordBase::d{x}', default=1.0) for x in 'xyz'], dtype=float)
                self.ncells    = np.rint((self.xmax - self.xmin)/self.coarse_dx).astype(int)
            else:
                self.ncells    = np.array([parfile.get(f'CoordBase::ncells_{x}', default=1) for x in 'xyz'], dtype=int)
                self.coarse_dx = (self.xmax - self.xmin)/self.ncells
        elif self.domainsize == 'spacing':
            xmin               = np.array([parfile.get(f'CoordBase::{x}min', default=0.0) for x in 'xyz'], dtype=float)
            self.coarse_dx     = np.array([parfile.get(f'CoordBase::d{x}', default=1.0) for x in 'xyz'], dtype=float)
            self.ncells        = np.array([parfile.get(f'CoordBase::ncells_{x}', default=1) for x in 'xyz'], dtype=int)
            self.zero_origin_x = np.array([parfile.get(f'CoordBase::zero_origin_{x}', default=False) for x in 'xyz'], dtype=bool)

            self.xmin = -0.5*self.coarse_dx*self.ncells
            self.xmax =  0.5*self.coarse_dx*self.ncells

            self.xmin[self.zero_origin_x] = xmin[self.zero_origin_x]
            self.xmax[self.zero_origin_x] = xmin[self.zero_origin_x] + self.coarse_dx[self.zero_origin_x]*self.ncells[self.zero_origin_x]
        else:
            raise ValueError(self.domainsize)

        # Account for the boundary
        self.boundary_size_lower = np.array([parfile.get(f'CoordBase::boundary_size_{x}_lower', default=1) for x in 'xyz'], dtype=int)
        self.boundary_size_upper = np.array([parfile.get(f'CoordBase::boundary_size_{x}_upper', default=1) for x in 'xyz'], dtype=int)

        self.boundary_shiftout_lower = np.array([parfile.get(f'CoordBase::boundary_shiftout_{x}_lower', default=0) for x in 'xyz'], dtype=int)
        self.boundary_shiftout_upper = np.array([parfile.get(f'CoordBase::boundary_shiftout_{x}_upper', default=0) for x in 'xyz'], dtype=int)

        self.boundary_internal_lower = np.array([parfile.get(f'CoordBase::boundary_internal_{x}_lower', default=False) for x in 'xyz'], dtype=bool)
        self.boundary_internal_upper = np.array([parfile.get(f'CoordBase::boundary_internal_{x}_upper', default=False) for x in 'xyz'], dtype=bool)
        
        self.boundary_staggered_lower = np.array([parfile.get(f'CoordBase::boundary_staggered_{x}_lower', default=False) for x in 'xyz'], dtype=bool)
        self.boundary_staggered_upper = np.array([parfile.get(f'CoordBase::boundary_staggered_{x}_upper', default=False) for x in 'xyz'], dtype=bool)

        external_lower = np.logical_not(self.boundary_internal_lower)
        external_upper = np.logical_not(self.boundary_internal_upper)

        self.xmin -= self.coarse_dx*self.boundary_shiftout_lower
        self.xmax += self.coarse_dx*self.boundary_shiftout_upper

        self.xmin[self.boundary_staggered_lower] -= 0.5*self.coarse_dx[self.boundary_staggered_lower]
        self.xmax[self.boundary_staggered_upper] += 0.5*self.coarse_dx[self.boundary_staggered_upper]

        self.xmin[external_lower] -= (self.boundary_size_lower[external_lower] - 1)*self.coarse_dx[external_lower]
        self.xmax[external_upper] += (self.boundary_size_upper[external_upper] - 1)*self.coarse_dx[external_upper]

        self.ncells = np.rint((self.xmax - self.xmin)/self.coarse_dx).astype(int)
