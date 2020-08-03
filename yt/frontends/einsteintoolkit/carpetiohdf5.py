"""
Interface for CarpetIOHDF5 files for yt.frontends.einsteintoolkit



"""

import os
import glob

from collections import defaultdict

from yt.utilities.on_demand_imports import _h5py as h5py

from .definitions import CactusParameters, SlicePlane

class CarpetIOHDF5Handler:
    PGA_GROUP = 'Parameters and Global Attributes'

    def __init__(self, base_path, file_pattern, slice_parameter):
        self._active_file = None

        if os.path.isdir(base_path) and file_pattern is not None:
            self.files = [CarpetIOHDF5File(p, self) for p in glob.glob(os.path.join(base_path, file_pattern))]
        else:
            self.files = [CarpetIOHDF5File(base_path, self)]
        
        self.dimensionality = self.files[0].dimensionality
        self.parameters     = CactusParameters(self.files[0].all_parameters)
        self.slice_plane    = SlicePlane.determine_slice_plane(slice_parameter, self.files[0])

        self.field_map = dict()
        for file in self:
            self.field_map.update({field: file for field in file.fields})
        
        self._index       = None
        self._iterations  = None
    
    @property
    def active_file(self):
        if self._active_file is None:
            self._active_file = self.files[0]
            self._active_file.open()
        return self._active_file
    
    def close_active_file(self):
        if self._active_file is not None:
            self._active_file.close()
            self._active_file = None
    
    @property
    def index(self):
        if self._index is None:
            index_file  = self.active_file
            index_field = index_file.fields[0]
            self._index = defaultdict(lambda: defaultdict(list))

            with index_file as fh:
                for dset in fh:
                    if not dset == self.PGA_GROUP:
                        field, remainder = dset.split(' ', 1)

                        if not field == index_field:
                            continue

                        iteration, level = [int(elem.split('=', 1)[-1]) for elem in remainder.split()[::2]]
                        self._index[iteration][level].append(dset)

        return self._index

    @property
    def iterations(self):
        if self._iterations is None:
            self._iterations = sorted(list(self.index.keys()))
        return self._iterations
    
    def get_datasets(self, iteration, level=None):
        if level is None:
            return sum(self.index[iteration].values(), start=list())
        else:
            return self.index[iteration][level]
    
    def get_levels(self, iteration):
        return sorted(list(self.index[iteration].keys()))
    
    def __iter__(self):
        return iter(self.files)

    @staticmethod
    def is_valid(path, **kwargs):
        try:
            CarpetIOHDF5Handler(path, kwargs.get('file_pattern', None), kwargs.get('slice_plane', None))
            return True
        except:
            return False

class CarpetIOHDF5File:
    def __init__(self, path, handler):
        self.path     = path
        self.filename = os.path.basename(path)
        self.handler  = handler
        self.fhandle  = None

        self._close_on_exit = True

        if not os.path.isfile(path):
            raise IOError(f'CarpetIOHDF5File: {path} is not a valid file')

        def get_dimensionality(name, dset):
            if CarpetIOHDF5Handler.PGA_GROUP not in name:
                return len(dset.shape)

        try:
            with self as fh:
                self.all_parameters = fh[CarpetIOHDF5Handler.PGA_GROUP]['All Parameters'][()].decode('utf-8')
                self.fields         = fh[CarpetIOHDF5Handler.PGA_GROUP]['Datasets'][()].decode('utf-8').splitlines()
                self.dimensionality = fh.visititems(get_dimensionality)
        except:
            raise IOError(f'CarpetIOHDF5File: {path} is not a valid Einstein Toolkit CarpetIOHDF5 file')

    def read_dataset(self, dataset, output):
        with self as fh:
            ds = h5py.h5d.open(fh.id, dataset.encode('latin-1'))
            ds.read(h5py.h5s.ALL, h5py.h5s.ALL, output)
        return output

    def open(self):
        assert self.fhandle is None
        self.fhandle = h5py.File(self.path, 'r')
        return self.fhandle

    def close(self):
        assert self.fhandle is not None
        self.fhandle.close()
        self.fhandle = None

    def __enter__(self):
        if self.fhandle is None:
            self.open()
            self._close_on_exit = True
        return self.fhandle

    def __exit__(self, extype, exvalue, traceback):
        if self.fhandle is not None and self._close_on_exit:
            self.close()
            self._close_on_exit = False