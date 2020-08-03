"""
Interface for CarpetIOHDF5 files for yt.frontends.einsteintoolkit



"""

import os
import glob
import enum
import re
import numpy as np

from collections import defaultdict

from yt.utilities.on_demand_imports import _h5py as h5py

from .util import lazy_property

class CarpetIOHDF5Handler:
    PGA_GROUP = 'Parameters and Global Attributes'

    def __init__(self, base_path, file_pattern, slice_parameter):
        if os.path.isdir(base_path) and file_pattern is not None:
            self.files = [CarpetIOHDF5File(p) for p in glob.glob(os.path.join(base_path, file_pattern))]
        else:
            self.files = [CarpetIOHDF5File(base_path)]
        
        self.dimensionality = self.files[0].dimensionality
        self.slice_plane = SlicePlane.determine_slice_plane(slice_parameter, self.files[0])

        self.field_map = dict()
        for file in self:
            self.field_map.update({field: file for field in file.fields})
        
        self._active_file = None
    
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
    def refinement_factor(self):
        return self.files[0].get_refinement_factor()
    
    @lazy_property
    def index(self):
        return self.build_index()
    
    def build_index(self):
        index_file = self.active_file
        index = DatasetIndex(index_file, index_file.fields[0])
        with index_file as fh:
            for dset in fh:
                if not dset == self.PGA_GROUP:
                    index.add_dataset(dset)
        return index
    
    @property
    def iterations(self):
        return self.index.iterations
    
    def get_datasets(self, iteration, level=None):
        if level is None:
            return sum(self.index[iteration].values(), start=list())
        else:
            return self.index[iteration][level]
    
    def get_levels(self, iteration):
        return sorted(list(self.index[iteration].keys()))

    @staticmethod
    def is_valid(path, **kwargs):
        try:
            CarpetIOHDF5Handler(path, kwargs.get('file_pattern', None), kwargs.get('slice_plane', None))
        except:
            return False

        return True
    
    def __iter__(self):
        return iter(self.files)

class CarpetIOHDF5File:
    def __init__(self, path):
        self.path     = path
        self.filename = os.path.basename(path)
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
    
    def get_refinement_factor(self, default=2):
        try:
            return int(re.search('Carpet::refinement_factor[\s*]=[\s*](\d+)', 
                                 self.all_parameters, re.IGNORECASE).groups()[0])
        except:
            return default

    def open(self):
        assert not self.is_open
        self.fhandle = h5py.File(self.path, 'r')

    def close(self):
        assert self.is_open
        self.fhandle.close()
        self.fhandle = None

    def ensure_open(self):
        if not self.is_open:
            self.open()

    @property
    def is_open(self):
        return self.fhandle is not None

    def __enter__(self):
        if not self.is_open:
            self.open()
            self._close_on_exit = True
        return self.fhandle

    def __exit__(self, extype, exvalue, traceback):
        if self.is_open and self._close_on_exit:
            self.close()
            self._close_on_exit = False

class DatasetIndex:
    def __init__(self, h5file, field):
        self.h5file = h5file
        self.field  = field

        # Nested dictionary using [iteration][refinement level]
        self.datasets = defaultdict(lambda: defaultdict(list))

    def add_dataset(self, dataset):
        field, remainder = dataset.split(' ', 1)

        if not field == self.field:
            return

        iteration, level = [int(elem.split('=',1)[-1]) for elem in remainder.split()[::2]]
        self.datasets[iteration][level].append(dataset)

    def __getitem__(self, iteration):
        return self.datasets[iteration]

    @lazy_property
    def iterations(self):
        return sorted(list(self.datasets.keys()))

class SlicePlane(enum.Enum):
    XY = enum.auto()
    XZ = enum.auto()
    YZ = enum.auto()

    def fill(self, arr, fill_value):
        if self is SlicePlane.XY:
            return np.append(arr, fill_value)
        elif self is SlicePlane.XZ:
            return np.insert(arr, 1, fill_value)
        elif self is SlicePlane.YZ:
            return np.insert(arr, 0, fill_value)

    def reshape(self, arr):
        if self is SlicePlane.XY:
            return np.expand_dims(arr, 2)
        elif self is SlicePlane.XZ:
            return np.expand_dims(arr, 1)
        elif self is SlicePlane.YZ:
            return np.expand_dims(arr, 0)

    @property
    def normal_index(self):
        return [SlicePlane.YZ, SlicePlane.XZ, SlicePlane.XY].index(self)

    @staticmethod
    def determine_slice_plane(param, h5file):
        if param is not None:
            return SlicePlane.from_string(param)
        else:
            return SlicePlane.from_filename(h5file.filename)

    @staticmethod
    def from_string(sp):
        if sp is not None:
            sp = sp.upper()
            if sp in SlicePlane.__members__:
                return SlicePlane.__members__[sp]
        return None

    @staticmethod
    def from_filename(sp):
        try:
            return SlicePlane.from_string(sp.split('.')[-2])
        except:
            return None