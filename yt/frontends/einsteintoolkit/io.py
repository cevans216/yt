"""
EinsteinToolkit-specific IO functions



"""

import numpy as np

from yt.utilities.io_handler import BaseIOHandler
from yt.geometry.selection_routines import GridSelector

class EinsteinToolkitIOHandler(BaseIOHandler):
    _particle_reader = False
    _dataset_type    = 'EinsteinToolkit'

    def _read_fluid_selection(self, chunks, selector, fields, size):
        data  = { field: np.empty(size, dtype=self.ds.index.float_type) for field in fields }
        files = [self.ds.field_map[f[-1]] for f in fields if f[0] == self._dataset_type]
        list(map(lambda f:f.ensure_open(), files))

        offset = 0
        for chunk in chunks:
            chunk_data = self._read_chunk_data(chunk, fields)
            for grid in chunk.objs:
                for field in fields:
                    fdata = chunk_data[grid.id].pop(field)
                    if isinstance(selector, GridSelector):
                        length = fdata.size
                        data[field][offset:offset+length] = np.reshape(fdata, length)
                    else:
                        length = grid.select(selector, fdata, data[field], offset)
                offset += length

        list(map(lambda f:f.close(), files))
        return data

    def _read_chunk_data(self, chunk, fields):
        chunk_data = dict()
        for grid in chunk.objs:
            chunk_data[grid.id] = { field: grid.grid_patch.read_field(field) for field in fields }
        return chunk_data

class HDF5GridPatch:
    def __init__(self, h5file, dataset):
        self.dataset = dataset
        self.suffix  = dataset.split(' ', 1)[-1]

        with h5file as fh:
            dataset = fh[dataset]
            attrs   = dataset.attrs

            self.dtype   = dataset.dtype
            self.origin  = attrs['origin']
            self.iorigin = attrs['iorigin']
            self.delta   = attrs['delta']
            self.level   = attrs['level']
            self.time    = attrs['time']
            self.shape   = np.flip(np.array(dataset.shape).astype(int))

            self.nghostzones = attrs['cctk_nghostzones']

    def __repr__(self):
        return f'HDF5GridPatch: {self.suffix}'

    def read_field(self, field_name, h5file):
        data = np.empty(self.shape[::-1], dtype=self.dtype)
        h5file.read_dataset(f'{field_name} {self.suffix}', data)
        return data.T