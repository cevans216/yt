"""
Data structure encapsulating a Carpet grid hierarchy

"""

import weakref
import numpy as np

from collections import defaultdict

from .io import HDF5GridPatch
from .interpolation import InterpolationHandler

class CarpetGrid:
    def __init__(self, h5handler, iteration):
        # Get GridPatch objects for each dataset and separate by refinement level
        self.grid_patches = defaultdict(list)
        for patch in map(lambda dset: GridPatch(h5handler, dset), h5handler.get_datasets(iteration)):
            self.grid_patches[patch.reflevel].append(patch)

        # Preprocess grid patches to remove redundancy. Typically only relevant for slice data.
        for reflevel in self.grid_patches:
            # Remove any duplicate patches (cover the same grid points).
            self.grid_patches[reflevel] = list(set(self.grid_patches[reflevel]))

            # Remove any grid patches that are completely contained within another
            filter_func = lambda p: not any([p is not o and o.contains(p) for o in self.grid_patches[reflevel]])
            self.grid_patches[reflevel] = list(filter(filter_func, self.grid_patches[reflevel]))

        # Determine domain geometry
        self.coarse_delta = self.grid_patches[0][0].delta
        self.left_edge    = np.amin(np.stack([patch.left_edge  for patch in self.grid_patches[0]], axis=0), axis=0)
        self.right_edge   = np.amax(np.stack([patch.right_edge for patch in self.grid_patches[0]], axis=0), axis=0)
        self.dimensions   = np.rint((self.right_edge - self.left_edge)/self.coarse_delta).astype(int)

        self.time = self.grid_patches[0][0].hdf5_patch.time

        # Flatten
        self.grid_patches = sum([patches for patches in self.grid_patches.values()], start=list())
        self.num_patches  = len(self.grid_patches)

        h5handler.close_active_file()

class GridPatch:
    def __init__(self, h5handler, dataset_name):
        self.h5handler  = h5handler
        self.hdf5_patch = HDF5GridPatch(h5handler.active_file, dataset_name)

        self.reflevel  = self.hdf5_patch.level
        self.component = int(self.hdf5_patch.suffix.split('=')[-1])
        self.id        = (self.reflevel, self.component)

        self.delta = self.hdf5_patch.delta
        self.dim   = len(self.hdf5_patch.shape)

        # Process ghost zones 
        self.ngz_lower = self.hdf5_patch.nghostzones.copy()
        self.ngz_upper = self.hdf5_patch.nghostzones.copy()

        self.iorigin = self.hdf5_patch.iorigin + self.ngz_lower
        self.vshape  = self.hdf5_patch.shape - (self.ngz_lower + self.ngz_upper)

        # Ensure that the edge of this patch falls on a cell edge of parent grids
        for axi in range(self.dim):
            if self.iorigin[axi] % 2 > 0:
                assert self.ngz_lower[axi] > 0
                self.ngz_lower[axi] -= 1
                self.iorigin  [axi] -= 1
                self.vshape   [axi] += 1
            if self.vshape[axi] % 2 == 0:
                assert self.ngz_upper[axi] > 0
                self.ngz_upper[axi] -= 1
                self.vshape   [axi] += 1
        
        self.upper_index = self.iorigin + self.vshape - 1

        self.left_edge  = self.hdf5_patch.origin + self.ngz_lower*self.delta
        self.right_edge = self.left_edge + (self.vshape - 1)*self.delta
        self.read_slice = tuple([slice(ng, ng+s) for ng,s in zip(self.ngz_lower, self.vshape)])
        self.shape      = self.vshape - 1

        self.volume = np.prod(self.right_edge - self.left_edge)

        # Hash key for rapid comparison
        self.hash_key = (self.dim, self.reflevel) + tuple(self.shape) + tuple(self.iorigin)
    
    def read_field(self, field_name):
        if isinstance(field_name, tuple):
            field_name = field_name[-1]
        
        # Read raw vertex-centered data from disk
        vdata = self.hdf5_patch.read_field(field_name, self.h5handler.field_map[field_name])

        # Remove ghost zones and do linear interpolation to dual grid (cell centered)
        data = vdata[self.read_slice]
        for sll, slr in InterpolationHandler.interp_slices(self.dim):
            data = 0.5*(data[sll] + data[slr])
        
        # If this is 2D data, reshape the result accordingly
        if self.dim == 2:
            data = self.h5handler.slice_plane.reshape(data)
        
        return data

    def __hash__(self):
        return hash(self.hash_key)

    # Two patches are equal if they cover the same grid points
    def __eq__(self, other):
        return (self.hash_key == other.hash_key)

    def contains(self, other):
        return not (np.any(other.iorigin < self.iorigin) or \
                    np.any(other.iorigin > self.upper_index) or \
                    np.any(other.upper_index < self.iorigin) or \
                    np.any(other.upper_index > self.upper_index))