"""
EinsteinToolkit data structures



"""

import os
import weakref
import time
import numpy as np

from yt.data_objects.grid_patch import AMRGridPatch
from yt.data_objects.static_output import Dataset
from yt.geometry.grid_geometry_handler import GridIndex
from yt.units.yt_array import YTArray
from yt.funcs import setdefaultattr
from yt.utilities.lib.misc_utilities import get_box_grids_level
from yt.funcs import mylog as log_handler

from .carpet import CarpetGrid, GridPatch
from .fields import EinsteinToolkitFieldInfo
from .carpetiohdf5 import CarpetIOHDF5Handler

class EinsteinToolkitGrid(AMRGridPatch):
    _id_offset = 0
    def __init__(self, id, index, grid_patch):
        super(EinsteinToolkitGrid, self).__init__(
            id, filename=index.index_filename, index=index)
        self.grid_patch = grid_patch
        self.Level = grid_patch.reflevel
        self.parents = list()
        self.Children = list()
        self.OverlappingSiblings = list()
    
    @property
    def Parent(self):
        return self.parents[0] if len(self.parents) > 0 else None
    
    def __repr__(self):
        return f'{type(self).__name__}_{self.id:04d} ({self.ActiveDimensions}'
    
    def _fill_child_mask(self, child, mask, tofill, dlevel=1):
        # This override is only needed until PR #2802 is merged.
        dlevel = child.Level - self.Level
        return super(EinsteinToolkitGrid, self)._fill_child_mask(child, mask, tofill, dlevel=dlevel)

class EinsteinToolkitGrid2D(EinsteinToolkitGrid):
    def _setup_dx(self):
        """
        AMRGridPatch._setup_dx handles 2D data differently than this frontend.

        """
        self.ds.dimensionality = 3
        super(EinsteinToolkitGrid2D, self)._setup_dx()
        self.ds.dimensionality = 2

class EinsteinToolkitHierarchy(GridIndex):
    def __init__(self, ds, dataset_type='EinsteinToolkit'):
        self.dataset_type   = dataset_type
        self.index_filename = ds.parameter_filename
        super(EinsteinToolkitHierarchy, self).__init__(ds, dataset_type)

    def _detect_output_fields(self):
        self.field_list = [(self.dataset_type, fname) for fname in self.ds.h5handler.field_map]

    def _count_grids(self):
        self.num_grids = self.ds.carpet_grid.num_patches
    
    def _initialize_grid_arrays(self):
        super(EinsteinToolkitHierarchy, self)._initialize_grid_arrays()
        self.grids = np.empty(self.num_grids, dtype=object)

    def _parse_index(self):
        for ind, grid_patch in enumerate(self.ds.carpet_grid.grid_patches):
            self._set_grid_arrays(ind, grid_patch)
        
        self.grid_left_edge  = self.ds.arr(self.grid_left_edge , self.ds.length_unit)
        self.grid_right_edge = self.ds.arr(self.grid_right_edge, self.ds.length_unit)
        self.max_level = np.amax(self.grid_levels)

    def _set_grid_arrays(self, ind, grid_patch):
        self.grid_left_edge     [ind,:] = grid_patch.left_edge
        self.grid_right_edge    [ind,:] = grid_patch.right_edge
        self.grid_dimensions    [ind,:] = grid_patch.shape
        self.grid_levels        [ind,0] = grid_patch.reflevel
        self.grid_particle_count[ind,0] = 0
        self.grids              [ind  ] = EinsteinToolkitGrid(ind, self, grid_patch)

    def _populate_grid_objects(self):
        # First pass: setup each individual grid
        for grid in self.grids:
            grid._prepare_grid()
            grid._setup_dx()

        # Second pass: reconstruct parent/child/sibling relationships
        for index, grid in enumerate(self.grids):
            for child_id in self._get_box_grids(index, grid.Level+1):
                grid.Children.append(self.grids[child_id])
                self.grids[child_id].parents.append(grid)
            for sibling_id in self._get_box_grids(index, grid.Level, min_index=index+1):
                grid.OverlappingSiblings.append(self.grids[sibling_id])

    def _grid_cells_overlap(self, gid, oid):
        return (  np.minimum(self.grid_right_edge[oid].d, self.grid_right_edge[gid].d) \
                - np.maximum(self.grid_left_edge[oid].d, self.grid_left_edge[gid].d)).prod()/self.grids[oid].dds.d.prod()
    
    def _get_box_grids(self, index, level, min_index=0):
        mask = np.empty(self.num_grids, dtype=np.int32)
        get_box_grids_level(self.grid_left_edge[index,:], self.grid_right_edge[index,:], level, 
                            self.grid_left_edge, self.grid_right_edge, self.grid_levels, mask, 
                            min_index=min_index)
        return [gid for gid in np.where(mask)[0] if self._grid_cells_overlap(index, gid) > 1]

class EinsteinToolkitHierarchy2D(EinsteinToolkitHierarchy):
    def _set_grid_arrays(self, ind, grid_patch):
        self.grid_left_edge     [ind,:] = self.ds.slice_plane.fill(grid_patch.left_edge , 0)
        self.grid_right_edge    [ind,:] = self.ds.slice_plane.fill(grid_patch.right_edge, 1)
        self.grid_dimensions    [ind,:] = self.ds.slice_plane.fill(grid_patch.shape     , 1)
        self.grid_levels        [ind,0] = grid_patch.reflevel
        self.grid_particle_count[ind,0] = 0
        self.grids              [  ind] = EinsteinToolkitGrid2D(ind, self, grid_patch)

class EinsteinToolkitDataset(Dataset):
    _index_class = EinsteinToolkitHierarchy
    _field_info_class = EinsteinToolkitFieldInfo

    @classmethod
    def _is_valid(self, *args, **kwargs):
        return CarpetIOHDF5Handler.is_valid(args[0], **kwargs)

    def __init__(self, filename, file_pattern=None, slice_plane=None,
                 iteration=None, code_mass_solar=1):
        self.fluid_types += ('EinsteinToolkit',)
        self.filename = filename
        self.code_mass_solar = code_mass_solar

        self.h5handler   = CarpetIOHDF5Handler(self.filename, file_pattern, slice_plane)
        self.slice_plane = self.h5handler.slice_plane
        self.iteration   = self.iterations[0] if iteration is None else iteration

        super(EinsteinToolkitDataset, self).__init__(filename, dataset_type='EinsteinToolkit')

    @property
    def iterations(self):
        return self.h5handler.iterations

    def _set_code_unit_attributes(self):
        setdefaultattr(self, 'length_unit', self.quan(self.code_mass_solar, 'l_geom'))
        setdefaultattr(self, 'mass_unit'  , self.quan(self.code_mass_solar, 'm_geom'))
        setdefaultattr(self, 'time_unit'  , self.quan(self.code_mass_solar, 't_geom'))

        setdefaultattr(self, 'velocity_unit', self.quan(1, 'c'))
        setdefaultattr(self, 'magnetic_unit', self.quan((8.35e15)/self.code_mass_solar, 'T'))
    
    def _parse_parameter_file(self):
        self.cosmological_simulation = False
        self.unique_identifier = f'{self.parameter_filename}-{time.ctime()}'
        self.refine_by = self.h5handler.parameters.get('carpet::refinement_factor', default=2)
        self.dimensionality = self.h5handler.dimensionality
        self.periodicity = 3*(False,)

        self.carpet_grid = CarpetGrid(self.h5handler, self.iteration)
        self.domain_left_edge  = self.carpet_grid.left_edge
        self.domain_right_edge = self.carpet_grid.right_edge
        self.domain_dimensions = self.carpet_grid.dimensions
        self.current_time      = self.carpet_grid.time

        if self.dimensionality == 2:
            self._setup_2d()

    def _setup_2d(self):
        self._index_class = EinsteinToolkitHierarchy2D

        if self.slice_plane is None:
            log_handler.error('Could not determine slice plane from input file.')
            raise RuntimeError('Could not determine slice plane from input file.')

        self.domain_left_edge  = self.slice_plane.fill(self.domain_left_edge , 0)
        self.domain_right_edge = self.slice_plane.fill(self.domain_right_edge, 1)
        self.domain_dimensions = self.slice_plane.fill(self.domain_dimensions, 1)