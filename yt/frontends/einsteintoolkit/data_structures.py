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
from .carpetiohdf5 import CarpetIOHDF5File, SlicePlane
from .util import determine_refinement_factor

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

        if self.index.ds.slice_plane is not None and self.Parent is not None:
            normal_index = self.index.ds.slice_plane.normal_index
            self.dds[normal_index] = self.Parent.dds[normal_index]

class EinsteinToolkitHierarchy(GridIndex):
    def __init__(self, ds, dataset_type='EinsteinToolkit'):
        self.dataset_type   = dataset_type
        self.dataset        = weakref.proxy(ds)
        self.index_filename = self.dataset.parameter_filename
        self.directory      = os.path.dirname(self.index_filename)
        self.float_type     = np.float64
        self.int_type       = np.int32
        super(EinsteinToolkitHierarchy, self).__init__(ds, dataset_type)

    def _detect_output_fields(self):
        self.field_list = [(self.dataset_type, fname) for fname in self.ds.field_map]

    def _count_grids(self):
        self.num_grids = self.ds.carpet_grid.num_patches

    def _parse_index(self):
        self.grid_left_edge      = np.empty((self.num_grids,3), dtype=self.float_type)
        self.grid_right_edge     = np.empty((self.num_grids,3), dtype=self.float_type)
        self.grid_dimensions     = np.empty((self.num_grids,3), dtype=self.int_type)
        self.grid_particle_count = np.empty((self.num_grids,1), dtype=self.int_type)
        self.grid_levels         = np.empty((self.num_grids,1), dtype=self.int_type)
        self.grids               = np.empty(self.num_grids    , dtype=object)

        for ind, grid_patch in enumerate(self.ds.carpet_grid.all_patches):
            self._set_grid_arrays(ind, grid_patch)
        
        self.max_level = np.amax(self.grid_levels)

        self.grid_left_edge  = self.ds.arr(self.grid_left_edge , self.ds.length_unit)
        self.grid_right_edge = self.ds.arr(self.grid_right_edge, self.ds.length_unit)

    def _set_grid_arrays(self, ind, grid_patch):
        self.grid_left_edge     [ind,:] = grid_patch.left_edge
        self.grid_right_edge    [ind,:] = grid_patch.right_edge
        self.grid_dimensions    [ind,:] = grid_patch.shape
        self.grid_levels        [ind,0] = grid_patch.reflevel
        self.grid_particle_count[ind,0] = 0
        self.grids              [ind  ] = EinsteinToolkitGrid(ind, self, grid_patch)

    def _populate_grid_objects(self):
        for g in self.grids:
            g._prepare_grid()
            g._setup_dx()
        self.reconstruct_children()
    
    def grid_cells_overlap(self, gid, oid):
        return (  np.minimum(self.grid_right_edge[oid].d, self.grid_right_edge[gid].d) \
                - np.maximum(self.grid_left_edge[oid].d, self.grid_left_edge[gid].d)).prod()/self.grids[oid].dds.d.prod()
    
    def reconstruct_children(self):
        mask = np.empty(self.num_grids, dtype=np.int32)
        for index, grid in enumerate(self.grids):
            get_box_grids_level(self.grid_left_edge[index,:], self.grid_right_edge[index,:],
                                self.grid_levels[index]+1, 
                                self.grid_left_edge, self.grid_right_edge, self.grid_levels, mask)

            for child_id in np.where(mask.astype(bool))[0]:
                if self.grid_cells_overlap(index, child_id) > 1:
                    grid.Children.append(self.grids[child_id])
                    self.grids[child_id].parents.append(grid)
            
            get_box_grids_level(self.grid_left_edge[index,:], self.grid_right_edge[index,:],
                                self.grid_levels[index], 
                                self.grid_left_edge, self.grid_right_edge, self.grid_levels, mask)

            for sibling_id in np.where(mask.astype(bool))[0]:
                if sibling_id > index and self.grid_cells_overlap(index, sibling_id) > 1:
                    grid.OverlappingSiblings.append(self.grids[sibling_id])

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

    def __init__(self, filename, dataset_type='EinsteinToolkit',
                 iteration=None,
                 code_mass_solar=1,
                 slice_plane=None,
                 file_pattern=None,
                 verify_amr_masking=False,
                 storage_filename=None,
                 units_override=None):
        self.fluid_types     += ('EinsteinToolkit',)
        self.code_mass_solar  = code_mass_solar
        self.filename         = filename
        self._h5files         = CarpetIOHDF5File.from_pattern(self.filename, file_pattern)
        self.iteration        = iteration
        self.slice_plane      = SlicePlane.determine_slice_plane(slice_plane, self.h5file)
        self.storage_filename = storage_filename

        if self.iteration is None:
            self.iteration = self.h5file.iterations[0]

        self.field_map = dict()
        for h5f in self._h5files:
            assert all([f not in self.field_map for f in h5f.fields])
            self.field_map.update({ f: h5f for f in h5f.fields})

        super(EinsteinToolkitDataset, self).__init__(filename, dataset_type,
                         units_override=units_override)

        # Verify that the sum of the cell volumes is equal to the total domain volume
        if verify_amr_masking:
            cell_volume   = self.all_data()['cell_volume'].sum().in_units('code_length**3').v
            domain_volume = (self.domain_right_edge - self.domain_left_edge).prod().in_units('code_length**3').v
            if not np.isclose(cell_volume, domain_volume):
                log_handler.error('Invalid AMR masking.')
                raise RuntimeError('Invalid AMR masking.')

    def _set_code_unit_attributes(self):
        setdefaultattr(self, 'length_unit', self.quan(self.code_mass_solar, 'l_geom'))
        setdefaultattr(self, 'mass_unit'  , self.quan(self.code_mass_solar, 'm_geom'))
        setdefaultattr(self, 'time_unit'  , self.quan(self.code_mass_solar, 't_geom'))

        setdefaultattr(self, 'velocity_unit', self.quan(1, 'c'))
        setdefaultattr(self, 'magnetic_unit', self.quan((8.35e15)/self.code_mass_solar, 'T'))

    def _parse_parameter_file(self):
        self.unique_identifier = f'{self.parameter_filename}-{time.ctime()}'

        self.refine_by      = determine_refinement_factor(self.h5file.all_parameters)
        self.dimensionality = self.h5file.dimensionality
        self.periodicity    = 3*(False,)

        self.h5file.open()

        self.carpet_grid = CarpetGrid(self)
        self.domain_left_edge  = self.carpet_grid.domain_left_edge
        self.domain_right_edge = self.carpet_grid.domain_right_edge
        self.domain_dimensions = self.carpet_grid.domain_dimensions
        self.current_time      = self.carpet_grid.time

        self.h5file.close()

        if self.dimensionality == 2:
            self._setup_2d()

        self.cosmological_simulation = 0
        self.current_redshift        = 0
        self.omega_lambda            = 0
        self.omega_matter            = 0
        self.hubble_constant         = 0

    def _setup_2d(self):
        self._index_class = EinsteinToolkitHierarchy2D

        if self.slice_plane is None:
            raise RuntimeError('Could not determine slice plane from input file.')

        self.domain_left_edge  = self.slice_plane.fill(self.domain_left_edge , 0)
        self.domain_right_edge = self.slice_plane.fill(self.domain_right_edge, 1)
        self.domain_dimensions = self.slice_plane.fill(self.domain_dimensions, 1)

    @classmethod
    def _is_valid(self, *args, **kwargs):
        return CarpetIOHDF5File.is_valid(args[0], **kwargs)

    @property
    def iterations(self):
        return self.h5file.iterations

    @property
    def h5file(self):
        return self._h5files[0]
