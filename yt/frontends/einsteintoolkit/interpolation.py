"""
Class with static methods to provide stencil slices for interpolation.

"""

import numpy as np

class InterpolationHandler:
    left_slice  = dict()
    right_slice = dict()

    @staticmethod
    def interp_left(dim):
        if dim not in InterpolationHandler.left_slice:
            InterpolationHandler.setup_slices(dim)
        return InterpolationHandler.left_slice[dim]
    
    @staticmethod
    def interp_right(dim):
        if dim not in InterpolationHandler.right_slice:
            InterpolationHandler.setup_slices(dim)
        return InterpolationHandler.right_slice[dim]
    
    @staticmethod
    def interp_slices(dim):
        return zip(InterpolationHandler.interp_left(dim), InterpolationHandler.interp_right(dim))
    
    @staticmethod
    def setup_slices(dim):
        InterpolationHandler.left_slice [dim] = [np.index_exp[:-1] + (dim-1)*np.index_exp[:]]
        InterpolationHandler.right_slice[dim] = [np.index_exp[1: ] + (dim-1)*np.index_exp[:]]
        for _ in range(dim-1):
            InterpolationHandler.left_slice [dim].append(tuple(np.roll(InterpolationHandler.left_slice [dim][-1], 1)))
            InterpolationHandler.right_slice[dim].append(tuple(np.roll(InterpolationHandler.right_slice[dim][-1], 1)))