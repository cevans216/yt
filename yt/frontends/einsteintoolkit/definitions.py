import enum
import numpy as np

class SlicePlane(enum.Enum):
    XY = enum.auto()
    XZ = enum.auto()
    YZ = enum.auto()

    @staticmethod
    def determine_slice_plane(slice_parameter, h5file):
        if slice_parameter is None:
            slice_parameter = h5file.filename.split('.')[-2]
        return SlicePlane.__members__[slice_parameter.upper()]

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

class CactusParameters:
    def __init__(self, all_parameters):
        self.parameters = dict()

        for line in all_parameters.splitlines():
            param, valstr = (part.strip() for part in line.split("=", 1))
            self.parameters[param.lower()] = cast_parameter(valstr.replace('"', ''))

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

def cast_parameter(vstr):
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