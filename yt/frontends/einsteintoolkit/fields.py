"""
EinsteinToolkit-specific fields



"""

from yt.fields.field_info_container import FieldInfoContainer
from yt.fields.magnetic_field import setup_magnetic_field_aliases

rho_units   = 'code_mass/code_length**3'
press_units = 'code_mass/(code_length*code_time**2)'
mag_units   = 'code_magnetic'
avec_units  = 'code_magnetic*code_length'

class EinsteinToolkitFieldInfo(FieldInfoContainer):
    known_other_fields = (
        # ADMBase
        ('ADMBASE::alp', ('', ['alp', 'alpha', 'lapse'], r'$\alpha$')),
        ('ADMBASE::dtalp', ('', ['dtalp', 'dtalpha', 'dtlapse'], r'$\partial_t \alpha$')),
        ('ADMBASE::betax', ('', ['betax', 'shift_x', 'beta_x'], r'$\beta^x$')),
        ('ADMBASE::betay', ('', ['betay', 'shift_y', 'beta_y'], r'$\beta^y$')),
        ('ADMBASE::betaz', ('', ['betaz', 'shift_z', 'beta_z'], r'$\beta^z$')),
        ('ADMBASE::dtbetax', ('', ['dtbetax', 'dtshift_x', 'dtbeta_x'], r'$\beta^x$')),
        ('ADMBASE::dtbetay', ('', ['dtbetay', 'dtshift_y', 'dtbeta_y'], r'$\beta^y$')),
        ('ADMBASE::dtbetaz', ('', ['dtbetaz', 'dtshift_z', 'dtbeta_z'], r'$\beta^z$')),
        ('ADMBASE::gxx', ('', ['gxx'], r'$\gamma_{xx}$')),
        ('ADMBASE::gyy', ('', ['gyy'], r'$\gamma_{yy}$')),
        ('ADMBASE::gzz', ('', ['gzz'], r'$\gamma_{zz}$')),
        ('ADMBASE::gxy', ('', ['gxy', 'gyx'], r'$\gamma_{xy}$')),
        ('ADMBASE::gxz', ('', ['gxz', 'gzx'], r'$\gamma_{xz}$')),
        ('ADMBASE::gyz', ('', ['gyz', 'gzy'], r'$\gamma_{yz}$')),
        ('ADMBASE::kxx', ('', ['kxx'], r'$K_{xx}$')),
        ('ADMBASE::kyy', ('', ['kyy'], r'$K_{yy}$')),
        ('ADMBASE::kzz', ('', ['kzz'], r'$K_{zz}$')),
        ('ADMBASE::kxy', ('', ['kxy'], r'$K_{xy}$')),
        ('ADMBASE::kxz', ('', ['kxz'], r'$K_{xz}$')),
        ('ADMBASE::kyz', ('', ['kyz'], r'$K_{yz}$')),
        # HydroBase
        ('HYDROBASE::rho', (rho_units, ['rho', 'density'], r'$\rho$')),
        ('HYDROBASE::press', (press_units, ['press', 'pressure'], r'$P$')),
        ('HYDROBASE::eps', ('', ['eps', 'specific_internal_energy'], r'$\epsilon$')),
        ('HYDROBASE::vel[0]', ('c', ['vel[0]', 'velx', 'velocity_x'], r'$v^x$')),
        ('HYDROBASE::vel[1]', ('c', ['vel[1]', 'vely', 'velocity_y'], r'$v^y$')),
        ('HYDROBASE::vel[2]', ('c', ['vel[2]', 'velz', 'velocity_z'], r'$v^z$')),
        ('HYDROBASE::w_lorentz', ('', ['w_lorentz', 'W', 'lorentz_factor'], r'$W$')),
        ('HYDROBASE::Bvec[0]', (mag_units, [], r'$B^x$')),
        ('HYDROBASE::Bvec[1]', (mag_units, [], r'$B^y$')),
        ('HYDROBASE::Bvec[2]', (mag_units, [], r'$B^z$')),
        ('HYDROBASE::Avec[0]', (avec_units, ['Ax'], r'$A^x$')),
        ('HYDROBASE::Avec[1]', (avec_units, ['Ay'], r'$A^y$')),
        ('HYDROBASE::Avec[2]', (avec_units, ['Az'], r'$A^z$')),
        ('HYDROBASE::Aphi', (avec_units, ['Aphi', 'scalar_potential'], r'$\Phi$')),
        ('WEYLSCAL4::Psi4r', ('', ['psi4_real'], r'$\Re \Psi_4$'))
    )

    known_particle_fields = ()

    def setup_fluid_fields(self):
        setup_magnetic_field_aliases(self, 'EinsteinToolkit', [f'HYDROBASE::Bvec[{i}]' for i in range(3)])
