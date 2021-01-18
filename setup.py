#!/usr/bin/env python
import glob
import os
import tempfile
import sys
from distutils.command.build_ext import build_ext as _build_ext
from distutils.errors import DistutilsOptionError

from setuptools import setup, Extension
from Cython.Build import cythonize


class build_ext(_build_ext):
    user_options = _build_ext.user_options + [
        ('gap-root=', None,
         'path to the GAP installation directory (GAP_ROOT); may also be '
         'given by the GAP_ROOT environment variable which is overridden by '
         'this flag'),
        ('gap-include=', None,
         'path to the GAP header files; by default they are assumed installed '
         'in a standard system header path unless --gap-root/GAP_ROOT is '
         'specified, in which case they will be relative to GAP_ROOT; may '
         'also be overridden with the GAP_INCLUDE environment variable'),
        ('gap-lib=', None,
         'directory containing the libgap binary; by default it is assumed '
         'installed in a standard system library path unless '
         '--gap-root/GAP_ROOT is specified, in which case it will be relative '
         'to GAP_ROOT; may also be overridden with the GAP_LIB environment '
         'variable')
    ]

    def initialize_options(self):
        super().initialize_options()
        self.gap_root = None
        self.gap_include = None
        self.gap_lib = None
        self._using_gap_root = False

    def finalize_options(self):
        super().finalize_options()

        self.gap_root = self._get_directory_option('gap_root')
        self.gap_include = self._get_directory_option('gap_include')
        self.get_lib = self._get_directory_option('gap_lib')

        if self.gap_root is not None:
            self._using_gap_root = True

            if self.gap_include is None:
                self.gap_include = os.path.join(self.gap_root, 'src')

            if self.gap_lib is None:
                self.gap_lib = os.path.join(self.gap_root, '.libs')

        # Automatic support for building in a conda environment with the
        # conda-provided Python; this is more reliable than using CONDA_PREFIX
        # which may not be set if a conda Python is run without activating
        # the environment
        if os.path.exists(os.path.join(sys.prefix, 'conda-meta')):
            self.include_dirs.insert(0, os.path.join(sys.prefix, 'include'))
            self.library_dirs.insert(0, os.path.join(sys.prefix, 'lib'))

    def run(self):
        if self._using_gap_root:
            # We are using the headers from a GAP_ROOT installation of GAP,
            # so the headers are not prefixed by a directory named gap/ as
            # expected by our sources (as would be the case when the headers
            # are installed as system headers) so we must make a symlink to
            # the gap headers named gap/
            #
            # We must also add the GAP_ROOT/gen directory for config.h
            # TODO: In newer GAP versions this appears to be renamed
            # GAP_ROOT/build
            include_temp = os.path.join(self.build_temp, 'include')
            gap_includes = [
                include_temp,
                os.path.join(self.gap_root, 'gen')
            ]
            gap_dir = os.path.join(include_temp, 'gap')
            if not os.path.exists(gap_dir):
                os.makedirs(os.path.join(include_temp))
                os.symlink(self.gap_include, gap_dir)
        elif self.gap_include is not None:
            gap_includes = [self.gap_include]
        else:
            gap_includes = []

        self.include_dirs = gap_includes + self.include_dirs

        if self.gap_lib is not None:
            self.library_dirs.insert(0, self.gap_lib)
            self.rpath.insert(0, self.gap_lib)

        if self.extensions:
            nthreads = getattr(self, 'parallel', None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            self.extensions[:] = cythonize(
                self.extensions, nthreads=nthreads, force=self.force,
                language_level=3, compiler_directives={'autotestdict': False})
        super().run()

    def _get_directory_option(self, opt):
        val = getattr(self, opt)

        if val is None:
            val = os.environ.get(opt.upper())

        if val is not None and not os.path.isdir(val):
            raise DistutilsOptionError(
                f'--{opt}/{opt.upper()} directory {val} does not exist or is '
                f'not a directory')

        return val


setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension('*', ['gappy/*.pyx'])],
    use_scm_version={'write_to': 'gappy/_version.py'}
)
