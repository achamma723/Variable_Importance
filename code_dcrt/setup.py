# -*- coding: utf-8 -*-
# Author: Binh Nguyen <tuan-binh.nguyen@inria.fr>

import os
import sys

from Cython.Build import cythonize
from setuptools import find_packages

PKG = 'sandbox'
LICENSE = 'BSD'
DESCRIPTION = 'Aggregation of Multiple Knockoffs for Python 3'

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)


def load_version():
    """Executes version.py in a globals dictionary and return it.
    Following format from Nilearn repo on github.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('sandbox', 'version.py')) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')  # for UNIX distros

    include_dirs = [numpy.get_include()]
    config = Configuration(PKG, parent_package, top_path)

    config.add_extension(
        name='sampling_covariance_fast',
        sources=['./sandbox/sampling_covariance_fast.pyx'],
        include_dirs=include_dirs,
        libraries=libraries)

    config.add_extension(
        name='inflate_non_null',
        sources=['./sandbox/inflate_non_null.pyx'],
        include_dirs=include_dirs,
        libraries=libraries)
    
    config.ext_modules = cythonize(
        config.ext_modules)
    
    return config


def setup_package(version):
    from numpy.distutils.core import setup
    
    with open('README.md', encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()

    setup(
        packages=find_packages(exclude=['contrib', 'docs', 'tests']),
        version=version,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        author='Binh T. Nguyen',
        author_email='tuan-binh.nguyen@inria.fr',
        license=LICENSE,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        install_requires=['numpy', 'scikit-learn', 'scipy', 'cython'],
        extras_require={'test': ['coverage']},
        zip_safe=False,  # the package can run out of an .egg file
        **configuration(top_path='').todict(),
    )


_VERSION_GLOBALS = load_version()
VERSION = _VERSION_GLOBALS['__version__']

if __name__ == "__main__":
    setup_package(VERSION)
