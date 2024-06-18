# coding: utf-8

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

from Cython.Compiler import Options
from Cython.Build import cythonize

Options.fast_fail = True

import platform
WIN32 = (platform.system() == 'Windows')

Oflag = ["-O3", "-march=native"]
extra_compile_args = Oflag #, "-fno-wrapv"] 
# extra_link_args = Oflag + ["-lm"]
# extra_compile_args_openmp = extra_compile_args + [("-fopenmp" if not WIN32 else "/openmp")]
# extra_link_args_openmp = [Oflag, "-lm", ("-fopenmp" if not WIN32 else "/openmp")]

# extra_compile_args = [] #, "-fno-wrapv"] 
extra_link_args = ["-lm"]
extra_compile_args_openmp = extra_compile_args + [("-fopenmp" if not WIN32 else "/openmp")]
extra_link_args_openmp = extra_link_args + [("-fopenmp" if not WIN32 else "/openmp")]

# cython_compile_time_env = {"USE_OPENMP":1}
cython_compiler_directives1 = dict(
    language_level='3',
    boundscheck=True,
    wraparound=True,
    nonecheck=True,
    embedsignature=True,
    initializedcheck=True,
    unraisable_tracebacks=True,  
)
cython_compiler_directives2 = dict(
    language_level='3',
    boundscheck=False,
    wraparound=False,
    nonecheck=False,
    embedsignature=True,
    initializedcheck=False,
    unraisable_tracebacks=True,  
)

Options._directive_defaults.update(cython_compiler_directives2)

ext_modules = [
    Extension(
        "mlgrad.inventory",
        ["lib/mlgrad/inventory.pyx"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
    Extension(
        "mlgrad.array_allocator",
        ["lib/mlgrad/array_allocator.pyx"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
    Extension(
        "mlgrad.list_values",
        ["lib/mlgrad/list_values.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.funcs",
        ["lib/mlgrad/funcs.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
    Extension(
        "mlgrad.funcs2",
        ["lib/mlgrad/funcs2.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.models",
        ["lib/mlgrad/models.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
    Extension(
        "mlgrad.miscfuncs",
        ["lib/mlgrad/miscfuncs.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.batch",
        ["lib/mlgrad/batch.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.avragg",
        ["lib/mlgrad/avragg.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args_openmp, 
        extra_link_args = extra_link_args_openmp,
#         cython_compile_time_env = cython_compile_time_env,
    ),
    Extension(
        "mlgrad.kaverage",
        ["lib/mlgrad/kaverage.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args, 
        extra_link_args = extra_link_args,
#         cython_compile_time_env = cython_compile_time_env,
    ),
    Extension(
        "mlgrad.loss",
        ["lib/mlgrad/loss.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
   Extension(
        "mlgrad.distance",
        ["lib/mlgrad/distance.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
   Extension(
        "mlgrad.dissimilarity",
        ["lib/mlgrad/dissimilarity.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    # Extension(
    #     "mlgrad.normalizer",
    #     ["lib/mlgrad/normalizer.pyx"],
    #     extra_compile_args = extra_compile_args,
    #     extra_link_args = extra_link_args,
    # ),
    Extension(
        "mlgrad.risks",
        ["lib/mlgrad/risks.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
#         cython_compile_time_env = cython_compile_time_env,
    ),
    Extension(
        "mlgrad.averager",
        ["lib/mlgrad/averager.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
    Extension(
        "mlgrad.weights",
        ["lib/mlgrad/weights.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.gd",
        ["lib/mlgrad/gd.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    # Extension(
    #     "mlgrad.irgd",
    #     ["lib/mlgrad/irgd.pyx"],
    #     extra_compile_args = extra_compile_args,
    #     extra_link_args = extra_link_args,
    # ),
    Extension(
        "mlgrad.kmeans",
        ["lib/mlgrad/kmeans.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.mlocation_scatter2",
        ["lib/mlgrad/mlocation_scatter2.pyx"],
        library=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
#     Extension(
#         "mlgrad.cytest",
#         ["lib/mlgrad/cytest.pyx"],
#         extra_compile_args = extra_compile_args,
#         extra_link_args = extra_link_args,
# #         cython_compile_time_env = cython_compile_time_env,
#     ),
 ]

#long_description = open('README.rst').read()

setup(
    name = 'mlgrad',
    version = '0.6',
    description = 'Robust Gradient Methods for Machine Learning & Data Analysis',
    author = 'Zaur Shibzukhov',
    author_email = "szport@gmail.com",
    license = "MIT License",
    ext_modules = cythonize(ext_modules, nthreads=4,
                            compiler_directives=cython_compiler_directives2),
    # ext_modules = ext_modules,
    package_dir = {'': 'lib'},
    cmdclass = {'build_ext': build_ext},
    packages = ['mlgrad', 'mlgrad.af', 'mlgrad.regr', 'mlgrad.boost', 'mlgrad.plots',
                'mlgrad.cls', 'mlgrad.pca', 'mlgrad.cluster', 'mlgrad.outl', 'mlgrad.smooth',
                'mlgrad.test'],
    package_data = {'': ['*.pxd']},
    url = 'https://github.org/intellimath/mlgrad',
    download_url = 'https://github.org/intellimath/mlgrad',
    long_description = "", # long_description,
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
