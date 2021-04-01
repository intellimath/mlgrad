# coding: utf-8

from setuptools import setup

from Cython.Distutils import Extension, build_ext
from Cython.Compiler import Options
from Cython.Build import cythonize

Options.fast_fail = True

import platform
WIN32 = (platform.system() == 'Windows')

extra_compile_args = ["-O3", "-march=native"] 
# extra_compile_args = ["-O3", "-march=native", "-mavx2", "-mfma", "-msse2",  "-fopt-info-vec-optimized"] 
extra_link_args = [] #["-lm"]
extra_compile_args_openmp = ["-O3", "-march=native", ("-fopenmp" if not WIN32 else "/openmp")]
# extra_compile_args_openmp = ["-O3", "-march=native", "-mavx2", "-mfma", "-msse2", "-fopt-info-vec-optimized", ("-fopenmp" if not WIN32 else "/openmp")]
extra_link_args_openmp = [("-fopenmp" if not WIN32 else "/openmp")] #, "-lm"]

# cython_compile_time_env = {"USE_OPENMP":1}

ext_modules = [
    Extension(
        "mlgrad.func",
        ["lib/mlgrad/func.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.model",
        ["lib/mlgrad/model.pyx"],
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
        ["lib/mlgrad/avragg.pyx"], #"lib/mlgrad/c/_avragg.c"],
        extra_compile_args = extra_compile_args_openmp, 
        extra_link_args = extra_link_args_openmp,
#         cython_compile_time_env = cython_compile_time_env,
    ),
    Extension(
        "mlgrad.kaverage",
        ["lib/mlgrad/kaverage.pyx"],
        extra_compile_args = extra_compile_args, 
        extra_link_args = extra_link_args,
#         cython_compile_time_env = cython_compile_time_env,
    ),
    Extension(
        "mlgrad.loss",
        ["lib/mlgrad/loss.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
   Extension(
        "mlgrad.distance",
        ["lib/mlgrad/distance.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
   Extension(
        "mlgrad.dissimilarity",
        ["lib/mlgrad/dissimilarity.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.regular",
        ["lib/mlgrad/regular.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.normalizer",
        ["lib/mlgrad/normalizer.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.risk",
        ["lib/mlgrad/risk.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
#         cython_compile_time_env = cython_compile_time_env,
    ),
    Extension(
        "mlgrad.averager",
        ["lib/mlgrad/averager.pyx"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
    Extension(
        "mlgrad.weights",
        ["lib/mlgrad/weights.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    # Extension(
    #     "mlgrad.stopcond",
    #     ["lib/mlgrad/stopcond.pyx"],
    # ),
    Extension(
        "mlgrad.gd",
        ["lib/mlgrad/gd.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    # Extension(
    #     "mlgrad.pbsag",
    #     ["lib/mlgrad/pbsag.pyx"],
    # ),
    # Extension(
    #     "mlgrad.irls",
    #     ["lib/mlgrad/irls.pyx"],
    # ),
    Extension(
        "mlgrad.irgd",
        ["lib/mlgrad/irgd.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    Extension(
        "mlgrad.kmeans",
        ["lib/mlgrad/kmeans.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    ),
    #Extension(
    #    "mlgrad.kagg",
    #    ["lib/mlgrad/kagg.pyx"],
    #    extra_compile_args = extra_compile_args,
    #),
    Extension(
        "mlgrad.mlocation_scatter",
        ["lib/mlgrad/mlocation_scatter.pyx"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
    ),
    Extension(
        "mlgrad.mlocation_scatter2",
        ["lib/mlgrad/mlocation_scatter2.pyx"],
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
    version = '0.4',
    description = 'Robust Gradient Methods for Machine Learning & Data Analisys',
    author = 'Zaur Shibzukhov',
    author_email = "szport@gmail.com",
    license = "MIT License",
#     ext_modules = cythonize(ext_modules),
    ext_modules = ext_modules,
    package_dir = {'': 'lib'},
    cmdclass = {'build_ext': build_ext},
    packages = ['mlgrad', 'mlgrad.regr', 'mlgrad.test'],
    package_data = {'': ['*.pxd']},
    url = 'https://bitbucket.org/intellimath/mlgrad',
    download_url = 'https://bitbucket.org/intellimath/mlgrad',
    long_description = "", # long_description,
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)
