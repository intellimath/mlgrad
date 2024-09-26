

from setuptools import Extension
from setuptools.command.build_py import build_py as _build_py

from Cython.Build import cythonize

import platform
WIN32 = (platform.system() == 'Windows')

if WIN32:
    Oflag = []
else:
    Oflag = ["-O3", "-march=native"]

extra_compile_args = Oflag #  ["-fno-wrapv"] 
# extra_link_args = Oflag + ["-lm"]
# extra_compile_args_openmp = extra_compile_args + [("-fopenmp" if not WIN32 else "/openmp")]
# extra_link_args_openmp = [Oflag, "-lm", ("-fopenmp" if not WIN32 else "/openmp")]

# extra_compile_args = [] #, "-fno-wrapv"] 
if WIN32:
    extra_link_args = []
else:
    extra_link_args = ["-lm"]

extra_compile_args_openmp = extra_compile_args + [("-fopenmp" if not WIN32 else "/openmp")]
extra_link_args_openmp = extra_link_args + [("-fopenmp" if not WIN32 else "/openmp")]

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

ext_modules = [
    Extension(
        "mlgrad.inventory",
        ["lib/mlgrad/inventory.pyx"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.array_allocator",
        ["lib/mlgrad/array_allocator.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.array_transform",
        ["lib/mlgrad/array_transform.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.list_values",
        ["lib/mlgrad/list_values.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.funcs",
        ["lib/mlgrad/funcs.pyx"],
        # libraries=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.funcs2",
        ["lib/mlgrad/funcs2.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.models",
        ["lib/mlgrad/models.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.batch",
        ["lib/mlgrad/batch.pyx"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.avragg",
        ["lib/mlgrad/avragg.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args_openmp, 
        extra_link_args = extra_link_args_openmp,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.kaverage",
        ["lib/mlgrad/kaverage.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args, 
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.loss",
        ["lib/mlgrad/loss.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
   Extension(
        "mlgrad.distance",
        ["lib/mlgrad/distance.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
   Extension(
        "mlgrad.dissimilarity",
        ["lib/mlgrad/dissimilarity.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
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
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.averager",
        ["lib/mlgrad/averager.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.weights",
        ["lib/mlgrad/weights.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.gd",
        ["lib/mlgrad/gd.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.kmeans",
        ["lib/mlgrad/kmeans.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.mlocation_scatter2",
        ["lib/mlgrad/mlocation_scatter2.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args_openmp,
        extra_link_args = extra_link_args_openmp,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "mlgrad.smooth._whittaker",
        ["lib/mlgrad/smooth/_whittaker.pyx"],
        # library=["-lm"],
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
 ]


class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

    def initialize_options(self):
        super().initialize_options()
        if self.distribution.ext_modules == None:
            self.distribution.ext_modules = cythonize(
                ext_modules, 
                compiler_directives=cython_compiler_directives1,
            )