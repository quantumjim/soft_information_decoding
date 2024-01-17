from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
            out = subprocess.check_output(['bazel', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake and Bazel must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions)
            )

        num_cores = os.cpu_count()

        # Build PyMatching with Bazel
        self.build_pymatching_with_bazel(num_cores)

        for ext in self.extensions:
            self.build_extension(ext, num_cores)

    def build_pymatching_with_bazel(self, num_cores):
        pymatching_dir = os.path.abspath('libs/PyMatching')
        subprocess.check_call(['bazel', 'build', '--jobs', str(num_cores), '//:pymatching', '//:libpymatching', '@stim//:stim_lib'], cwd=pymatching_dir)

    def build_extension(self, ext, num_cores):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
        '-DPYTHON_EXECUTABLE=' + sys.executable,
        '-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang',  # Path to C compiler
        '-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++',
        '-DCMAKE_CXX_FLAGS=-fopenmp'  # Path to C++ compiler
    ] # for omp compiling


        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--verbose', '--', '-j' + str(num_cores)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='soft_information',
    version='0.1.0',
    author='Maurice D. Hanisch',
    description='A Python package for C++ integration with additional Python code for soft information decoding.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[CMakeExtension('cpp_soft_info')],
    cmdclass=dict(build_ext=CMakeBuild),
    python_requires=">=3.11",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    zip_safe=False,
)
