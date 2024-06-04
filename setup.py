from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import os
import subprocess
import sys
import sysconfig

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
            subprocess.check_output(['bazel', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake and Bazel must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions)
            )

        if os.environ.get('BAZEL_BUILD') == '1':
            self.build_pymatching_with_bazel()
        else:
            print("Skipping Bazel build due to SKIP_BAZEL_BUILD flag.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_pymatching_with_bazel(self):
        num_cores = os.cpu_count()
        pymatching_dir = os.path.abspath('libs/PyMatching')
        env = os.environ.copy()
        env['BAZEL_SH'] = r'C:\Program Files\Git\bin\bash.exe'
        env['CC'] = 'clang-cl'
        env['CXX'] = 'clang-cl'
        subprocess.check_call(['bazel', 'build', '--jobs', str(num_cores), '--cxxopt=/std:c++20', '--crosstool_top=@bazel_tools//tools/cpp:default-clang-cl', '//:pymatching', '//:libpymatching', '@stim//:stim_dev_wheel'], cwd=pymatching_dir, env=env)
        # subprocess.check_call(['bazel', 'build', '--jobs', str(num_cores), '--cxxopt=/std:c++20', '--crosstool_top=@bazel_tools//tools/cpp:default-clang-cl', '//:libpymatching'], cwd=pymatching_dir, env=env)
        # TODO: remove building stim and just build pymatching

    def build_extension(self, ext):
        num_cores = os.cpu_count()
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_CXX_COMPILER=clang++',
            '-DCMAKE_CXX_FLAGS=-fopenmp'
        ]
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--verbose']
        compiler_type = sysconfig.get_platform()
        if 'win' in compiler_type:
            build_args += ['--', '/p:CL_MPCount=' + str(num_cores)]
        else:
            build_args += ['--', f'-j{num_cores}']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        self.create_init_py()

    def create_init_py(self):
        init_file_path = os.path.join(self.install_lib, 'cpp_soft_info', '__init__.py')
        with open(init_file_path, 'w') as f:
            f.write("from .cpp_soft_info import *\n")

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
    cmdclass={
        'build_ext': CMakeBuild,
        'install': CustomInstallCommand,
    },
    python_requires=">=3.11",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    zip_safe=False,
    include_package_data=True,
    package_data={
        'cpp_soft_info': ['*.pyd'],
    },
)
