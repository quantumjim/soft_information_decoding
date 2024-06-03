from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
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
            out = subprocess.check_output(['cmake', '--version'])
            out = subprocess.check_output(['bazel', '--version'])
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
        self.move_pyd_files(extdir, cfg)

    def move_pyd_files(self, extdir, cfg):
        release_dir = os.path.join(extdir, cfg)
        target_dir = os.path.join('src', 'cpp_soft_info')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for filename in os.listdir(release_dir):
            if filename.endswith('.pyd'):
                src_file = os.path.join(release_dir, filename)
                dest_file = os.path.join(target_dir, filename)
                if os.path.isfile(dest_file):
                    os.remove(dest_file)
                os.rename(src_file, dest_file)
                print(f"Moved {filename} to {dest_file}")

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
    include_package_data=True,
    package_data={
        'cpp_soft_info': ['*.pyd'],
    },
)
