from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
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

        num_cores = os.cpu_count()

        if os.environ.get('BAZEL_BUILD') == '1':
            try:
                out = subprocess.check_output(['bazel', '--version'])
            except OSError:
                raise RuntimeError(
                    "Bazel must be installed to build the following extensions: " +
                    ", ".join(e.name for e in self.extensions)
                )
            num_cores = os.cpu_count()
            # Build PyMatching with Bazel
            self.build_pymatching_with_bazel(num_cores)
        else:
            print("Skipping Bazel build due to SKIP_BAZEL_BUILD flag.")

        for ext in self.extensions:
            self.build_extension(ext, num_cores)

    def build_pymatching_with_bazel(self, num_cores):
        pymatching_dir = os.path.abspath('libs/PyMatching')
        env = os.environ.copy()
        env['BAZEL_SH'] = r'C:\Program Files\Git\bin\bash.exe'
        # Set up environment variables for Clang
        env['CC'] = 'clang-cl'
        env['CXX'] = 'clang-cl'
        # Use Clang with Bazel
        subprocess.check_call(['bazel', 'build', '--jobs', str(num_cores), '--cxxopt=/std:c++20', '--crosstool_top=@bazel_tools//tools/cpp:default-clang-cl', '//:pymatching', '//:libpymatching', '@stim//:stim_dev_wheel'], cwd=pymatching_dir, env=env)

    def build_extension(self, ext, num_cores):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_CXX_COMPILER=clang++',  # Use Clang for the extension
            '-DCMAKE_CXX_FLAGS=-fopenmp'  # Enable OpenMP support
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg, '--verbose']

        # Detect if MSBuild or make is being used
        compiler_type = sysconfig.get_platform()
        if 'win32' in compiler_type or 'win-amd64' in compiler_type:
            build_args += ['--', '/p:CL_MPCount=' + str(num_cores)]  # Use /p:CL_MPCount for MSBuild to enable multi-core builds
        else:
            build_args += ['--', f'-j{num_cores}']  # Use -j for make to enable multi-core builds

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(f"Running CMake with arguments: {cmake_args}")
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        
        if 'win3d' in compiler_type or 'win-amd64' in compiler_type:
            release_dir = os.path.join(extdir, cfg)
            print(f"Looking for .pyd files in {release_dir}")
            for filename in os.listdir(release_dir):
                if filename.endswith('.pyd'):
                    full_file_name = os.path.join(release_dir, filename)
                    print(f"Found .pyd file: {full_file_name}")
                    if os.path.isfile(full_file_name):
                        dest_file_name = os.path.join('src', filename)
                        print(f"Moving {full_file_name} to {dest_file_name}")
                        if os.path.isfile(dest_file_name):
                            os.remove(dest_file_name)
                        os.rename(full_file_name, dest_file_name)
                        # Verify the file has been moved
                        if os.path.isfile(dest_file_name):
                            print(f"Successfully moved {filename} to src directory")
                        else:
                            print(f"Failed to move {filename} to src directory")
                else:
                    print(f"Skipping {filename} as it is not a .pyd file")


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
