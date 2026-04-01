"""Build script for g1_bridge_sdk — pybind11 C++ DDS bridge.

Uses CMakeExtension + CMakeBuild pattern so ``pip install .`` triggers
cmake → make → installs the resulting .so into the Python environment.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent.resolve()

        cfg = "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_args = ["--config", cfg]

        # Allow parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            ncpu = os.cpu_count() or 1
            build_args += [f"-j{ncpu}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )


setup(
    name="g1_bridge_sdk",
    version="0.1.0",
    description="G1 C++ DDS bridge via pybind11 (unitree_sdk2)",
    ext_modules=[CMakeExtension("g1_bridge_sdk")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
)
