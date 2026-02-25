import shutil
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from setuptools.command.egg_info import egg_info


def vendor_script():
    base_dir = Path(__file__).resolve().parent
    src = base_dir.parent / "src" / "moon_reader" / "compute_read_stats.py"
    dst_dir = base_dir / "moon_reader"

    if src.exists():
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / "compute_read_stats.py"
        shutil.copy(src, dst)
        (dst_dir / "__init__.py").touch(exist_ok=True)


class VendorEggInfo(egg_info):
    def run(self):
        vendor_script()
        super().run()


class VendorBuildPy(build_py):
    def run(self):
        vendor_script()
        super().run()


class VendorSdist(sdist):
    def run(self):
        vendor_script()
        super().run()


setup(
    cmdclass={
        "egg_info": VendorEggInfo,
        "build_py": VendorBuildPy,
        "sdist": VendorSdist,
    }
)
