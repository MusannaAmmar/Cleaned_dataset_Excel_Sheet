from setuptools import setup, find_packages

setup(
    name="financial-data-parser",
    version="0.1",
    packages=find_packages(include=['config', 'src', 'src.core']),
    package_dir={
        '': '.',  # root package is in the current directory
    }
)