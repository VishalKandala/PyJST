"""Compatibility installer for Python environments with pre-PEP-660 pip."""

from setuptools import find_packages, setup


setup(
    name="pyjst",
    version="0.1.0",
    description="A structured-grid Euler solver using JST artificial dissipation",
    packages=find_packages(include=["pyjst", "pyjst.*"]),
    install_requires=["numpy>=1.24"],
    python_requires=">=3.12",
    entry_points={"console_scripts": ["pyjst=pyjst.__main__:main"]},
)
