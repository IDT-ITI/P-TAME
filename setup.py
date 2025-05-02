#!/usr/bin/env python

from setuptools import setup

setup(
    name="ptame",
    version="1.0.0",
    description="P-TAME project",
    author="Mariano V. Ntrougkas",
    author_email="ntrougkas@iti.gr",
    url="https://github.com/marios1861/PAMELA",
    install_requires=["lightning", "hydra-core"],
    entry_points={
        "console_scripts": [
            "ptame-train = ptame.train:main",
            "ptame-print = ptame.print_map:main",
            "ptame-sweep = ptame.ax_sweep:main",
        ]
    },
)
