
from setuptools import setup, find_packages
from setuptools.command.install import install

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="mini_LiTOY",
    version="0.1.5",
    description=" Minimalist LiTOY task sorting algorithm based on ELO scores",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thiswillbeyourgithub/mini_LiTOY",
    packages=find_packages(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    license="GPLv3",
    keywords=["tasks", "sorting", "ELO", "scoring", "organization", "smart", "tool", "productivity", "litoy", "list", "life", "learning", "optimization"],
    python_requires=">=3.11",

    entry_points={
        'console_scripts': [
            'mlitoy=mini_LiTOY.__init__:cli_launcher',
        ],
    },

    install_requires=[
        "fire >= 0.6.0",
        "typeguard >= 0.4.3",
        "rich>=13.7.1",
        "prompt-toolkit>=3.0.40",
        "platformdirs >= 4.2.2",
        "rtoml >= 0.11.0",
    ],

)
