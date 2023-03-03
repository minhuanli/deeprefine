from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("deeprefine/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

__version__ = getVersionNumber()

setup(name="deeprefine",
    version=__version__,
    author="Minhaun Li",
    license="MIT",
    description="Pytorch implementation of deep generative model assisted structure refinement", 
    # url="",
    author_email='minhuanli@g.harvard.edu',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[],
    entry_points={
        "console_scripts": [
            "dr.preppdb=deeprefine.commandline.preppdb:main"
        ]
    }
)