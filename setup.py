"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
import os, codecs

# Get version
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'cwm', 'version.py')).read())

# Get the long description from the README file
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

__version__ = '0.0.1'

setup(
    name='cwm',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='Counterfactual World Models',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/neuroailab/CounterfactualWorldModels',

    # Author details
    author='Stanford Neuroscience and Artificial Intelligence Lab',
    author_email='honglinc@stanford.edu',

    # Choose your license
    license='Stanford',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],

    # What does your project relate to?
    keywords='deep learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html

    install_requires=[
        'numpy',
        'scipy',
        'matplotlib==3.5.2',
        'itermplot',
        'h5py',
        'torch',
        'torchvision',
        'av',
        'torchaudio',
        'timm',
        'kornia',
        'future',
        'einops',
        'opencv-python',
        'iopath',
        'IPython',
        'jupyter',
        'jupyterlab',
        'jupyter-client==7.3.1',
        'tornado==6.1'
    ],
)

