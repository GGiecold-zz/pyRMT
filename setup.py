#!/usr/bin/env python
# -*- coding: utf-8 -*-


from codecs import open
from os import path
from sys import version
from setuptools import setup, find_packages

    
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding = 'utf-8') as f:
    long_description = f.read()
    

setup(name = 'pyRMT',
      version = '0.1.0',
      
      description = 'Python for Random Matrix Theory: cleaning schemes for noisy correlation matrices',
      long_description = long_description,
                    
      url = 'https://github.com/GGiecold/pyRMT',
      download_url = 'https://github.com/GGiecold/pyRMT',
      
      author = 'Gregory Giecold',
      author_email = 'g.giecold@gmail.com',
      maintainer = 'Gregory Giecold',
      maintainer_email = 'g.giecold@gmail.com',
      
      license = 'MIT License',
      
      packages = find_packages(),
      
      py_modules = ['pyRMT'],
      platforms = ('Any',),
      install_requires = ['numpy', 'pandas', 'sklearn'],
                          
      classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: End Users/Desktop',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',          
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Mathematics', ],
                   
      keywords = 'applied-mathematics cleaning correlation-matrices noise-reduction random-matrix-theory',
)
