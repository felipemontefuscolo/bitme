#!/usr/bin/env python
from setuptools import setup
import os
from os.path import dirname, join, realpath

here = dirname(realpath(__file__))

packs = [dir for dir in os.listdir(here) if os.path.isdir(dir) if '__init__.py' in os.listdir(dir)]

setup(name='bitmex-trader',
      version=0.01,
      description='bot for BitMEX API',
      url='https://github.com/felipemontefuscolo/bitme',
      long_description=open(join(here, 'README.md')).read(),
      author='Felipe Montefuscolo',
      author_email='no-email@gmail.com',
      install_requires=[
          'requests',
          'websocket-client',
          'cdecimal',
          'pandas',
          'simpy'
      ],
      packages=packs
      )
