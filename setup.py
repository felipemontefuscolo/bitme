#!/usr/bin/env python
from setuptools import setup
from os.path import dirname, join

import market_maker


here = dirname(__file__)


setup(name='bitmex-trader',
      version=101,
      description='bot for BitMEX API',
      url='https://github.com/felipemontefuscolo/bitme',
      long_description=open(join(here, 'README.md')).read(),
      author='Felipe Montefuscolo',
      author_email='no-email@gmail.com',
      install_requires=[
          'requests',
          'websocket-client',
          'future'
      ],
      packages=['auth',
                'common',
                'live',
                'research',
                'sim',
                'tactic'
                'tools']#,
      #entry_points={
      #    'console_scripts': ['entry_point = market_maker:run']
      #}
      )
