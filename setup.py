import os
from setuptools import setup, find_packages


setup(
    name='biasedcot',
    version='0.0.1',
    author='Yue Wan',
    author_email='yuw253@pitt.edu',
    description='Analysis of confirmation bias within chain of thoughts',
    packages=find_packages(),
    include_package_data=True,
)
