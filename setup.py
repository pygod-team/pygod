from setuptools import find_packages, setup

setup(name='pygod',
      version='0.1.0',
      description='Graph Outlier/Anomaly Detection in Python',
      author='PyGOD Team',
      author_email='dev@pygod.org',
      packages=find_packages(),
      install_requires=[
          'torch_geometric>=2.0.3'
      ],
      package_data={"": ["./*"]},
      include_package_data=True,
      license='APACHE',
      entry_points={
          'console_scripts': [
              "pygod = pygod.cli.cli:main"
          ]
      },
      url='https://github.com/pygod-team/pygod',
      )