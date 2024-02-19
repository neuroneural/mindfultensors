from setuptools import setup

setup(name='mindfultensors',
      version='0.0.1',
      author='Sergey Plis',
      author_email='s.m.plis@gmail.com',
      packages=['mindfultensors'],
      url='http://pypi.python.org/pypi/mindfultensors/',
      license='MIT',
      description='Dataloader that serves MRI images from a mogodb',
      long_description=open('README.md').read(),
      install_requires=[
          "numpy",
          "scipy >= 1.7",
          "pymongo >= 4.0",
          "redis",
          "torch",
          ""],
    )
