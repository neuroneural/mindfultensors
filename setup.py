from setuptools import setup

__version__="0.0.0"
exec(open('version.py').read())
setup(name='mindfultensors',
      version=__version__,
      author='Sergey Plis',
      author_email='s.m.plis@gmail.com',
      packages=['mindfultensors'],
      url='http://pypi.python.org/pypi/mindfultensors/',
      license='MIT',
      description='Dataloader that serves MRI images from a mogodb',
      long_description_content_type="text/markdown",
      long_description=open('README.md').read(),
      install_requires=[
          "numpy",
          "scipy >= 1.7",
          "pymongo >= 4.0",
          "torch",
          ""],
    )
