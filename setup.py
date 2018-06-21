try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

Description = "A pure python implementation of various standard optimization methods."

setup(
    name='Optimithon',
    version='0.2.0',
    author='Mehdi Ghasemi',
    author_email='mehdi.ghasemi@gmail.com',
    packages=['Optimithon'],
    url='https://github.com/mghasemi/Optimithon.git',
    license='MIT License',
    description=Description,
    long_description=open('readme.rst').read(),
    keywords=["Numerical", "Optimization"],
    install_requires=['numpy']
)
