This code aims to provide flexible and extendable implementation of various standard and experimental optimization
methods. The development is gradual and priority will be given to those methods that seem to be more exciting to
implement (based on personal interests).

This project was mainly motivated by my lake of knowledge in numerical optimization, otherwise there are excellent
(open source) tools for almost every task that a machine can do. This occurred while I was working on an other
scientific project on global optimization `Irene <http://irene.readthedocs.io/>`_, which tends to provide a tool to
find a lower bound for the global minimum of a function when the function and constraints are (algebraically) well
presented.

Requirements
=============================
Dependencies are minimal as the code is mainly written in python:

    - `NumPy <http://www.numpy.org/>`_.
    - `Numdifftools <https://github.com/pbrod/numdifftools>`_ (optional).

Documentation
==============================
The documentation is produced by `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ and is intended to cover code usage
as well as a bit of theory to explain each method briefly.
For more details refer to the `documentation <http://optimithon.readthedocs.io/>`_.

License
=============================
This code is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_.
