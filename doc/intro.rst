=====================
Introduction
=====================

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

Download
================
`Optimithon` can be obtained from `https://github.com/mghasemi/optimithon <https://github.com/mghasemi/>`_.

Installation
=========================
To install `Optimithon`, run the following in terminal::

    sudo python setup.py install

Documentation
==============================
The documentation is produced by `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ and is intended to cover code usage
as well as a bit of theory to explain each method briefly.
For more details refer to the `documentation <http://optimithon.readthedocs.io/>`_.

License
=============================
This code is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_:

MIT License
------------------

    Copyright (c) 2018 Mehdi Ghasemi

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.