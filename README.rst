# coding=utf-8================================================
imgProcessor - A python image processing library
================================================

.. image:: https://img.shields.io/badge/License-GPLv3-red.svg
.. image:: https://img.shields.io/badge/python-2.6%7C2.7-yellow.svg

Based on `scikit-image <http://scikit-image.org/docs/dev/install.html>`_,`numba <http://numba.pydata.org>`_ and `OpenCV2.4 <http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv>`_

- Browse the `API Documentation <http://radjkarl.github.io/imgProcessor>`_
- Fork the code on `github <https://github.com/radjkarl/imgProcessor>`_

.. image:: https://cloud.githubusercontent.com/assets/350050/15593492/ee8924a8-2369-11e6-9127-45752628e22d.png
    :align: center
    :alt: showcase

Installation
^^^^^^^^^^^^

**imgProcessor** is listed in the Python Package Index. You can install it typing::

    pip install imgProcessor

Tests
^^^^^
**fancyWidgets** uses mostly the 'one class/function per file' rule. Running each module as program, like::

    python -m imgProcessor.camera.PerspectiveCorrection

will execute the test procedure of this module.

To run all tests type::

    python -m imgProcessor.tests
