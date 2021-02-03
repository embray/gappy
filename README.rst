|logo|

#################################
gappy — a Python interface to GAP
#################################

|docs-badge| |tests-badge|

gappy provides a Python interface to the `GAP
<https://www.gap-system.org/>`_ computer algebra system by linking to its
library interface.

It allows calling functions in GAP directly from Python, and passing
supported Python objects back to GAP.

gappy is based on SageMath's `LibGAP
<https://doc.sagemath.org/html/en/reference/libs/sage/libs/gap/libgap.html>`_
interface to GAP, originally developed by Volker Braun, but is completely
independent of Sage--it does not require or use Sage at all, and can be used
in any Python code.  If there is enough interest, it may also be enhanced
with a complementary GAP package for interacting with Python from within
GAP.


.. contents::
    :local:
    :depth: 3


Quickstart
==========

To start using GAP functions from Python, just run:

.. code-block:: python

    >>> from gappy import gap

Then any global variable in GAP, including functions, can be accessed as
attributes on `gap` like:

.. code-block:: python

    >>> gap.Cite()
    Please use one of the following samples
    to cite GAP version from this installation

    Text:

    [GAP] GAP – Groups, Algorithms, and Programming, Version 4.dev, The GAP Group, https://www.gap-system.org.
    ...

All global variables that would be available in a GAP session can be
accessed in this way:

.. code-block:: python

    >>> GAPInfo.Version
    "4.dev"

Most basic Python types have direct equivalents in GAP, and can be passed
directly to GAP functions without explicit conversion to their equivalent
GAP types:

.. code-block:: python

    >>> S4 = gap.SymmetricGroup(4)
    >>> S4
    Sym( [ 1 .. 4 ] )

You can also call "methods" on ``GapObj``\s.  This is just syntactic sugar
for calling a GAP function with that object as its first argument, in cases
where that function supports the object bound to the method.  For example:

.. code-block:: python

    >>> S4.GeneratorsOfGroup()
    [ (1,2,3,4), (1,2) ]

Values returned from GAP functions are GAP objects wrapped in a Python class
for containing them called ``GapObj``:

.. code-block:: python

    >>> type(S4)
    <class 'gappy.gapobj.GapObj'>

There are also specialized subclasses of ``GapObj`` for many types of objects
in GAP.  To explicitly convert a Python object directly to its GAP
equivalent, you can *call* ``gap`` like:

.. code-block:: python

    >>> one = gap(1)
    >>> type(one)
    <class 'gappy.gapobj.GapInteger'>

GAP objects are displayed (with `repr`) or stringified (with `str`) the same
way they would be in GAP, when displaying the object in the REPL or when
calling GAP's ``Print()`` function on the object, respectively:

.. code-block:: python

    >>> one
    1
    >>> s = gap("Hello GAP!")
    >>> s
    "Hello GAP!"
    >>> print(s)
    Hello GAP!

Not all GAP objects have an equivalent in basic Python types, so there is
no implicit conversion from GAP back to Python.  However, all Python types
that can be converted to GAP objects can be converted back to their
equivalent Python types in a symmetrical manner:

.. code-block:: python

    >>> int(one)
    1
    >>> type(int(one))
    <class 'int'>
    >>> str(s)
    'Hello GAP!'
    >>> type(str(s))
    <class 'str'>

Likewise for `float`\s, `list`\s, `dict`\s, among others.

Finally, you can execute arbitrary GAP code directly with ``gap.eval``.
This is often the easiest way to construct more complicated GAP objects,
especially if you are more familiar with GAP syntax.  The return value of
``gap.eval`` is the result of evaluating the same statement in GAP (the
semicolon is optional when evaluating a single statement):

.. code-block:: python

    >>> rec = gap.eval('rec(a:=123, b:=456, Sym3:=SymmetricGroup(3))')
    >>> rec['Sym3']
    Sym( [ 1 .. 3 ] )

This is also an easy way to declare new GAP functions from gappy:

.. code-block:: python

    >>> sign = gap.eval("""sign := function(n)
    ...     if n < 0 then
    ...         return -1;
    ...     elif n = 0 then
    ...         return 0;
    ...     else
    ...         return 1;
    ...     fi;
    ... end;""")
    >>> sign
    <GAP function "sign">
    >>> sign(0)
    0
    >>> sign(-99)
    -1

See the full API documentation for many additional examples of how to use
the ``gap`` object as well as the built-in ``GapObj`` types.


Installation
============

.. note::

    These instructions will be updated once there are releases on PyPI.

Prerequisites
-------------

* Supported platforms: Linux, MacOS, Cygwin.

  * Likely works with most other \*BSD flavors but has not been tested.

* Python 3.6 or up with development headers installed.  On Debian-based
  systems this means:

  .. code-block:: shell

      $ sudo apt-get install python3.7-dev

* GAP 4.10.2 or greater

Currently it is necessary to install from source:

.. code-block:: shell

    $ git clone https://github.com/embray/gappy.git
    $ cd gappy/

It is possible to install gappy in the usual way using pip:

.. code-block:: shell

    $ pip install .

However, depending on how GAP is installed, some extra steps may be
required.  In particular, if you installed GAP from source using the
typical instructions on the `GAP website
<https://www.gap-system.org/Download/index.html>`_ you will need to make
sure the libgap shared library is built by running:

.. code-block:: shell

    $ make install-libgap

in the GAP source directory.

You will also need to point to the location of your GAP installation by
setting the ``GAP_ROOT`` environment variable like:

.. code-block:: shell

    $ GAP_ROOT=<path/to/gap/root> pip install .

If you needed to provide ``GAP_ROOT`` for the installation, it is also
generally necessary to set this environment variable *before* using gappy,
so that it can find the path to your GAP installation.  See the
documentation for the ``Gap`` class for more information.

If using GAP from a distribution system such as APT on Debian/Ubuntu or from
Conda, however, the GAP library (libgap) is typically installed in a
standard system location, and it may not be necessary to provide
``GAP_ROOT``.  See the next section for example.

Conda installation
------------------

To give an example of the above point, you can install gappy in a Conda
environment as follows:

.. code-block:: shell

    $ conda create -n gap
    $ conda activate gap
    $ conda install -c conda-forge gap-defaults==4.11 python==3.8
    $ pip install .

Alternatively, you can create the conda environment using the supplied
`environment.yml
<https://github.com/embray/gappy/blob/master/environment.yml>`_ file:

.. code-block:: shell

    $ conda env create

.. note::

    With Conda and other distributions that install libgap to a standard
    system location (e.g. ``/usr/lib/libgap.so``) it may not be necessary to
    set the ``GAP_ROOT`` environment variable, as the library can locate
    your GAP root automatically in most cases.

.. warning::

    The conda package for GAP 4.11 had dependency conflicts with Python 3.7
    so you must use Python 3.8 or above, or GAP 4.10.2 with Python 3.7.

Cygwin installation
-------------------

Additional notes for installation on Cygwin:

* The dependency ``psutil`` does not support Cygwin.  However, there is an
  unofficial fork which does at:
  https://github.com/embray/psutil/tree/cygwin/v3.  You can install it by
  running:

  .. code-block:: shell

      $ pip install git+https://github.com/embray/psutil.git@cygwin/v3

* The path to the libgap DLL (filename ``cyggap-0.dll``) needs to be on
  your ``PATH`` environment variable in order for gappy to be importable.
  To do this you can either copy it from your GAP installation to a standard
  location like:

  .. code-block:: shell

      $ cp /path/to/gap_root/.libs/cyggap-0.dll /usr/local/bin

  or you can modify your environment to point to where GAP places the built
  DLL:

  .. code-block:: shell

    $ export PATH="/path/to/gap_root/.libs:$PATH"

  and add this to your ``.profile``.

.. |logo| image:: https://raw.githubusercontent.com/embray/gappy/master/docs/images/gappy-logo.svg.png
    :alt: gappy logo
    :align: middle

.. |docs-badge| image:: https://readthedocs.org/projects/gappy/badge/?version=latest
    :target: https://gappy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |tests-badge| image:: https://github.com/embray/gappy/workflows/Tests/badge.svg
    :target: https://github.com/embray/gappy/actions?query=workflow%3ATests
    :alt: Test Status
