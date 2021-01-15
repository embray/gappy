API Documentation
#################

.. contents::
    :local:
    :depth: 3

.. py:data:: gap
    :type: gappy.core.Gap

    The default GAP interpreter instance.  Most users can run::

        >>> from gappy import gap

    and immediately begin using GAP from here.  However, if you wish to
    customize the initialization parameters of the GAP interpreter (e.g.
    set the ``gap_root`` path) you can run::

        >>> from gappy import Gap
        >>> gap = Gap(...)

    .. note::

        Upon first using ``gap``, whether to access a global variable run
        a function, there may be a noticeable delay upon GAP initialization;
        after the first use it will be faster.

``gappy.core``
==============

.. automodule:: gappy.core
    :members:


``gappy.gapobj``
================

.. automodule:: gappy.gapobj
    :members:


``gappy.exceptions``
====================

.. automodule:: gappy.exceptions
    :members:


``gappy.context_managers``
==========================

.. automodule:: gappy.context_managers
    :members:
