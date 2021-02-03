"""Additional tests for gappy"""


from gappy import gap


def test_write_to_file(tmp_path):
    """
    Test that libgap can write to files

    See :trac:`16502`, :trac:`15833`.
    """
    fname = str(tmp_path / 'test.txt')
    message = "Ceci n'est pas une groupe"
    gap.PrintTo(fname, message)
    with open(fname, 'r') as f:
        assert f.read() == message

    assert gap.StringFile(fname) == message


def test_gap_exec(capfd):
    """A regression test originally from Sage."""

    gap.Exec('echo hello from the shell')
    stdio = capfd.readouterr()
    assert stdio.out.rstrip() == 'hello from the shell'


def test_gap_function_re():
    """Tests of the regular expression for GAP function declarations."""

    m = gap._gap_function_re.match('function()')
    assert m and m.group() == 'function()'
    m = gap._gap_function_re.search('''
        blah blah blah

        function ( a, b )
    ''')
    assert m and m.group().strip() == 'function ( a, b )'


def test_lazy_function_as_argument():
    """
    Regression test for bug with lazy functions.

    When a lazy function is used as an argument to another GAP function, ensure
    that the function is initialized.
    """

    def make_gap_function():
        gap._gap_function.cache_clear()

        @gap.gap_function
        def OnTuples(omega, g):
            """
            Just a wrapper for OnTuples to demonstrate the bug.

            function(omega, g)
                return OnTuples(omega, g);
            end;
            """

        return OnTuples

    G = gap.Group(gap.eval('(1,2,3)'), gap.eval('(2,3,4)'))
    O = gap.Orbit(G, [1], make_gap_function())
    assert O == [[1], [2], [3], [4]]

    # Make sure it works even if the lazy function is wrapped in some other
    # object that has a converter to GapObj registered (regression test from
    # the Sage integration)
    class MyGapFunction:
        def __init__(self, obj):
            self.obj = obj

    gap.register_converter(MyGapFunction, lambda mgf, gap: mgf.obj)
    wrapped_OnTuples = MyGapFunction(make_gap_function())
    O = gap.Orbit(G, [1], wrapped_OnTuples)
    assert O == [[1], [2], [3], [4]]
