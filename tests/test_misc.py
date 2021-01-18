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
