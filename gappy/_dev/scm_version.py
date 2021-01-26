import os
import os.path as pth
import warnings
try:
    from setuptools_scm import get_version
    if 'CI' in os.environ:
        # ignore warning about shallow git repositories when running
        # CI builds where this is not so important I think
        warnings.filterwarnings('ignore', '.*is shallow and may cause errors')
    version = get_version(root=pth.join('..', '..'), relative_to=__file__)
except Exception:
    raise ImportError(
        'setuptools_scm broken or not installed; could not determine package '
        'version')
