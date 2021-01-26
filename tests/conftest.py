"""Additional pytest configuration."""


def pytest_report_header(config):
    try:
        from gappy import gap
    except Exception as exc:
        return f'gap: could not import gappy: {exc}'

    try:
        gap.initialize()
        return (f'gap: GAP_ROOT={gap.gap_root} '
                f'GAPInfo.Version={gap.GAPInfo.Version}')
    except Exception as exc:
        return f'gap: GAP installation not detected or broken: {exc}'
