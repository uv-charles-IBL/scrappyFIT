"""Is a slow fit the machine or the code? Time it, with the machine's state.

usage: python -I tools/bench_fit.py [repo root] [--repeat N] [--threads]

Prints the date, the Windows power plan, CPU clock (% Processor Performance),
BLAS thread variables and library versions, then:

  cpu      a fixed pure-numpy workload (no scrappyFIT code): if this is slow
           too, the machine is slow, not the fit
  quartz   Session.run_fit on tests/data/quartz.txt, the default-options
           recipe of tests/test_golden_spectra.py (J1's fit), best of N

With --threads it reruns both in subprocesses under OMP_NUM_THREADS /
OPENBLAS_NUM_THREADS = unset, 1 and 16, and checks the fitted chi2 is the
same bit for bit under each.

Written 2026-10-20 (fix/fit-speed). Reference: J1.fit took 7.13 s on
2026-10-18 (tests/gui_timings_before.json).
"""
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def machine():
    info = {'date': time.strftime('%Y-%m-%d %H:%M'), 'node': platform.node(),
            'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
            'env': {k: os.environ.get(k) for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'MKL_NUM_THREADS')}}
    if sys.platform == 'win32':
        def ps(cmd):
            try:
                return subprocess.run(['powershell', '-NoProfile', '-c', cmd],
                                      capture_output=True, text=True,
                                      timeout=30).stdout.strip()
            except Exception as ex:          # noqa: BLE001
                return 'n/a (%s)' % ex
        info['power_plan'] = ps('powercfg /getactivescheme')
        info['cpu_perf_pct'] = ps(
            "(Get-Counter '\\Processor Information(_Total)\\% Processor "
            "Performance' -SampleInterval 1 -MaxSamples 3).CounterSamples | "
            "ForEach-Object { [int]$_.CookedValue }")
        info['cpu_busy_pct'] = ps(
            "(Get-CimInstance Win32_Processor).LoadPercentage")
    return info


def cpu_workload():
    """Fixed work shaped like a fit: many small numpy ops and 8k-long arrays."""
    import numpy as np
    rng = np.random.default_rng(0)
    A = rng.random((2000, 48))
    x = np.arange(8192, dtype=float)
    t0 = time.perf_counter()
    s = 0.0
    for i in range(3000):
        f = np.zeros(8192)
        f[100:400] += np.exp(-((x[100:400] - 250.0) / (3.0 + i % 7)) ** 2)
        M = A.T @ (A * f[:2000, None])
        s += float(np.linalg.inv(M + np.eye(48))[0, 0])
    return time.perf_counter() - t0


def quartz_fit(root):
    sys.path.insert(0, root)
    from scrappyfit.session import Session
    s = Session()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _paths
    s.load(_paths.lab_spectrum('quartz.txt', root))
    s.set_calibration(0.0016984, -0.37225)
    t0 = time.perf_counter()
    r = s.run_fit(['C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si'])
    return time.perf_counter() - t0, repr(float(r.chi2))


def run(root, repeat):
    cpu = min(cpu_workload() for _ in range(repeat))
    fits = [quartz_fit(root) for _ in range(repeat)]
    return {'cpu_s': round(cpu, 3),
            'quartz_s': round(min(t for t, _ in fits), 3),
            'quartz_all_s': [round(t, 3) for t, _ in fits],
            'chi2': sorted({c for _, c in fits})}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    root = os.path.abspath(args[0]) if args else ROOT
    repeat = 3
    if '--repeat' in sys.argv:
        repeat = int(sys.argv[sys.argv.index('--repeat') + 1])
    if '--child' in sys.argv:
        print(json.dumps(run(root, repeat)))
        return
    import numpy
    import scipy
    print('machine', json.dumps(machine(), indent=1))
    print('numpy %s scipy %s python %s' % (numpy.__version__, scipy.__version__,
                                           sys.version.split()[0]))
    print('root', root)
    settings = [('as is', {})]
    if '--threads' in sys.argv:
        settings += [('OMP=1', {'OMP_NUM_THREADS': '1'}),
                     ('OPENBLAS=1', {'OPENBLAS_NUM_THREADS': '1'}),
                     ('OMP=16', {'OMP_NUM_THREADS': '16'}),
                     ('unset', None)]
    for label, env in settings:
        e = dict(os.environ)
        if env is None:
            for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                e.pop(k, None)
        else:
            e.update(env)
        out = subprocess.run([sys.executable, '-I', os.path.abspath(__file__), root,
                              '--child', '--repeat', str(repeat)],
                             capture_output=True, text=True, env=e)
        res = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 \
            else {'error': out.stderr[-500:]}
        print('%-11s %s' % (label, json.dumps(res)), flush=True)


if __name__ == '__main__':
    main()
