"""Bounded native vLLM/JAX capture; run by the serving owner after benchmarks."""
import argparse
import json
import signal
import time
import urllib.request
from pathlib import Path


def post(base, route, timeout):
    req = urllib.request.Request(base.rstrip('/') + route, data=b'{}',
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status


def capture(base, seconds, output, label, agents, on_started=None):
    if not 1 <= seconds <= 60:
        raise ValueError('capture duration must be 1..60 seconds')
    record = {'label': label, 'agents': agents, 'seconds_requested': seconds,
              'base_url': base, 'started_request_unix': time.time(), 'stop_ok': False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2))
    try:
        record['start_status'] = post(base, '/start_profile', 60)
        record['started_ack_unix'] = time.time()
        print('PROFILE_ACTIVE', flush=True)
        if on_started is not None:
            on_started()
        time.sleep(seconds)
    finally:
        # A timed-out start may have taken effect; always attempt stop.
        record['stop_request_unix'] = time.time()
        try:
            record['stop_status'] = post(base, '/stop_profile', 180)
            record['stop_ok'] = True
        except Exception as exc:
            record['stop_error'] = repr(exc)
            raise
        finally:
            record['finished_unix'] = time.time()
            output.write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--label', required=True)
    p.add_argument('--agents', type=int, choices=(16,32), required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    capture(a.base_url, a.seconds, a.output, a.label, a.agents)

if __name__ == '__main__':
    main()
