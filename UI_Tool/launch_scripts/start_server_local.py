"""Start an isolated local controller and UI workers from the release configuration."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, build_opener, ProxyHandler
import uuid

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}')


def expand(value, env):
    if isinstance(value, str):
        def substitute(match):
            key, default = match.groups()
            result = env.get(key) or default
            if result is None:
                raise ValueError(f'Set {key} before using this configuration')
            return result
        return ENV_PATTERN.sub(substitute, value)
    if isinstance(value, list):
        return [expand(item, env) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, env) for key, item in value.items()}
    return value


def load_config(path):
    env = dict(os.environ, PROJECT_ROOT=str(PROJECT_ROOT))
    with Path(path).open(encoding='utf-8') as stream:
        config = expand(yaml.safe_load(stream), env)
    if not isinstance(config, dict) or 'controller_config' not in config:
        raise ValueError('Configuration requires controller_config')
    return config


def post_json(url, payload=None, timeout=5):
    opener = build_opener(ProxyHandler({}))
    request = Request(url, data=json.dumps(payload or {}).encode(),
                      headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


class ServerManager:
    def __init__(self, config):
        self.config = config
        base = Path(config.get('base_dir', 'UI_Tool'))
        self.base_dir = (base if base.is_absolute() else PROJECT_ROOT / base).resolve()
        runtime = Path(config.get('runtime_dir', 'runtime/ui-tool'))
        self.runtime_root = (runtime if runtime.is_absolute() else PROJECT_ROOT / runtime).resolve()
        self.run_dir = self.runtime_root / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
        self.controller = config['controller_config']
        self.controller_addr = f"http://127.0.0.1:{int(self.controller['cmd']['port'])}"
        self.workers = [next(iter(item.values())) for item in config.get('tool_worker_config', [])]
        self.processes = []
        self.files = []

    def command(self, component, controller=False):
        options = component['cmd']
        script = (self.base_dir / options['script-addr']).resolve()
        if not script.is_file():
            raise FileNotFoundError(script)
        command = [component.get('python', sys.executable), '-u', str(script)]
        if not controller:
            command += ['--controller-address', self.controller_addr]
            if 'worker-address' not in options:
                command += ['--worker-address', f"http://127.0.0.1:{int(options['port'])}"]
        for key, value in options.items():
            if key == 'script-addr' or value is None or value is False:
                continue
            command.append('--' + key)
            if value is not True:
                command.append(str(value))
        return command

    def validate(self):
        ports = []
        for component in [self.controller] + self.workers:
            self.command(component, component is self.controller)
            port = int(component['cmd']['port'])
            if not 1 <= port <= 65535 or port in ports:
                raise ValueError(f'Duplicate or invalid port: {port}')
            ports.append(port)
            for key in ['model_path', 'som-model-path', 'caption-model-path']:
                if key in component['cmd'] and not Path(component['cmd'][key]).exists():
                    raise FileNotFoundError(f"{key}: {component['cmd'][key]}")

    def plan(self):
        return {'controller': self.controller_addr, 'run_dir': str(self.run_dir),
                'components': [
                    {'name': item['worker_name'], 'gpu': item.get('cuda_visible_devices', ''),
                     'command': self.command(item, item is self.controller)}
                    for item in [self.controller] + self.workers]}

    def check_ports(self):
        for component in [self.controller] + self.workers:
            with socket.socket() as sock:
                # Wildcard bind also detects a service listening on another local interface.
                sock.bind(('0.0.0.0', int(component['cmd']['port'])))

    def start_process(self, component, controller=False):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(PROJECT_ROOT)
        env['OMP_NUM_THREADS'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['UI_TOOL_LOG_DIR'] = str(self.run_dir / 'logs')
        env['MPLCONFIGDIR'] = str(self.run_dir / 'cache/matplotlib')
        env['YOLO_CONFIG_DIR'] = str(self.run_dir / 'cache/ultralytics')
        env['HF_HOME'] = env.get('UI_TOOL_HF_HOME', str(self.runtime_root / 'cache/huggingface'))
        if 'cuda_visible_devices' in component:
            env['CUDA_VISIBLE_DEVICES'] = str(component['cuda_visible_devices'])
        logfile = self.run_dir / 'logs' / (component['worker_name'] + '.stdout.log')
        stream = logfile.open('x', encoding='utf-8')
        self.files.append(stream)
        command = self.command(component, controller)
        print(f"Starting {component['worker_name']}: {shlex.join(command)}", flush=True)
        process = subprocess.Popen(command, cwd=self.run_dir, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append((component['worker_name'], process))

    def check_alive(self):
        for name, process in self.processes:
            if process.poll() is not None:
                raise RuntimeError(f'{name} exited with {process.returncode}; see {self.run_dir}/logs')

    def wait_ready(self, component=None):
        deadline = time.monotonic() + float(self.config.get('startup_timeout', 600))
        last_error = None
        while time.monotonic() < deadline:
            self.check_alive()
            try:
                if component is None:
                    post_json(self.controller_addr + '/list_models')
                else:
                    name = component.get('registered_name', component['worker_name'])
                    address = post_json(self.controller_addr + '/get_worker_address', {'model': name}).get('address')
                    if not address:
                        raise RuntimeError(f'{name} has not registered')
                    status = post_json(address.rstrip('/') + '/worker_get_status')
                    if name not in status.get('model_names', []):
                        raise RuntimeError(f'{name} status does not match')
                return
            except Exception as error:
                last_error = error
            time.sleep(float(self.config.get('retry_interval', 1)))
        raise TimeoutError(f'Service startup timed out: {last_error}; logs: {self.run_dir}/logs')

    def start(self):
        self.validate()
        self.check_ports()
        (self.run_dir / 'logs').mkdir(parents=True)
        self.start_process(self.controller, controller=True)
        self.wait_ready()
        for component in self.workers:
            self.start_process(component)
            self.wait_ready(component)
        info = {'controller_addr': self.controller_addr,
                'workers': [item.get('registered_name', item['worker_name']) for item in self.workers],
                'pids': {name: proc.pid for name, proc in self.processes}}
        (self.run_dir / 'controller_addr.json').write_text(json.dumps(info, indent=2) + '\n')
        print(f'All services ready at {self.controller_addr}. Runtime: {self.run_dir}', flush=True)

    def shutdown(self):
        # Each child is in a process group created by this manager. No global process search.
        for name, process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
        for stream in self.files:
            stream.close()
        if self.run_dir.exists():
            (self.run_dir / 'stopped.txt').write_text('Services owned by this launcher have stopped.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'UI_Tool/launch_scripts/config/all_service_example_local_1.yaml'))
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved commands; do not bind ports or load models.')
    args = parser.parse_args()
    manager = ServerManager(load_config(args.config))
    if args.dry_run:
        print(json.dumps(manager.plan(), indent=2))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        manager.start()
        while True:
            manager.check_alive()
            time.sleep(1)
    except KeyboardInterrupt:
        print('Stopping services started by this launcher.', flush=True)
    finally:
        manager.shutdown()


if __name__ == '__main__':
    main()
