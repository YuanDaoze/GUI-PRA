"""Check controller discovery and worker/API availability; optionally execute UI tools.

This script intentionally depends only on the UI-Tool HTTP contract.  Keeping the
check independent from ``os-critic-bench`` means it can be shipped and used from
the tool-only archive.
"""
import argparse
import base64
import json
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from urllib.request import build_opener, ProxyHandler, Request

ROOT = Path(__file__).resolve().parents[1]


class ToolClient:
    """Small dependency-light client for the Controller and worker endpoints."""

    def __init__(self, controller):
        # Keep ``--help`` usable before the optional runtime dependencies are
        # installed; the real service check still requires requests.
        import requests
        self.controller = controller.rstrip('/')
        self.session = requests.Session()
        self.session.trust_env = False
        models = self.post(self.controller + '/list_models', {}, timeout=10).get('models', [])
        self.available_tools = list(models)
        self.addresses = {}
        for name in self.available_tools:
            response = self.post(self.controller + '/get_worker_address', {'model': name}, timeout=10)
            if response.get('address'):
                self.addresses[name] = self._rewrite_worker_address(response['address']).rstrip('/')

    def _rewrite_worker_address(self, address):
        """Make controller-returned loopback worker URLs usable remotely."""
        if os.environ.get('GUIPRA_REWRITE_LOCAL_WORKER_ADDR', '1') == '0':
            return address
        worker, controller = urlparse(address), urlparse(self.controller)
        local = {'localhost', '127.0.0.1', '0.0.0.0'}
        if worker.hostname in local and controller.hostname not in local and worker.port:
            return urlunparse(worker._replace(netloc=f'{controller.hostname}:{worker.port}'))
        return address

    def post(self, url, payload, timeout):
        response = self.session.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def call_tool(self, name, payload):
        address = self.addresses.get(name)
        if not address:
            return {'error_code': 1, 'text': f'No registered worker for {name}'}
        return self.post(address + '/worker_generate', payload, timeout=120)


def check_model(base_url, model, api_key=None):
    root = base_url.rstrip('/')
    if root.endswith('/chat/completions'):
        root = root[:-len('/chat/completions')]
    if not root.endswith('/v1'):
        root += '/v1'
    headers = {'Authorization': 'Bearer ' + api_key} if api_key else {}
    request = Request(root + '/models', headers=headers)
    with build_opener(ProxyHandler({})).open(request, timeout=10) as stream:
        models = [item['id'] for item in json.load(stream).get('data', [])]
    if model not in models:
        raise RuntimeError(f'Model {model!r} is not served at {root}; available={models}')
    return {'url': root, 'model': model}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', default=os.environ.get('GUIPRA_TOOL_CONTROLLER_ADDR', 'http://127.0.0.1:20001'))
    p.add_argument('--router-url', default=os.environ.get('GUIPRA_VLLM_BASE_URL'))
    p.add_argument('--router-model', default=os.environ.get('GUIPRA_ROUTER_MODEL', 'qwen3-vl-8b-instruct'))
    p.add_argument('--zoom-url', default=os.environ.get('ZOOM_API_URL'))
    p.add_argument('--zoom-model', default=os.environ.get('ZOOM_MODEL_NAME', 'qwen3-vl-32b-instruct'))
    p.add_argument('--smoke', action='store_true', help='Run real Point, OmniParser and Zoom calls on a synthetic screenshot.')
    p.add_argument('--image', default=str(ROOT / 'assets/smoke_ui.png'))
    p.add_argument('--output-dir', default=str(ROOT / 'runtime/tool-smoke'))
    args = p.parse_args()
    manager = ToolClient(args.controller)
    required = {'Point', 'omni_parser', 'ZoomInSubfigure'}
    missing = required - set(manager.available_tools)
    if missing:
        raise RuntimeError(f'Missing workers: {sorted(missing)}')
    for name in sorted(required):
        address = manager.addresses.get(name)
        if not address:
            raise RuntimeError(f'Controller did not return an address for {name}')
        status = manager.post(address + '/worker_get_status', {}, timeout=10)
        if name not in status.get('model_names', []):
            raise RuntimeError(f'Unexpected worker status for {name}: {status}')
    print('All three workers are registered and reachable.')
    if args.router_url:
        print(check_model(args.router_url, args.router_model, os.environ.get('GUIPRA_API_KEY')))
    if args.zoom_url:
        print(check_model(args.zoom_url, args.zoom_model, os.environ.get('ZOOM_API_KEY')))
    if args.smoke:
        image = base64.b64encode(Path(args.image).read_bytes()).decode()
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name in ['Point', 'omni_parser', 'ZoomInSubfigure']:
            params = {'image': image}
            if name != 'omni_parser':
                params['param'] = 'The blue Save button in the center'
            result = manager.call_tool(name, params)
            if result.get('error_code', 1) != 0 or not result.get('edited_image'):
                raise RuntimeError(f'{name} smoke failed: {result.get("text")}')
            raw = base64.b64decode(result.pop('edited_image'))
            from PIL import Image
            from io import BytesIO
            with Image.open(BytesIO(raw)) as im:
                im.save(output / (name + '.png'))
            (output / (name + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
            print(f'{name}: response and image saved in {output}')


if __name__ == '__main__':
    main()
