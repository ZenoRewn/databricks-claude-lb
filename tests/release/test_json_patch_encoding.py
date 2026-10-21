"""Preserve full CAS tests across JSON string encodings. Author: Zeno Ren."""
import copy
import json
import ssl
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from operations.release import kube


COMMAND = 'python - <<\'PY\'\nif 1 < 2 and 3 > 2:\n print("<&>", "汉😀", "\\\\u003c", "\\t")\nPY\n'


def html_json(text):
    return text.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


@pytest.mark.parametrize('as_bytes', [False, True])
def test_patch_encoding_preserves_json_values_and_all_other_request_options(as_bytes):
    operations = [{'op': 'test', 'path': '/spec/template',
                   'value': {'command': COMMAND, 'escaped': '\\u003c', 'controls': '\t\r\n\b\f\x00'}}]
    original = json.dumps(operations, ensure_ascii=False)
    body = original.encode() if as_bytes else original
    headers = {'Content-Type': 'application/json-patch+json', 'Authorization': 'Bearer synthetic-fixture',
               'Content-Length': str(len(original.encode()))}
    before = copy.deepcopy(headers)
    result = object(); inner = SimpleNamespace(request=Mock(return_value=result), clear=Mock(),
                                               connection_pool_kw={'cert_reqs': 'CERT_REQUIRED'})
    wrapped = kube._JSONPatchPool(inner)
    returned = wrapped.request('PATCH', 'https://fixture.invalid/resource', body=body, headers=headers,
                               timeout=(3, 8), preload_content=True, retries=False)
    assert returned is result
    args, kwargs = inner.request.call_args
    assert args == ('PATCH', 'https://fixture.invalid/resource')
    encoded = kwargs['body']
    assert isinstance(encoded, bytes if as_bytes else str)
    assert json.loads(encoded) == operations
    assert encoded == (html_json(original).encode() if as_bytes else html_json(original))
    assert kwargs['headers']['Authorization'] == headers['Authorization']
    assert kwargs['headers']['Content-Length'] == str(len(encoded if as_bytes else encoded.encode()))
    assert kwargs['timeout'] == (3, 8) and kwargs['preload_content'] is True and kwargs['retries'] is False
    assert headers == before
    assert wrapped.connection_pool_kw is inner.connection_pool_kw
    wrapped.clear(); inner.clear.assert_called_once()


@pytest.mark.parametrize('method,content_type,body', [
    ('GET', 'application/json-patch+json', '<>&'),
    ('POST', 'application/json-patch+json', '<>&'),
    ('PATCH', 'application/strategic-merge-patch+json', '<>&'),
    ('PATCH', 'application/merge-patch+json', '<>&'),
    ('PATCH', 'application/json', '<>&'),
    ('PATCH', 'application/json-patch+json', '[{"op":"test","value":"plain"}]'),
])
def test_unrelated_methods_formats_and_already_safe_bodies_are_unchanged(method, content_type, body):
    headers = {'Content-Type': content_type}
    inner = SimpleNamespace(request=Mock())
    kube._JSONPatchPool(inner).request(method, 'https://fixture.invalid', body=body, headers=headers, timeout=7)
    kwargs = inner.request.call_args.kwargs
    assert kwargs['body'] is body and kwargs['headers'] is headers
    assert kwargs['timeout'] == 7


def test_header_case_charset_and_byte_content_length_are_handled():
    headers = {'content-type': 'Application/JSON-Patch+JSON; charset=utf-8', 'content-length': '3'}
    inner = SimpleNamespace(request=Mock())
    kube._JSONPatchPool(inner).request('patch', 'https://fixture.invalid', body=b'"<"', headers=headers)
    assert inner.request.call_args.kwargs['body'] == b'"\\u003c"'
    assert inner.request.call_args.kwargs['headers']['content-length'] == '8'
    assert headers['content-length'] == '3'


@pytest.mark.parametrize('body', [None, {}, [], bytearray(b'[]'), b'\xff'])
def test_unexpected_serialized_body_is_rejected_before_transport(body):
    inner = SimpleNamespace(request=Mock())
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        kube._JSONPatchPool(inner).request('PATCH', 'https://fixture.invalid', body=body,
                                         headers={'Content-Type': 'application/json-patch+json'})
    inner.request.assert_not_called()


def test_errors_are_delegated_without_logging_or_reclassification():
    failure = RuntimeError('synthetic pool failure')
    inner = SimpleNamespace(request=Mock(side_effect=failure))
    with pytest.raises(RuntimeError) as caught:
        kube._JSONPatchPool(inner).request('PATCH', 'https://fixture.invalid', body='["<"]',
                                         headers={'Content-Type': 'application/json-patch+json'})
    assert caught.value is failure


class RawStringTestPool:
    """Emulate the observed API comparison after the real SDK serializes."""
    def __init__(self):
        self.calls = []
        self.connection_pool_kw = {'cert_reqs': 'CERT_REQUIRED', 'ca_certs': 'synthetic-ca.pem'}
        self.current = {'metadata': {'name': 'lb', 'uid': 'u', 'resourceVersion': '10'},
                        'spec': {'template': {'spec': {'initContainers': [{'command': ['sh', '-c', COMMAND]}]}}}}

    def request(self, method, url, **kwargs):
        from urllib3.response import HTTPResponse
        self.calls.append((method, url, kwargs))
        if method == 'PATCH':
            expected_raw = html_json(json.dumps(COMMAND))
            status = 200 if expected_raw in kwargs['body'] else 422
        else:
            status = 200
        data = self.current if status == 200 else {'kind': 'Status', 'code': 422, 'message': 'test failed'}
        return HTTPResponse(status=status, body=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})


def sdk_api():
    # The SDK is a controller dependency, not an application dependency. CI's
    # Python 3.12 job installs its existing lock; other app-only jobs may skip.
    sdk = pytest.importorskip('kubernetes', reason='requires the pinned release-controller SDK lock')
    inner = RawStringTestPool()
    real_rest = sdk.client.rest.RESTClientObject
    def load_configuration(*, client_configuration, **kwargs):
        client_configuration.host = 'https://fixture.invalid'
        client_configuration.verify_ssl = True
        client_configuration.ssl_ca_cert = 'synthetic-ca.pem'
        client_configuration.api_key['authorization'] = 'synthetic-fixture'
        client_configuration.api_key_prefix['authorization'] = 'Bearer'
    def rest(configuration, *args, **kwargs):
        value = real_rest(configuration, *args, **kwargs)
        # Only the wire peer is simulated; serialization/auth/timeouts are SDK code.
        assert value.pool_manager.connection_pool_kw['cert_reqs'] == ssl.CERT_REQUIRED
        value.pool_manager = inner
        return value
    with patch.object(sdk.config, 'load_kube_config', side_effect=load_configuration), \
            patch.object(sdk.client.rest, 'RESTClientObject', side_effect=rest):
        api = kube.KubeAPI(context='synthetic', kubeconfig='synthetic-kubeconfig')
    return api, inner, sdk


def test_actual_sdk_preserves_whole_template_cas_auth_tls_and_timeout():
    api, inner, _ = sdk_api()
    operation = {'op': 'test', 'path': '/spec/template', 'value': copy.deepcopy(inner.current['spec']['template'])}
    result = api.patch('deployment', 'lab', inner.current, [operation])
    assert result == inner.current
    method, url, kwargs = inner.calls[-1]
    assert method == 'PATCH' and url == 'https://fixture.invalid/apis/apps/v1/namespaces/lab/deployments/lb'
    operations = json.loads(kwargs['body'])
    assert operations == [
        {'op': 'test', 'path': '/metadata/uid', 'value': 'u'},
        {'op': 'test', 'path': '/metadata/resourceVersion', 'value': '10'}, operation]
    assert kwargs['headers']['authorization'] == 'Bearer synthetic-fixture'
    assert kwargs['headers']['Content-Type'] == 'application/json-patch+json'
    assert kwargs['timeout'].connect_timeout == 3 and kwargs['timeout'].read_timeout == 8
    assert api.client.configuration.verify_ssl is True
    assert api.client.configuration.ssl_ca_cert == 'synthetic-ca.pem'
    assert api.client.rest_client.pool_manager._pool is inner
    assert api.client.rest_client.pool_manager.connection_pool_kw is inner.connection_pool_kw
    api.client.close()


def test_sdk_wrapper_is_per_client_and_does_not_patch_global_json_encoder():
    api, inner, sdk = sdk_api()
    ordinary = sdk.client.ApiClient(sdk.client.Configuration())
    try:
        assert isinstance(api.client.rest_client.pool_manager, kube._JSONPatchPool)
        assert not isinstance(ordinary.rest_client.pool_manager, kube._JSONPatchPool)
        assert json.dumps('<>&') == '"<>&"'
        api.get('deployment', 'lab', 'lb')
        assert inner.calls[-1][0] == 'GET'
    finally:
        api.client.close(); ordinary.close()
