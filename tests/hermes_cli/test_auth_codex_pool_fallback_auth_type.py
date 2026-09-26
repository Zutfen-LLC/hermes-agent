"""The Codex login store's pool fallback never hands out an ``api_key`` row as the OAuth credential
(ops-supervisor#216): the Codex endpoint rejects API keys, so such a row is not the subscription login."""

import json

import pytest

from hermes_cli.auth_codex import resolve_codex_runtime_credentials
from hermes_cli.auth_constants import AuthError

SVCACCT = "sk-svcacct-FIXTURE-NOT-A-REAL-KEY"


def _write_pool(rows):
    from hermes_constants import get_hermes_home
    get_hermes_home().joinpath("auth.json").write_text(json.dumps(
        {"version": 1, "credential_pool": {"openai-codex": rows}}), encoding="utf-8")


def test_api_key_rows_are_not_the_codex_login():
    _write_pool([{"id": "svc", "auth_type": "api_key", "source": "manual", "access_token": SVCACCT}])
    with pytest.raises(AuthError) as failure:
        resolve_codex_runtime_credentials()
    assert failure.value.code == "codex_auth_missing" and SVCACCT not in str(failure.value)

    _write_pool([{"id": "svc", "auth_type": "api_key", "source": "manual", "access_token": SVCACCT},
                 {"id": "dc", "auth_type": "oauth", "source": "manual:device_code", "access_token": "oauth-row"}])
    assert resolve_codex_runtime_credentials()["api_key"] == "oauth-row"
