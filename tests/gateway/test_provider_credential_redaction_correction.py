"""Request-scoped provider-secret redaction remains private and ends with worker lifetime."""

from contextlib import contextmanager

from agent import redact


@contextmanager
def isolated_profile(home):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_exact_arbitrary_secret_is_redacted_only_while_profile_lease_is_live(tmp_path):
    secret = "ordinary-password-without-credential-pattern"
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    with isolated_profile(home_a):
        lease = redact.register_provider_credential_redaction(secret)
        assert redact.redact_sensitive_text(f"echo {secret}", force=True) == "echo «redacted-provider-credential»"
        with isolated_profile(home_b):
            assert redact.redact_sensitive_text(f"echo {secret}", force=True) == f"echo {secret}"
        lease.release()
        assert redact.redact_sensitive_text(f"echo {secret}", force=True) == f"echo {secret}"


def test_shared_exact_secret_is_reference_counted(tmp_path):
    secret = "arbitrary-secret-that-regexes-cannot-find"
    with isolated_profile(tmp_path / "profile"):
        first = redact.register_provider_credential_redaction(secret)
        second = redact.register_provider_credential_redaction(secret)
        first.release()
        assert "«redacted-provider-credential»" in redact.redact_sensitive_text(secret, force=True)
        second.release()
        assert redact.redact_sensitive_text(secret, force=True) == secret
        second.release()  # idempotent release
        assert redact.redact_sensitive_text(secret, force=True) == secret
