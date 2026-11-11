"""Shared TLS trust: system certificates plus the public certifi roots."""
import ssl
import certifi


def tls_context():
    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    return context
