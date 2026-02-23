"""Minimal test: bind to port 55555 with Twisted (no tlspyo, no subprocess).

Run on Windows from project root:
  python docs/debug_listen_port.py

If this fails, you will see the real exception (e.g. permission, firewall).
If this succeeds, Test-NetConnection to 127.0.0.1:55555 should work;
then the issue is likely tlspyo's subprocess on Windows (now worked around
by running the relay server in a thread when TLS is disabled).
"""

import sys


def main():
    port = 55555
    try:
        from twisted.internet import reactor
        from twisted.internet.protocol import Factory, Protocol
    except ImportError as e:
        print("Twisted not installed:", e, file=sys.stderr)
        sys.exit(1)

    class Dummy(Protocol):
        pass

    print(f"Listening on 0.0.0.0:{port} (Ctrl+C to stop)...")
    reactor.listenTCP(port, Factory.forProtocol(Dummy))
    reactor.run()


if __name__ == "__main__":
    main()
