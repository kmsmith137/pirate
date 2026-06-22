"""Small network/NIC helpers shared by the server and fake X-engine entry points."""

import re
import fnmatch


def extract_ip(addr):
    """Extract the IP part from an 'ip:port' string (splits on last ':').

    Raises RuntimeError if 'addr' does not contain a colon.
    """
    i = addr.rfind(':')
    if i < 0:
        raise RuntimeError(f"Expected 'ip:port' string, got {addr!r}")
    return addr[:i]


# IPv4 octet (0-255), and a full dotted-quad matcher.
_OCTET = r'(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
_IPV4_RE = re.compile(rf'{_OCTET}(?:\.{_OCTET}){{3}}$')
_GLOB_CHARS = '*?['


def resolve_ip_spec(hw, ipspec, context=''):
    """Resolve the 'ip' part of a config 'ip:port' entry to a concrete local IPv4
    address on THIS machine. 'ipspec' may be written in any of three forms:

      - a literal IPv4 address (e.g. '10.0.0.2')    -> returned unchanged;
      - a glob (e.g. '10.0.0.*') matched against this machine's IPv4 addresses
        -> the unique matching address (it is an error to match zero or >1);
      - a network device / NIC name (e.g. 'enp13s0f0np0') -> that device's IPv4
        address.

    The glob and device-name forms let ONE config file be shared across a cluster
    of machines that have different IP addresses: each machine resolves the spec
    to its own local address. 'hw' is a pirate_frb.Hardware instance.

    'context' is prepended to every error message (e.g. the YAML field that the
    spec came from). Raises RuntimeError with a verbose explanation on failure.
    """
    if not isinstance(ipspec, str) or ipspec == '':
        raise RuntimeError(f"{context}expected a non-empty IP spec, got {ipspec!r}.")

    # (1) Glob: match against this machine's IPv4 addresses.
    if any(c in ipspec for c in _GLOB_CHARS):
        local_ips = sorted(set(hw.ip_addrs))
        matches = sorted(ip for ip in local_ips if fnmatch.fnmatch(ip, ipspec))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(
                f"{context}IP glob {ipspec!r} matched none of this machine's IPv4 "
                f"addresses {local_ips}. Check that this machine has an interface on "
                f"the expected subnet, or fix the glob.")
        raise RuntimeError(
            f"{context}IP glob {ipspec!r} is ambiguous: it matches multiple local "
            f"IPv4 addresses {matches}. Tighten the glob so it matches exactly one.")

    # (2) Literal IPv4 address: pass through (locality is checked downstream).
    if _IPV4_RE.fullmatch(ipspec):
        return ipspec

    # (2b) Looks numeric (digits/dots) but is not a valid dotted-quad.
    if re.fullmatch(r'[0-9.]+', ipspec):
        raise RuntimeError(
            f"{context}{ipspec!r} looks like an IPv4 address but is malformed "
            f"(expected four octets in 0-255, e.g. '10.0.0.2').")

    # (3) Otherwise: a network device (NIC) name.
    nics = sorted(set(hw.nics))
    if ipspec not in nics:
        listing = ', '.join(f'{n} -> {hw.ip_addr_from_nic(n)}' for n in nics)
        raise RuntimeError(
            f"{context}{ipspec!r} is not a literal IPv4 address, contains no glob "
            f"metacharacter ('*', '?', '['), and is not a network device with an "
            f"assigned IPv4 address on this machine. Devices with IPv4 addresses "
            f"are: [{listing}].")
    return hw.ip_addr_from_nic(ipspec)


def resolve_addr(hw, addr, context=''):
    """Resolve a config 'ipspec:port' entry to a concrete 'ip:port' string, where
    'ipspec' is resolved by resolve_ip_spec() (literal IPv4 / glob / device name).

    'context' is prepended to every error message. Raises RuntimeError on failure.
    """
    if not isinstance(addr, str):
        raise RuntimeError(f"{context}expected an 'ip:port' string, got {type(addr).__name__}.")
    ncolon = addr.count(':')
    if ncolon != 1:
        raise RuntimeError(
            f"{context}expected exactly one ':' separating ip and port in 'ip:port', "
            f"got {addr!r} ({ncolon} colon(s)).")
    ipspec, port = addr.split(':', 1)
    if not re.fullmatch(r'[0-9]+', port):
        raise RuntimeError(
            f"{context}port must be a positive integer, got {port!r} in {addr!r}.")
    pnum = int(port)
    if not (1 <= pnum <= 65535):
        raise RuntimeError(f"{context}port {pnum} is out of range [1, 65535] in {addr!r}.")
    ip = resolve_ip_spec(hw, ipspec, context=context)
    return f'{ip}:{pnum}'


def check_mtu(hw, label, ip_addr, min_mtu, min_mtu_param, is_dst_addr=False):
    """Raise RuntimeError if the NIC routing for 'ip_addr' has MTU below min_mtu.

    'label' is a free-form descriptor (e.g. 'FrbServer 0 data[1]') shown in
    the exception text. 'min_mtu_param' is the YAML key name (e.g.
    'min_data_mtu') so the error message points the user at the right knob.
    Set is_dst_addr=True for FakeXEngine destinations.

    Called by pirate_frb.run_server and pirate_frb.run_fake_xengine.
    """
    nic = hw.nic_from_ip_addr(ip_addr, is_dst_addr=is_dst_addr)
    mtu = hw.mtu_from_nic(nic)
    if mtu < min_mtu:
        raise RuntimeError(
            f"{label}: NIC {nic!r} ({ip_addr}) has MTU {mtu}, below the required "
            f"minimum {min_mtu} (config param {min_mtu_param!r}).\n"
            f"  - If the small MTU is intentional, lower {min_mtu_param!r} in the "
            f"server YAML config to <= {mtu}.\n"
            f"  - If the small MTU is unintentional, reconfigure the NIC to MTU "
            f">= {min_mtu} (e.g. 'sudo ip link set {nic} mtu {min_mtu}')."
        )
