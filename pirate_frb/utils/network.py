"""Small network/NIC helpers shared by the server and fake X-engine entry points."""


def extract_ip(addr):
    """Extract the IP part from an 'ip:port' string (splits on last ':').

    Raises RuntimeError if 'addr' does not contain a colon.
    """
    i = addr.rfind(':')
    if i < 0:
        raise RuntimeError(f"Expected 'ip:port' string, got {addr!r}")
    return addr[:i]


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
