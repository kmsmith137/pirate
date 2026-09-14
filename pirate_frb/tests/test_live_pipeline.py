"""CPU-only checks for the live recipe, transport source and event monitor."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .. import LivePipeline as live


def test_live_pipeline():
    recipe_path = Path(__file__).parent / "data" / "chord_8beams.yml"
    # Any Python-side attempt to persist configuration or reports is an error.
    original_open = Path.open
    def read_only(path, mode="r", *args, **kwargs):
        assert not any(c in mode for c in "wax+"), (path, mode)
        return original_open(path, mode, *args, **kwargs)
    with patch.object(Path, "open", read_only):
        recipe, bundle = live.load_live_recipe(recipe_path)
    assert bundle["beam_ids"] == recipe["observation"]["beam_ids"]
    assert "metadata_path" not in bundle
    assert "dedispersion_config_path" not in bundle
    assert bundle["nchunks"] > 0

    from ..Observation import validate_recipe
    single = deepcopy(recipe)
    single["observation"]["beam_ids"] = [1]
    single["bursts"] = [burst for burst in single["bursts"] if burst["beam_id"] == 1]
    single["dedispersion_overrides"] = dict(beams_per_batch=1, num_active_batches=1)
    single["grouper"]["execution"]["beam_batch_size"] = 1
    validate_recipe(single)
    single["dedispersion_overrides"]["num_active_batches"] = 2
    try:
        validate_recipe(single)
    except ValueError as exc:
        assert "active beam batch capacity" in str(exc)
    else:
        raise AssertionError("accepted more active batches than beam batches")

    chunk = int(np.random.randint(0, bundle["nchunks"]))
    calls = []
    class Frame:
        def __init__(self, beam):
            self.beam = beam
        def randomize_many(self, **kwargs):
            calls.append((self.beam, kwargs))
    class Frames:
        def get_frame(self, i):
            return Frame(bundle["beam_ids"][i])
        def validate(self):
            pass
    frames = Frames()
    allocator = SimpleNamespace(get_frame_set=lambda i: frames if i == chunk else None)
    def stream(data, address, **kwargs):
        assert data is bundle
        source = kwargs["frame_source_factory"](SimpleNamespace(beam_ids=bundle["beam_ids"]), allocator)
        assert source(chunk) is frames
    def pulse(burst, md):
        return SimpleNamespace(it_start=0, it_end=1, burst=burst)
    with patch("pirate_frb.Observation.stream_observation", stream), \
         patch("pirate_frb.Observation.make_pulse", pulse), \
         patch("ksgpu.seed_default_rng"), \
         patch.object(live, "atomic_print", side_effect=print), redirect_stdout(StringIO()):
        live.run_observation(recipe, bundle, "127.0.0.1:19701")
    assert len(calls) == len(bundle["beam_ids"])
    for beam, options in calls:
        assert options["normalize"] and options["gaussian"]
        assert options["dt_sp"] == chunk * bundle["samples_per_chunk"]
        assert [p.burst for p in options["pulses"]] == [b for b in recipe["bursts"] if b["beam_id"] == beam]

    from ..rpc.grpc import frb_sifter_pb2 as pb
    event = pb.FrbEvent(beam_id=7, tree_index=1, dm=100, snr=20,
                       fpga_timestamp=12500000, subband_freq_lo_MHz=500,
                       subband_freq_hi_MHz=1000)
    line = live.format_detection(event, 4e-6, 300)
    assert "TOA=50.000000 s" in line and "500.000-1000.000 MHz" in line

    class Context:
        def abort(self, code, message):
            raise ValueError(message)
    class Server:
        def add_insecure_port(self, address):
            return 19703
        def start(self):
            pass
        def stop(self, grace):
            return SimpleNamespace(wait=lambda: None)
        def wait_for_termination(self):
            context = Context()
            service.CheckConfiguration(pb.ConfigMessage(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                xengine_yaml=bundle["metadata_yaml"],
                grouper_yaml=bundle["grouper_config_yaml"]), context)
            request = pb.FrbEventsMessage(beam_set_id=bundle["metadata"]["beamset"], events=[event])
            assert service.FrbEvents(request, context).ok
            request.from_simulator = True
            try:
                service.FrbEvents(request, context)
            except ValueError:
                pass
            else:
                raise AssertionError("event_monitor accepted injected truth as a detection")
    service = None
    def register(value, server):
        nonlocal service
        service = value
    output = StringIO()
    with patch("grpc.server", lambda *a, **kw: Server()), \
         patch("pirate_frb.rpc.grpc.frb_sifter_pb2_grpc.add_FrbSifterServicer_to_server", register), \
         patch.object(Path, "open", read_only), \
         patch.object(live, "atomic_print", side_effect=print), redirect_stdout(output):
        live.run_event_monitor(bundle, "127.0.0.1:19703")
    assert output.getvalue().count("[event_monitor] beam=") == 1

    from ..OnlineGrouper import run_online_grouper
    from ..Observation import stream_observation
    for target, run in (
        ("pirate_frb.rpc.FrbGrouper",
         lambda: run_online_grouper(bundle, "unused")),
        ("pirate_frb.rpc.FrbSearchClient",
         lambda: stream_observation(bundle, "unused",
                                    frame_source_factory=lambda *a: None)),
    ):
        with patch(target, side_effect=RuntimeError("deliberate connection failure")), \
             patch.object(Path, "open", read_only):
            try:
                run()
            except RuntimeError as exc:
                assert "deliberate connection failure" in str(exc)
            else:
                raise AssertionError("connection failure did not propagate")

    # A clean native stop must return from the dedisperser loop and release resources.
    stopped = []
    poll = iter((False, True))
    server = SimpleNamespace(
        start=lambda *a, **kw: None,
        server=SimpleNamespace(poll_from_python=lambda **kw: next(poll)),
        stop=lambda: stopped.append(True))
    with patch("pirate_frb.DedispersionServer.DedispersionServer", return_value=server), \
         patch.object(live, "atomic_print", side_effect=print), redirect_stdout(StringIO()):
        live.run_dedisperser(bundle, live.addresses(19700), 0)
    assert stopped == [True]

    # The public command names must dispatch to the corresponding component.
    from argparse import ArgumentParser
    parser = ArgumentParser()
    live.add_live_parser(parser.add_subparsers(dest="command", required=True))
    commands = {
        "grouper": "pirate_frb.OnlineGrouper.run_online_grouper",
        "dedisperser": "pirate_frb.LivePipeline.run_dedisperser",
        "event_monitor": "pirate_frb.LivePipeline.run_event_monitor",
        "observation": "pirate_frb.LivePipeline.run_observation",
    }
    for role, target in commands.items():
        args = parser.parse_args(["live", role, str(recipe_path)])
        assert args.role == role and args.base_port == 19700
        with patch.object(live, "load_live_recipe", return_value=(recipe, bundle)), \
             patch(target) as component, patch.dict("os.environ"), patch.object(live, "atomic_print", side_effect=print), redirect_stdout(StringIO()):
            args.func(args)
        component.assert_called_once()
        if role == "dedisperser":
            assert component.call_args.args[-1] == 0
        elif role == "grouper":
            assert component.call_args.args[1].endswith(":19702")
            assert component.call_args.kwargs["sifter_addr"].endswith(":19703")
        elif role == "event_monitor":
            assert component.call_args.args[-1].endswith(":19703")
        else:
            assert component.call_args.args[-1].endswith(":19701")
    for obsolete in ("detector", "receiver", "injector"):
        from contextlib import redirect_stderr
        with redirect_stderr(StringIO()):
            try:
                parser.parse_args(["live", obsolete, str(recipe_path)])
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("obsolete live command remains exposed")
