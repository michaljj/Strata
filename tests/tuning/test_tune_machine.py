"""The machine probes: a card's compute capability and the read-only topology, PCIe and clock state.

Two rules are asserted here. Every value a probe could not read is ``None``, never ``0`` and never a guess, so the
report can say "unknown" instead of pretending. And no probe ever asks ``nvidia-smi`` to change a setting. The
commands are driven through ``tune_machine._run`` and ``subprocess.run`` with mocks, so this file needs no card.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from strata_control import autotune, tune_machine

NVIDIA_CSV = (
    "0, NVIDIA GeForce RTX 2080 Ti, 7.5, 22528, 20, 595.91.07\n"
    "1, NVIDIA GeForce RTX 2080 Ti, 7.5, 22528, 21, 595.91.07\n"
)

TOPO = (
    "\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\n"
    "GPU0\t X \tNV2\t0-27\t0\n"
    "GPU1\tNV2\t X \t0-27\t0\n"
)

#: one GPU of the plain ``nvidia-smi -q`` query: the PCIe link, the clocks and the persistence mode it carries.
MACHINE = """GPU 00000000:01:00.0
    PCI
        GPU Link Info
            PCIe Generation
                Max                       : 3
                Current                   : 3
            Link Width
                Max                       : 16x
                Current                   : 16x
    Clocks
        Graphics                          : 300 MHz
        SM                                : 300 MHz
    Max Clocks
        Graphics                          : 2100 MHz
    Performance State                     : P8
    Persistence Mode                      : Enabled
"""

#: nvidia-smi arguments that change machine state. A probe may never carry one.
SETTERS = ("-pm", "-lgc", "-rgc", "-c", "-ac", "-pl")


def _smi_run(argv: list[str], timeout: float = 10.0) -> str | None:
    """The four read-only queries of a tune report, answered from the samples above."""
    joined = " ".join(argv)
    if "--query-gpu" in joined:
        return NVIDIA_CSV
    if "topo" in argv:
        return TOPO
    if argv[-1] == "-q":
        return MACHINE
    return None


class ComputeCapabilityTest(unittest.TestCase):
    def test_parses_each_generation_to_one_integer(self):
        self.assertEqual(tune_machine.compute_cap("7.5"), 75)
        self.assertEqual(tune_machine.compute_cap("8.6"), 86)
        self.assertEqual(tune_machine.compute_cap("12.0"), 120)
        self.assertEqual(tune_machine.compute_cap(" 9.0 "), 90)

    def test_unreadable_is_none_never_zero(self):
        for bad in ("[N/A]", "", None, "7", "abc", "7.5.1", "N/A"):
            with self.subTest(bad=bad):
                self.assertIsNone(tune_machine.compute_cap(bad))


class NvidiaCardsTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(tune_machine.shutil, "which", return_value="/usr/bin/nvidia-smi")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_row_carries_compute_capability(self):
        with mock.patch.object(tune_machine, "_run", return_value=NVIDIA_CSV):
            cards = tune_machine.gpus()
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0], {"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 2080 Ti",
                                    "compute_cap": 75, "vram_total_mib": 22528, "vram_used_mib": 20,
                                    "driver": "595.91.07"})
        self.assertEqual(cards[1]["compute_cap"], 75)

    def test_missing_field_is_none_and_the_card_is_kept(self):
        with mock.patch.object(tune_machine, "_run",
                               return_value="0, NVIDIA GeForce RTX 2080 Ti, [N/A], 22528, 20, 595.91.07\n"):
            cards = tune_machine.gpus()
        self.assertEqual(len(cards), 1)
        self.assertIsNone(cards[0]["compute_cap"])

    def test_malformed_csv_drops_no_column_into_a_guess(self):
        with mock.patch.object(tune_machine, "_run", return_value="0, only, three\n\nrubbish\n"):
            self.assertEqual(tune_machine.gpus(), [])

    def test_query_asks_for_compute_capability(self):
        with mock.patch.object(tune_machine, "_run", return_value=NVIDIA_CSV) as run:
            tune_machine.gpus()
        argv = run.call_args.args[0]
        self.assertTrue(any("compute_cap" in str(a) for a in argv))


class NoNvidiaTest(unittest.TestCase):
    def test_absent_nvidia_smi_reads_no_nvidia_card(self):
        with mock.patch.object(tune_machine.shutil, "which", return_value=None), \
             mock.patch.object(tune_machine.sys, "platform", "linux"), \
             mock.patch.object(tune_machine, "Path", side_effect=_no_drm):
            self.assertEqual(tune_machine.gpus(), [])


def _no_drm(p):
    """A path factory with no /sys/class/drm tree, so the AMD branch finds nothing on any machine."""
    return pathlib.Path("/nonexistent-strata-test") / str(p).strip("/").replace("/", "_")


class AmdCardsTest(unittest.TestCase):
    def test_amd_row_is_unchanged(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        card = pathlib.Path(tmp.name) / "drm" / "card0" / "device"
        card.mkdir(parents=True)
        (card / "vendor").write_text("0x1002\n")
        (card / "mem_info_vram_total").write_text(str(24 * 1024 * tune_machine.MIB))
        (card / "mem_info_vram_used").write_text(str(2 * 1024 * tune_machine.MIB))
        (card / "product_name").write_text("Radeon AI PRO R9700\n")

        def factory(p):
            if str(p) == "/sys/class/drm":
                return pathlib.Path(tmp.name) / "drm"
            return pathlib.Path(p)

        with mock.patch.object(tune_machine.shutil, "which", return_value=None), \
             mock.patch.object(tune_machine.sys, "platform", "linux"), \
             mock.patch.object(tune_machine, "Path", side_effect=factory):
            cards = tune_machine.gpus()
        self.assertEqual(cards, [{"vendor": "amd", "index": 0, "name": "Radeon AI PRO R9700",
                                  "vram_total_mib": 24576, "vram_used_mib": 2048, "driver": None}])


class DiagnosticsTest(unittest.TestCase):
    def test_every_probe_is_read_and_reported(self):
        with mock.patch.object(tune_machine.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
             mock.patch.object(tune_machine, "_run", side_effect=_smi_run) as run:
            d = tune_machine.diagnostics()
        self.assertEqual(d["gpus"], [{"index": 0, "name": "NVIDIA GeForce RTX 2080 Ti", "compute_cap": 75},
                                     {"index": 1, "name": "NVIDIA GeForce RTX 2080 Ti", "compute_cap": 75}])
        self.assertEqual(d["topology"], [{"gpu": 0, "links": {"GPU0": "X", "GPU1": "NV2"}},
                                         {"gpu": 1, "links": {"GPU0": "NV2", "GPU1": "X"}}])
        self.assertEqual(d["pcie"][0]["pci_gpu_link_info_pcie_generation_max"], "3")
        self.assertEqual(d["pcie"][0]["pci_gpu_link_info_link_width_max"], "16x")
        self.assertEqual(d["clocks"][0]["clocks_graphics"], "300 MHz")
        self.assertEqual(d["clocks"][0]["max_clocks_graphics"], "2100 MHz")
        self.assertEqual(d["clocks"][0]["performance_state"], "P8")
        self.assertEqual(d["clocks"][0]["persistence_mode"], "Enabled")
        # the two lists stay separate: a clock is not a link field
        self.assertNotIn("clocks_graphics", d["pcie"][0])
        self.assertNotIn("pci_gpu_link_info_link_width_max", d["clocks"][0])
        for call in run.call_args_list:
            argv = [str(a) for a in call.args[0]]
            self.assertFalse([f for f in SETTERS if f in argv], f"a probe asked for a setting change: {argv}")

    def test_no_nvidia_smi_reports_every_field_unavailable(self):
        with mock.patch.object(tune_machine.shutil, "which", return_value=None), \
             mock.patch.object(tune_machine.sys, "platform", "linux"), \
             mock.patch.object(tune_machine, "Path", side_effect=_no_drm):
            d = tune_machine.diagnostics()
        self.assertEqual(d, {"gpus": None, "topology": None, "pcie": None, "clocks": None})

    def test_topology_timeout_leaves_that_field_none_and_reads_the_rest(self):
        def fake_run(argv, **kw):   # (_run passes errors=... too)
            if "topo" in argv:
                raise subprocess.TimeoutExpired(argv, timeout)
            return mock.Mock(stdout=_smi_run(argv), returncode=0)

        with mock.patch.object(tune_machine.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
             mock.patch.object(tune_machine.subprocess, "run", side_effect=fake_run):
            d = tune_machine.diagnostics()
        self.assertIsNone(d["topology"])
        self.assertEqual(d["pcie"][0]["pci_gpu_link_info_link_width_max"], "16x")
        self.assertEqual(d["clocks"][0]["performance_state"], "P8")

    def test_a_query_that_prints_nothing_is_none_not_an_empty_report(self):
        def fake_run(argv, **kw):   # (_run passes errors=... too)
            if "--query-gpu" in " ".join(argv):
                return mock.Mock(stdout=NVIDIA_CSV, returncode=0)
            if "topo" in argv:
                return mock.Mock(stdout=TOPO, returncode=0)
            return mock.Mock(stdout="", returncode=1)

        with mock.patch.object(tune_machine.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
             mock.patch.object(tune_machine.subprocess, "run", side_effect=fake_run):
            d = tune_machine.diagnostics()
        self.assertEqual(d["gpus"][0]["compute_cap"], 75)
        self.assertEqual(len(d["topology"]), 2)
        self.assertIsNone(d["pcie"])
        self.assertIsNone(d["clocks"])

    def test_nvidia_smi_without_the_compute_cap_field_still_reports_what_it_can(self):
        def fake_run(argv, **kw):   # (_run passes errors=... too)
            if "--query-gpu" in " ".join(argv):
                return mock.Mock(stdout="0, RTX 2080 Ti, [N/A], 22528, 20, 595.91.07\n", returncode=0)
            return mock.Mock(stdout=_smi_run(argv) or "", returncode=0)

        with mock.patch.object(tune_machine.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
             mock.patch.object(tune_machine.subprocess, "run", side_effect=fake_run):
            d = tune_machine.diagnostics()
        self.assertIsNone(d["gpus"][0]["compute_cap"])
        self.assertEqual(d["pcie"][0]["pci_gpu_link_info_link_width_max"], "16x")


class ResultFieldTest(unittest.TestCase):
    def test_a_tune_result_starts_with_no_machine_report(self):
        self.assertEqual(autotune.TuneResult().machine, {})

    def test_the_report_survives_to_dict(self):
        r = autotune.TuneResult()
        r.machine = {"gpus": [{"index": 0, "name": "RTX 2080 Ti", "compute_cap": 75}]}
        self.assertEqual(r.to_dict()["machine"]["gpus"][0]["compute_cap"], 75)


if __name__ == "__main__":
    unittest.main()
