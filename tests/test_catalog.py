"""Fault catalog, device routing and adding a manual to the indexes (CPU, fake VLM)."""
import numpy as np

from docagent.catalog import (Catalog, Fault, build, device_name, fault_pages, ground_field, merge, parse_rows,
                              route)
from docagent.retrieve import TextIndex
from docagent.visual import VisualIndex
from docagent.vlm import GenResult
from test_cpu_parts import HashEmbedder, store  # noqa: F401

DOCS = ["schneider_atv320_programming.pdf", "abb_acs580_firmware.pdf", "grundfos_cr_io.pdf",
        "grundfos_scala2_io.pdf", "siemens_s71200_easybook.pdf", "ti_tmp117.pdf", "ti_lm35.pdf"]


class FakeVLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt, images=None, system=None, max_new_tokens=256):
        self.prompts.append(prompt)
        return GenResult(self.reply, 1.0, 10, 0.0)


def test_device_names():
    assert device_name("schneider_atv320_programming.pdf") == "Schneider ATV320"
    assert device_name("abb_acs580_firmware.pdf") == "ABB ACS580"
    assert device_name("grundfos_scala2_io.pdf") == "Grundfos SCALA2"
    assert device_name("ti_tmp117.pdf") == "TI TMP117"


def test_route_by_model_then_brand():
    assert route("ACS580 shows fault 2310", DOCS) == ["abb_acs580_firmware.pdf"]
    assert route("my acs-580 drive trips", DOCS) == ["abb_acs580_firmware.pdf"]
    assert route("S7-1200 PLC error LED", DOCS) == ["siemens_s71200_easybook.pdf"]
    assert route("TMP117 vs LM35 accuracy", DOCS) == ["ti_tmp117.pdf", "ti_lm35.pdf"]
    assert route("Grundfos pump alarm", DOCS) == ["grundfos_cr_io.pdf", "grundfos_scala2_io.pdf"]
    assert route("What does ISA-18.2 say about alarm floods?", DOCS) == []


def test_parse_rows_skips_headers_and_noise():
    reply = ("CODE | NAME | PROBABLE CAUSES | REMEDY\n"
             "OCF | Overcurrent | Ramp too short; motor too small | Check ACC; check motor\n"
             "| SCF1 | Motor short circuit | Short at output | Check cables |\n"
             "Some sentence without separators\n"
             "--- | --- | --- | ---\n"
             "This is not a code | x | y | z")
    rows = parse_rows(reply, "Schneider ATV320", "atv.pdf", 410)
    assert [r.code for r in rows] == ["OCF", "SCF1"]
    assert rows[0].causes == "Ramp too short; motor too small" and rows[1].remedy == "Check cables"
    assert parse_rows("NONE", "d", "x.pdf", 1) == []


def test_merge_keeps_longest_text_and_all_pages():
    a = Fault("D", "x.pdf", "OCF", "Overcurrent", "short", "", [410])
    b = Fault("D", "x.pdf", "ocf", "", "a much longer cause list", "Check ramp", [411])
    (m,) = merge([a, b])
    assert m.name == "Overcurrent" and m.causes == "a much longer cause list" and m.pages == [411, 410]   # page with the most text first


def test_find_codes_is_strict_about_short_codes(tmp_path):
    cat = Catalog(tmp_path / "faults.jsonl")
    cat.add([Fault("Schneider ATV320", "atv.pdf", "OCF"), Fault("Schneider ATV320", "atv.pdf", "SCF1")], "atv.pdf")
    cat.add([Fault("ABB ACS580", "abb.pdf", "2310"), Fault("ABB ACS580", "abb.pdf", "2")], "abb.pdf")
    codes = lambda q, docs=None: [f.code for f in cat.find_codes(q, docs)]  # noqa: E731
    assert codes("the drive on line 2 shows OCF") == ["OCF"]
    assert codes("drive shows scf1") == ["SCF1"]            # digit codes: any case
    assert codes("what does ocf mean") == []                # letters only: as printed
    assert codes("fault 2310 on the ACS580") == ["2310"]
    assert codes("it shows fault 2 again") == ["2"]
    assert codes("OCF and 2310", ["abb.pdf"]) == ["2310"]   # routed: other manuals ignored
    assert sorted(f.code for f in cat.search("acs580")) == ["2", "2310"]


def test_catalog_save_load_and_done(tmp_path):
    cat = Catalog(tmp_path / "faults.jsonl")
    cat.add([Fault("D", "x.pdf", "E1", "Dry run", pages=[3])], "x.pdf")
    cat.add([], "datasheet.pdf")                             # read, no codes: not read again
    cat.save()
    again = Catalog(tmp_path / "faults.jsonl")
    assert len(again) == 1 and again.done == {"x.pdf", "datasheet.pdf"} and again.faults[0].pages == [3]


def test_build_reads_only_fault_pages(store, tmp_path):  # noqa: F811
    assert fault_pages(store, "pump_A.pdf") == [2]
    vlm = FakeVLM("E-47 | Discharge pressure too high | Closed outlet valve | Open the outlet valve")
    cat = build(store, vlm, Catalog(tmp_path / "c" / "faults.jsonl"), progress=lambda m: None)
    assert len(vlm.prompts) == 1 and "manual of the Pump A" in vlm.prompts[0]
    (f,) = cat.faults
    assert (f.code, f.pages, f.remedy) == ("E-47", [2], "Open the outlet valve")
    assert (tmp_path / "c" / "faults.jsonl").exists()
    build(store, vlm, cat, progress=lambda m: None)          # already done: no new model calls
    assert len(vlm.prompts) == 1


def test_text_index_add_pages(store):  # noqa: F811
    pages = store.doc_pages("pump_A.pdf")
    idx = TextIndex.build(store, HashEmbedder())
    n = len(idx.chunks)
    for p in pages:   # same pages under a new name stand in for an uploaded manual
        p.doc_id = "pump_B.pdf"
    assert idx.add_pages(pages) > 0 and len(idx.chunks) > n
    hits = idx.search("error E-47 discharge pressure", k=1, doc_ids=["pump_B.pdf"])
    assert hits[0].doc_id == "pump_B.pdf" and hits[0].page == 2


def test_visual_index_add():
    eye = np.eye(4, dtype=np.float32)
    vi = VisualIndex([("a", 1), ("a", 2)], eye[:2], np.array([0, 1]), {"q": eye[[3]]})
    vi.add([("b", 1), ("b", 2)], [eye[[2]], eye[[3]]])
    assert vi.search("q", k=1)[0][0] == ("b", 2)
    assert vi.search("q", k=1, doc_ids=["a"])[0][0][0] == "a"


def test_grounding_drops_causes_the_page_does_not_print(store, tmp_path):  # noqa: F811
    page = "Bit 4 Overcurrent fault: Yes = drive has tripped on fault 2310 Overcurrent."
    assert ground_field("Overload; Incorrect motor sizing; Faulty motor", page) == ""
    fault_page = "2310 Overcurrent. Cause: acceleration time too short. Remedy: increase the ramp time."
    assert ground_field("Acceleration time too short; Faulty motor", fault_page) == "Acceleration time too short"
    assert ground_field("anything at all", "") == "anything at all"   # scanned page: can't check, keep
    cat = Catalog(tmp_path / "faults.jsonl")
    cat.add([Fault("Pump A", "pump_A.pdf", "E-47", "", "Closed outlet valve; Broken impeller blade",
                   "Open the outlet valve", [1, 2])], "pump_A.pdf")
    assert cat.reground(store) == 1
    (f,) = cat.faults
    assert f.causes == "Closed outlet valve" and f.pages[0] == 2


def test_parse_rows_reads_clearing_column():
    (r,) = parse_rows("CFF | Incorrect config | Option removed | Check modules | Press ENT twice", "D", "a.pdf", 405)
    assert (r.remedy, r.clearing) == ("Check modules", "Press ENT twice")


def test_tidy_moves_items_to_their_column():
    from docagent.catalog import tidy
    r = tidy(Fault("D", "a.pdf", "ANF", "Load slipping", "Speed feedback not correct; Check the motor; Add a braking resistor",
                   "This detected error requires a power reset of the device after its cause has been removed."))
    assert r.causes == "Speed feedback not correct"
    assert r.remedy == "Check the motor; Add a braking resistor"
    assert r.clearing.startswith("This detected error requires a power reset")


def test_figure_numbers_are_not_codes_and_mixed_case_codes_match_in_capitals(tmp_path):
    assert parse_rows("TM053827 | Power off | | Press the button", "G", "g.pdf", 21) == []
    cat = Catalog(tmp_path / "faults.jsonl")
    cat.add([Fault("Schneider ATV320", "atv.pdf", "phf"), Fault("Schneider ATV320", "atv.pdf", "AsF")], "atv.pdf")
    codes = lambda q: [f.code for f in cat.find_codes(q)]  # noqa: E731
    assert codes("drive shows PHF") == ["phf"] and codes("ASF on the display") == ["AsF"]
    assert codes("the asf thing") == []      # lowercase word: not trusted


def test_demo_likely_faults_dedupes_unhashable_faults():
    """The symptom path crashed on Kaggle: dict.fromkeys on Fault (an unhashable dataclass)."""
    import importlib.util
    from types import SimpleNamespace
    from pathlib import Path
    from docagent.catalog import Fault
    spec = importlib.util.spec_from_file_location("demo_app", Path(__file__).parents[1] / "demo" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    rows = [Fault("Schneider ATV320", "atv.pdf", "OCF", "Overcurrent", "too fast ramp", pages=[410]),
            Fault("Schneider ATV320", "atv.pdf", "OBF", "Overbraking", "braking too strong", pages=[411])]
    fake = SimpleNamespace(catalog=SimpleNamespace(faults=rows),
                           vlm=SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(text="OCF, ocf OBF")))
    got = app.Demo._likely_faults(fake, "The drive trips when the motor accelerates", ["atv.pdf"])
    assert [x.code for x in got] == ["OCF", "OBF"]
