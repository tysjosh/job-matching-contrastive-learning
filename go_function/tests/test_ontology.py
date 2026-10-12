import pytest

from go_function.ontology import (TRUE_PATH, DgzRule, OntologyError, ancestors, descendants,
                                  isa_siblings, logical_definition, parents, parse_obo,
                                  parse_report, two_hop)

BP, MF = "GO:0008150", "GO:0003674"


@pytest.fixture
def ont(mini_obo):
    return parse_obo(mini_obo)


@pytest.fixture
def anc(ont):
    return ancestors(ont)


def test_parse_terms_typed_edges(ont):
    assert len(ont.terms) == 12
    t = ont.terms["GO:0000012"]
    assert t.namespace == "biological_process"
    assert t.definition == "Process C, part of D."
    assert set(t.edges) == {("is_a", "GO:0000010"), ("part_of", "GO:0000013")}
    assert ont.terms["GO:0000016"].edges == (("is_a", "GO:0000014"),
                                             ("positively_regulates", "GO:0000011"))
    assert ont.terms["GO:0000016"].intersection == ((None, "GO:0000015"),
                                                    ("positively_regulates", "GO:0000011"))
    # Typedef stanzas are skipped
    assert "positively_regulates" not in ont.terms and "regulates" not in ont.terms


def test_alt_and_obsolete(ont):
    assert ont.primary("GO:0000099") == "GO:0000011"
    assert ont.primary("GO:0000011") == "GO:0000011"
    assert ont.primary("GO:0000030") is None
    assert ont.primary("GO:9999999") is None
    assert "GO:0000030" in ont.obsolete and "GO:0000030" not in ont.terms
    rep = parse_report(ont)
    assert rep["n_live"] == 12 and rep["n_obsolete"] == 1 and rep["n_alt_ids"] == 1


def test_regulates_never_propagated(anc):
    assert anc["GO:0000016"] == {"GO:0000016", "GO:0000014", "GO:0000015", BP}
    assert "GO:0000011" not in anc["GO:0000014"]


def test_part_of_within_namespace_propagated(anc):
    assert anc["GO:0000012"] == {"GO:0000012", "GO:0000010", "GO:0000013", BP}


def test_cross_namespace_part_of_not_propagated_under_tp(ont, anc):
    assert anc["GO:0000020"] == {"GO:0000020", MF}
    cross = ancestors(ont, rels=TRUE_PATH, same_namespace=False)
    assert "GO:0000013" in cross["GO:0000020"]


def test_dgz_rule_propagates_every_relation(mini_obo):
    rule = DgzRule(mini_obo)
    assert rule.closure("GO:0000020") == {"GO:0000020", MF, "GO:0000013", BP,
                                         "GO:0005515", "GO:0005488"}
    assert "GO:0000011" in rule.closure("GO:0000016")
    assert "GO:0000010" in rule.closure("GO:0000016")
    assert rule.closure("GO:0000030") == frozenset()
    assert rule.closure("GO:9999999") == frozenset()
    assert rule.prop(["GO:0000012", "GO:0000015"]) == {"GO:0000012", "GO:0000010", "GO:0000013",
                                                       "GO:0000015", BP}


def test_dgz_alt_closure_lacks_primary(mini_obo):
    rule = DgzRule(mini_obo)
    c = rule.closure("GO:0000099")
    assert "GO:0000099" in c and "GO:0000011" not in c
    assert {"GO:0000010", BP} <= c


def test_descendants_inverse(anc):
    desc = descendants(anc)
    assert desc["GO:0000013"] == {"GO:0000013", "GO:0000012"}
    assert desc["GO:0000014"] == {"GO:0000014", "GO:0000016"}
    for t, a in anc.items():
        for u in a:
            assert t in desc[u]


def test_parents(ont):
    p = parents(ont)
    assert p["GO:0000012"] == {"GO:0000010", "GO:0000013"}
    assert p["GO:0000020"] == {MF}
    assert parents(ont, rels=("regulates",))["GO:0000014"] == {"GO:0000011"}


def test_isa_siblings_exclude_anc_desc(ont, anc):
    desc = descendants(anc)
    assert isa_siblings(ont, "GO:0000011", anc, desc) == {"GO:0000012"}
    # children of the root: 10, 13, 15; 13 is excluded from its own siblings
    assert isa_siblings(ont, "GO:0000013", anc, desc) == {"GO:0000010", "GO:0000015"}


def test_two_hop(ont, anc):
    desc = descendants(anc)
    assert two_hop(ont, "GO:0000013", anc, desc) == {"GO:0000010", "GO:0000015"}
    assert two_hop(ont, "GO:0000011", anc, desc) == {"GO:0000012"}


def test_logical_definition(ont):
    assert logical_definition(ont, "GO:0000016") == (
        "GO:0000015", (("positively_regulates", "GO:0000011"),))
    assert logical_definition(ont, "GO:0000012") == (None, ())


def _write(tmp_path, body):
    p = tmp_path / "x.obo"
    p.write_text("format-version: 1.2\n\n" + body)
    return p


def test_dangling_edge_raises_with_line(tmp_path):
    p = _write(tmp_path, "[Term]\nid: GO:0000001\nname: a\nnamespace: biological_process\n"
                         "is_a: GO:0000002 ! missing\n")
    with pytest.raises(OntologyError) as e:
        parse_obo(p)
    assert e.value.line == 7 and "line 7" in str(e.value)


def test_missing_namespace_raises(tmp_path):
    p = _write(tmp_path, "[Term]\nid: GO:0000001\nname: a\n")
    with pytest.raises(OntologyError) as e:
        parse_obo(p)
    assert e.value.line == 3


def test_cycle_raises(tmp_path):
    p = _write(tmp_path,
               "[Term]\nid: GO:0000001\nname: a\nnamespace: biological_process\n"
               "is_a: GO:0000002\n\n[Term]\nid: GO:0000002\nname: b\n"
               "namespace: biological_process\npart_of_dummy: x\n"
               "relationship: part_of GO:0000001\n")
    ont = parse_obo(p)
    with pytest.raises(OntologyError):
        ancestors(ont)


def test_edge_to_obsolete_dropped_and_alt_mapped(tmp_path):
    p = _write(tmp_path,
               "[Term]\nid: GO:0000001\nname: a\nnamespace: biological_process\n"
               "alt_id: GO:0000009\n\n"
               "[Term]\nid: GO:0000002\nname: b\nnamespace: biological_process\n"
               "is_a: GO:0000003\nis_a: GO:0000009\n\n"
               "[Term]\nid: GO:0000003\nname: c\nnamespace: biological_process\n"
               "is_obsolete: true\n")
    ont = parse_obo(p)
    assert ont.terms["GO:0000002"].edges == (("is_a", "GO:0000001"),)
    assert ont.report["edges_to_obsolete_dropped"] == 1
    assert ont.report["edges_to_alt_mapped"] == 1
