#!/usr/bin/env python3
"""Audit cited references against saved Crossref records and BibTeX entries."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "paper/access.tex").read_text()
bib = (ROOT / "paper/references.bib").read_text()
records = json.loads((ROOT / "evidence/crossref_records.json").read_text())
doi_keys = json.loads((ROOT / "evidence/doi_to_key.json").read_text())

cited = set()
for group in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    cited.update(x.strip() for x in group.split(",") if x.strip())
bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
lookup = {}
for row in records:
    m = row.get("message", row)
    if m.get("DOI"):
        lookup[m["DOI"].lower()] = m

crossref_key_doi = {key: doi for doi, key in doi_keys.items()}
missing_bib = sorted(cited - bib_keys)
missing_crossref = []
non_journal = []
used_journals = []
for key in sorted(cited):
    doi = crossref_key_doi.get(key)
    if doi is None:
        if key.startswith("d_"):
            missing_crossref.append({"key": key, "reason": "no saved DOI mapping"})
        else:
            non_journal.append({"key": key, "source_type": "supplementary/non-Crossref"})
        continue
    m = lookup.get(doi.lower())
    if m is None:
        missing_crossref.append({"key": key, "doi": doi, "reason": "DOI absent from saved records"})
    elif m.get("type") == "journal-article":
        used_journals.append({
            "key": key, "doi": m.get("DOI"),
            "title": (m.get("title") or [""])[0],
            "journal": (m.get("container-title") or [""])[0],
            "year": ((m.get("published-print") or m.get("published") or m.get("issued") or {})
                     .get("date-parts", [[None]])[0][0]),
            "publisher": m.get("publisher"), "type": m.get("type"),
        })
    else:
        non_journal.append({"key": key, "doi": doi, "type": m.get("type")})

summary = {
    "journal_article_records_in_crossref_audit": sum(
        1 for x in records if x.get("message", x).get("type") == "journal-article"),
    "unique_cited_keys": len(cited),
    "unique_cited_verified_journal_dois": len({x["doi"].lower() for x in used_journals}),
    "cited_verified_journals": used_journals,
    "cited_supplementary_nonjournal_sources": non_journal,
    "missing_bib_entries": missing_bib,
    "missing_crossref_records": missing_crossref,
}
out = ROOT / "evidence/reference_audit.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
with (ROOT / "evidence/reference_audit.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["doi", "year", "journal", "title", "publisher"], delimiter="\t")
    w.writeheader()
    for item in used_journals:
        w.writerow({k: item.get(k, "") for k in ("doi", "year", "journal", "title", "publisher")})
print(json.dumps({k: v for k, v in summary.items() if k not in ("cited_verified_journals", "cited_supplementary_nonjournal_sources")}, indent=2))
if missing_bib or missing_crossref:
    raise SystemExit(1)
