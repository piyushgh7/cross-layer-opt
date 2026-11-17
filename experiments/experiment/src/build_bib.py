#!/usr/bin/env python3
"""Build a traceable BibTeX file from the saved Crossref audit records."""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "evidence/crossref_records.json").read_text())


def latex_escape(value):
    value = html.unescape(str(value or ""))
    return (value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("$", r"\$"))


def key_for(doi):
    return "d_" + re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")


entries = []
map_rows = {}
for record in rows:
    m = record.get("message", record)
    doi = m.get("DOI")
    if not doi or m.get("type") != "journal-article":
        continue
    key = key_for(doi)
    map_rows[doi.lower()] = key
    authors = m.get("author", [])
    author_text = " and ".join(
        "{} {}".format(a.get("given", ""), a.get("family", "")).strip()
        for a in authors
    ) or "Unknown"
    published = m.get("published-print") or m.get("published") or m.get("issued") or {}
    date_parts = published.get("date-parts", [[""]])[0]
    year = date_parts[0] if date_parts else ""
    fields = {
        "author": author_text,
        "title": m.get("title", [""])[0],
        "journal": m.get("container-title", [""])[0],
        "year": year,
        "volume": m.get("volume"),
        "number": m.get("issue"),
        "pages": m.get("page") or m.get("article-number"),
        "doi": doi,
    }
    # Protect established acronym casing from IEEEtran's sentence casing.
    t = fields["title"]
    acronyms = ("FEMNIST", "IoT", "IEEE", "PQC", "ML-KEM", "TLS", "QUIC",
                "FL", "AI", "GDP", "GDPR", "RL", "CNN", "OpenTitan")
    replacements = {}
    for i, acronym in enumerate(acronyms):
        marker = f"ZZACRO{i}ZZ"
        t = re.sub(rf"\b{re.escape(acronym)}\b", marker, t, flags=re.I)
        replacements[marker] = "{" + acronym + "}"
    t = latex_escape(t)
    for marker, protected in replacements.items():
        t = t.replace(marker, protected)
    fields["title"] = t
    body = [f"@article{{{key},"]
    for name, value in fields.items():
        if value not in (None, ""):
            if name != "title":
                value = latex_escape(value)
            body.append(f"  {name} = {{{value}}},")
    body.append("}")
    entries.append("\n".join(body))

# Official/benchmark material is supplementary and never counted as journal literature.
entries.extend([
    """@misc{nist_fips203,
  author = {{National Institute of Standards and Technology}},
  title = {{Module-Lattice-Based Key-Encapsulation Mechanism Standard}},
  howpublished = {{FIPS PUB 203}},
  year = {2024},
  doi = {10.6028/NIST.FIPS.203},
  url = {https://csrc.nist.gov/pubs/fips/203/final},
}""",
    """@misc{nist_sd19,
  author = {{Patrick J. Grother}},
  title = {{NIST Special Database 19: Handprinted Forms and Characters Database}},
  howpublished = {{National Institute of Standards and Technology, Standard Reference Data 19, 2nd ed.}},
  year = {2016},
  doi = {10.18434/T4H01C},
  url = {https://www.nist.gov/srd/nist-special-database-19},
}""",
    r"""@misc{caldas2019leaf,
  author = {Caldas, Sebastian and Duddu, Sai Meher Karthik and Wu, Peter and Li, Tian and Kone\v{c}n\'{y}, Jakub and McMahan, H. Brendan and Smith, Virginia and Talwalkar, Ameet},
  title = {{LEAF: A Benchmark for Federated Settings}},
  howpublished = {{arXiv:1812.01097}},
  year = {2019},
  url = {https://arxiv.org/abs/1812.01097},
}""",
    """@inproceedings{yuan2024int8fl,
  author = {Yuan, Jinliang and Wang, Shangguang and Li, Hongyu and Xu, Daliang and Li, Yuanchun and Xu, Mengwei and Liu, Xuanzhe},
  title = {{Towards Energy-Efficient Federated Learning via INT8-Based Training on Mobile DSPs}},
  booktitle = {{Proceedings of the ACM Web Conference 2024}},
  pages = {2786--2794},
  year = {2024},
  doi = {10.1145/3589334.3645341},
}""",
    """@misc{htc_desire_hd_battery,
  author = {{HTC}},
  title = {{HTC Desire HD User Guide and Battery Specification}},
  year = {2010},
  note = {{1230 mAh lithium-ion battery; device guide}},
  url = {https://documents.cdn.ifixit.com/bJdrmFu1djldDfZL.pdf},
}""",
])

(ROOT / "paper/references.bib").write_text("\n\n".join(entries) + "\n")
(ROOT / "evidence/doi_to_key.json").write_text(json.dumps(map_rows, indent=2) + "\n")
print(f"Wrote {len(entries)} BibTeX records to paper/references.bib")
