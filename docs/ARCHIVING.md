# Archive a version and obtain a DOI

The code version is prepared as `1.0.0`; no DOI has been minted and no release
has been published as part of this repository edit. A GitHub commit or tag does
not by itself generate a DOI.

## Recommended: GitHub release connected to Zenodo

1. Sign in to [Zenodo](https://zenodo.org/) and link the GitHub account `NUDTPingLuo`.
2. Open the profile menu, choose **GitHub**, click **Sync now**, and enable
   `NUDTPingLuo/Fed-PDE`. Do this before publishing the release.
3. Review `CITATION.cff`, `.zenodo.json`, `LICENSE`, the author order, version
   and the final code snapshot. Both metadata files currently specify MIT and
   the five manuscript authors. No article DOI or ORCID has been invented.
4. In [GitHub Releases](https://github.com/NUDTPingLuo/Fed-PDE/releases/new), create
   tag `v1.0.0` from the verified submission commit and publish release
   **Fed-PDE v1.0.0 — manuscript code and recorded results**. Summarize the code,
   result files, reproduction commands and validation. Publishing a release
   after enabling integration triggers Zenodo's archive ingestion.
5. Wait for the Zenodo record, inspect its files and metadata, and verify that
   the record DOI resolves publicly. Cite the **specific version DOI** for the
   manuscript's software snapshot. The all-versions/concept DOI serves a different
   purpose: locating the evolving project.
6. Add the assigned version DOI to the manuscript's Code availability statement
   and bibliography. Add the DOI and archive link to the README and citation
   metadata on the development branch without changing the archived tag.

If a code change is needed after archival, release a new version rather than
moving a published tag. Keep `CITATION.cff` and `.zenodo.json` synchronized:
Zenodo gives `.zenodo.json` precedence when both files exist.

## Alternative: upload the fixed source archive directly

Upload a ZIP of the verified code snapshot to a new Zenodo record, choose
**Software**, enter the five creators, version `1.0.0`, description and MIT
license, and publish it. Record the Git commit in the description. Zenodo can
reserve a DOI in a draft, but a reserved identifier is not evidence that the
archive is publicly accessible. Avoid archiving the same version through both
routes as separate unrelated records.

## Official instructions

- [Enable GitHub integration](https://help.zenodo.org/docs/github/enable-repository/)
- [Archive a release](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [Metadata precedence](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)
- [Reserve/register a DOI](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/)

Zenodo account authorization and publication are separate from the Codex GitHub
connection. This repository contains the metadata for that step, not a claim
that it has already happened.
