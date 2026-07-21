# Data snapshots

The experiment scripts read these files directly from this directory. The snapshots are committed to the repository to make the paper results reproducible without external downloads. Upstream sources and checksums are recorded below for provenance.

| Local file | Data rows | Size (bytes) | SHA-256 | Upstream source |
|---|---:|---:|---|---|
| `News.csv` | 39,644 | 24,311,769 | `B66D9088632308CC27FA35AF847650D174A5A50503987C4E511DE94A99D1C218` | [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity), DOI [`10.24432/C5NS3V`](https://doi.org/10.24432/C5NS3V) |
| `housing.csv` | 20,640 | 1,423,529 | `8A3727F4CF54AC1A327F69B1D5B4DB54C5834EA81C6E4EFC0D163300022A685E` | California Housing snapshot distributed with [Hands-On Machine Learning](https://github.com/ageron/handson-ml2/tree/master/datasets/housing); see also [scikit-learn's California Housing documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| `train.csv` | 21,263 | 23,859,780 | `4DFB6E3A1F6FFD969E5A5E42F093C4800D1E2A6C8B1E309F8FCD9F23D86952F3` | [UCI Superconductivty Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data), DOI [`10.24432/C53P47`](https://doi.org/10.24432/C53P47) |
| `communities.data` | 1,994 | 1,102,815 | `09E0B5C07EAE24C1EFAB19B2EDEE05E160E7F5743B6F31E31EEC3D73624DA2EA` | [UCI Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime), DOI [`10.24432/C53W3X`](https://doi.org/10.24432/C53W3X) |

## Licensing and attribution

- Online News Popularity, Superconductivty Data, and Communities and Crime are distributed by the UCI Machine Learning Repository under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the dataset authors and UCI pages linked above.
- The `housing.csv` snapshot is redistributed from the Apache-2.0-licensed `ageron/handson-ml2` repository. The underlying observations derive from the 1990 California census housing data.

## Verify local copies

On macOS or Linux:

```sh
sha256sum data/News.csv data/housing.csv data/train.csv data/communities.data
```

On Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 data/News.csv,data/housing.csv,data/train.csv,data/communities.data
```
