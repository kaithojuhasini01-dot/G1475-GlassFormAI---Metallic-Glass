\#**Feature Generation Log**



\*\*Date:\*\* 06-10-2026

\*\*Library Used:\*\* Matminer (Magpie preset)

\*\*Input File:\*\* data/cleaned/metallic\_glass\_cleaned.csv

\*\*Output File:\*\* data/features/numerical\_features.csv



\## Results

\- Processed all 998 records successfully.

\- Original columns: 8

\- Generated columns: 140 (Composition + targets + 132 Matminer descriptors)

\- Features generated include: Mean Electronegativity, Mean Melting Temperature, Average Deviation of Valence Electrons, etc.



\## Notes

\- Used `ElementProperty.from\_preset(preset\_name="magpie")` for automated feature generation.

\- Fixed a `RuntimeError` regarding multiprocessing by wrapping the script in `if \_\_name\_\_ == "\_\_main\_\_":`.

\- The generated 140 features are close to the paper's 132 hand-crafted descriptors, providing a strong numerical base for the ML models.

