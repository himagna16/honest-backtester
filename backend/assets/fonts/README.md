# Bundled fonts

Static TTFs of the same faces the web app uses, so the PNG/PDF export renders
in the product's typography (Montserrat display + Inter body) rather than
matplotlib's default DejaVu Sans — and renders identically on the Linux backend.

- Montserrat — SIL Open Font License 1.1 (github.com/JulietaUla/Montserrat)
- Inter — SIL Open Font License 1.1 (github.com/rsms/inter)

Both are free to bundle and redistribute under the OFL. Sourced as static
weights via fontsource (jsDelivr). Referenced by file path in backend/export.py,
so the family-name metadata quirk (fontsource labels them "…Thin") is irrelevant.
