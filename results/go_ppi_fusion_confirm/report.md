**Headline:** GO fusion (AZ-X, exact pruned BP terms, per-anchor z) improves within-anchor ranking, replicated: H1: replicated [HuRI: equivalent] (not supported on literature-independent HuRI labels); H2: replicated [HuRI: equivalent] (not supported on literature-independent HuRI labels); H3: replicated [HuRI: equivalent] (not supported on literature-independent HuRI labels)

Pair-specific X − S (A): H1 +0.0313 [+0.0240, +0.0391] p=0.0010 n=735 positive; H2 +0.0503 [+0.0357, +0.0656] p=0.0010 n=735 positive; H3 +0.0360 [+0.0260, +0.0465] p=0.0010 n=729 positive | leak UX − X (A): H1 -0.0000 [-0.0029, +0.0031] p=0.9295 n=735 equivalent; H2 +0.0007 [-0.0061, +0.0078] p=0.8616 n=735 equivalent; H3 -0.0009 [-0.0050, +0.0031] p=0.6397 n=729 equivalent

A: Secondary GZ-P: H1g positive, H2g positive, H3g positive, nDCG@10 positive; AZ-X − GZ-P: H3 inconclusive, nDCG@10 negative | Bclean: Secondary GZ-P: H1g positive, H2g positive, H3g positive, nDCG@10 positive; AZ-X − GZ-P: H3 positive, nDCG@10 negative

# GO/PPI fusion confirmation (AZ-X primary, GZ-P secondary): report


Pins: `results/go_ppi_fusion_confirm/pins_r3.json` sha256 `965e56f4bb062eba0346c6b9b2809248023ea34adc7a855d732235aeee7c2630`

## Prior exposure (§0.3, disclosed, not corrected)

- Cohort A test and validation anchors were all scored under fusion before (publication run, F§5); AZ-X itself was chosen on cohort A's validation file and scored once on its test file (§0.1b).
- Older 3-checkpoint fusion probes on subsamples of the same files (results/ontology_ceiling/fusion_probe_*).
- Zero-shot text-only metrics on the published files; masked-text smoke models without φ (go_withholding smoke).
- φ-only simGIC ceilings on the whole interactome (HuRI 0.548 vs Lit-BM 0.674), label-level counts, fold simulations under seed 2026 (confirmatory seed 2027), label-free φ statistics.
- The best-fusion search (208 variants, validation only) and its one test scoring; no cohort-B fold, HuRI file or train-partition anchor was read, so Bclean is unaffected.

## Final labels (trained, AZ-X)

- H1: replicated [HuRI: equivalent] (not supported on literature-independent HuRI labels) (label 7)
- H2: replicated [HuRI: equivalent] (not supported on literature-independent HuRI labels) (label 7)
- H3: replicated [HuRI: equivalent] (not supported on literature-independent HuRI labels) (label 7)

## Co-primary, zero-shot and secondary-fusion Holm triples (Δ fused − text)

| hyp | cohort | arm | Δ [95% CI] | companion | Holm | positive | harm | TOST |
|---|---|---|---|---|---|---|---|---|
| H1 | A | AZ-X | +0.0313 [+0.0240, +0.0391] p=0.0010 n=735 | 98.33% [+0.0220, +0.0412] | rejected | True | False | not equivalent |
| H2 | A | AZ-X | +0.0503 [+0.0357, +0.0656] p=0.0010 n=735 | 97.50% [+0.0331, +0.0679] | rejected | True | False | not equivalent |
| H3 | A | AZ-X | +0.0360 [+0.0260, +0.0465] p=0.0010 n=729 | 95.00% [+0.0260, +0.0465] | rejected | True | False | not equivalent |
| H1z | A | AZ-X | +0.0295 [+0.0214, +0.0376] p=0.0010 n=735 | 98.33% [+0.0195, +0.0399] | rejected | True | False | not equivalent |
| H2z | A | AZ-X | +0.0476 [+0.0319, +0.0640] p=0.0010 n=735 | 97.50% [+0.0295, +0.0666] | rejected | True | False | not equivalent |
| H3z | A | AZ-X | +0.0352 [+0.0242, +0.0464] p=0.0010 n=729 | 95.00% [+0.0242, +0.0464] | rejected | True | False | not equivalent |
| H1g | A | GZ-P | +0.0290 [+0.0212, +0.0374] p=0.0010 n=735 | 98.33% [+0.0193, +0.0394] | rejected | True | False | not equivalent |
| H2g | A | GZ-P | +0.0416 [+0.0276, +0.0568] p=0.0010 n=735 | 97.50% [+0.0252, +0.0583] | rejected | True | False | not equivalent |
| H3g | A | GZ-P | +0.0299 [+0.0192, +0.0411] p=0.0010 n=729 | 95.00% [+0.0192, +0.0411] | rejected | True | False | not equivalent |
| H1 | Bclean | AZ-X | +0.0217 [+0.0180, +0.0256] p=0.0010 n=2691 | 98.33% [+0.0172, +0.0263] | rejected | True | False | not equivalent |
| H2 | Bclean | AZ-X | +0.0397 [+0.0309, +0.0490] p=0.0010 n=2691 | 97.50% [+0.0298, +0.0500] | rejected | True | False | not equivalent |
| H3 | Bclean | AZ-X | +0.0252 [+0.0194, +0.0312] p=0.0010 n=2669 | 95.00% [+0.0194, +0.0312] | rejected | True | False | not equivalent |
| H1z | Bclean | AZ-X | +0.0221 [+0.0183, +0.0259] p=0.0010 n=2691 | 98.33% [+0.0175, +0.0271] | rejected | True | False | not equivalent |
| H2z | Bclean | AZ-X | +0.0354 [+0.0269, +0.0442] p=0.0010 n=2691 | 97.50% [+0.0258, +0.0457] | rejected | True | False | not equivalent |
| H3z | Bclean | AZ-X | +0.0249 [+0.0197, +0.0302] p=0.0010 n=2669 | 95.00% [+0.0197, +0.0302] | rejected | True | False | not equivalent |
| H1g | Bclean | GZ-P | +0.0217 [+0.0173, +0.0262] p=0.0010 n=2691 | 98.33% [+0.0162, +0.0274] | rejected | True | False | not equivalent |
| H2g | Bclean | GZ-P | +0.0249 [+0.0155, +0.0346] p=0.0010 n=2691 | 97.50% [+0.0137, +0.0359] | rejected | True | False | not equivalent |
| H3g | Bclean | GZ-P | +0.0197 [+0.0136, +0.0261] p=0.0010 n=2669 | 95.00% [+0.0136, +0.0261] | rejected | True | False | not equivalent |
| H1 | all_B | AZ-X | +0.0219 [+0.0187, +0.0251] p=0.0010 n=3861 | 98.33% [+0.0180, +0.0260] | rejected | True | False | not equivalent |
| H2 | all_B | AZ-X | +0.0392 [+0.0319, +0.0467] p=0.0010 n=3861 | 97.50% [+0.0310, +0.0480] | rejected | True | False | not equivalent |
| H3 | all_B | AZ-X | +0.0248 [+0.0202, +0.0295] p=0.0010 n=3834 | 95.00% [+0.0202, +0.0295] | rejected | True | False | not equivalent |
| H1z | all_B | AZ-X | +0.0232 [+0.0198, +0.0265] p=0.0010 n=3861 | 98.33% [+0.0191, +0.0274] | rejected | True | False | not equivalent |
| H2z | all_B | AZ-X | +0.0346 [+0.0273, +0.0419] p=0.0010 n=3861 | 97.50% [+0.0263, +0.0434] | rejected | True | False | not equivalent |
| H3z | all_B | AZ-X | +0.0261 [+0.0216, +0.0308] p=0.0010 n=3834 | 95.00% [+0.0216, +0.0308] | rejected | True | False | not equivalent |
| H1g | all_B | GZ-P | +0.0204 [+0.0167, +0.0240] p=0.0010 n=3861 | 98.33% [+0.0158, +0.0251] | rejected | True | False | not equivalent |
| H2g | all_B | GZ-P | +0.0223 [+0.0142, +0.0301] p=0.0010 n=3861 | 97.50% [+0.0129, +0.0311] | rejected | True | False | not equivalent |
| H3g | all_B | GZ-P | +0.0180 [+0.0129, +0.0232] p=0.0010 n=3834 | 95.00% [+0.0129, +0.0232] | rejected | True | False | not equivalent |

## Secondary metrics S1-S5 (AZ-X; uncorrected 95% CIs) and controls

| quantity | cohort | model | pooled | hard | easy | g1g0 | ap | rr | ndcg10 | p10 |
|---|---|---|---|---|---|---|---|---|---|---|
| AZ-X | A | trained | +0.0313 [+0.0240, +0.0391] positive | +0.0360 [+0.0260, +0.0465] positive | +0.0263 [+0.0201, +0.0334] positive | +0.0083 [+0.0064, +0.0103] positive | +0.0503 [+0.0357, +0.0656] positive | +0.0440 [+0.0285, +0.0605] positive | +0.0168 [+0.0132, +0.0205] positive | +0.0017 [+0.0011, +0.0024] positive |
| AZ-S | A | trained | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent |
| X-S | A | trained | +0.0313 [+0.0240, +0.0391] positive | +0.0360 [+0.0260, +0.0465] positive | +0.0263 [+0.0201, +0.0334] positive | +0.0083 [+0.0064, +0.0103] positive | +0.0503 [+0.0357, +0.0656] positive | +0.0440 [+0.0285, +0.0605] positive | +0.0168 [+0.0132, +0.0205] positive | +0.0017 [+0.0011, +0.0024] positive |
| C1:P-X | A | trained | -0.0126 [-0.0182, -0.0069] negative | -0.0149 [-0.0227, -0.0071] negative | -0.0096 [-0.0147, -0.0047] negative | +0.0065 [+0.0044, +0.0086] positive | -0.0210 [-0.0331, -0.0085] negative | -0.0190 [-0.0327, -0.0054] negative | -0.0003 [-0.0035, +0.0027] equivalent | -0.0007 [-0.0012, -0.0003] negative |
| C1a:P-N | A | trained | +0.0034 [+0.0019, +0.0051] positive | +0.0045 [+0.0023, +0.0070] positive | +0.0028 [+0.0012, +0.0045] positive | +0.0018 [+0.0010, +0.0028] positive | +0.0076 [+0.0042, +0.0109] positive | +0.0073 [+0.0035, +0.0109] positive | +0.0031 [+0.0020, +0.0043] positive | +0.0001 [-0.0000, +0.0004] equivalent |
| C1a:N-X | A | trained | -0.0161 [-0.0221, -0.0101] negative | -0.0194 [-0.0275, -0.0113] negative | -0.0123 [-0.0176, -0.0072] negative | +0.0047 [+0.0024, +0.0067] positive | -0.0285 [-0.0415, -0.0155] negative | -0.0263 [-0.0403, -0.0124] negative | -0.0034 [-0.0068, -0.0003] negative | -0.0008 [-0.0013, -0.0004] negative |
| C1t:P-T | A | trained | -0.0046 [-0.0071, -0.0025] negative | -0.0043 [-0.0078, -0.0012] negative | -0.0043 [-0.0067, -0.0021] negative | -0.0019 [-0.0035, -0.0005] negative | -0.0049 [-0.0100, -0.0000] negative | -0.0040 [-0.0099, +0.0013] equivalent | -0.0030 [-0.0047, -0.0014] negative | -0.0003 [-0.0006, -0.0001] negative |
| C1ii:P-X@0.4 | A | trained | -0.0053 [-0.0108, +0.0007] inconclusive | -0.0089 [-0.0169, -0.0003] negative | -0.0015 [-0.0070, +0.0040] equivalent | +0.0154 [+0.0119, +0.0188] positive | -0.0164 [-0.0296, -0.0038] negative | -0.0164 [-0.0311, -0.0022] negative | +0.0068 [+0.0029, +0.0105] positive | +0.0001 [-0.0004, +0.0005] equivalent |
| C3:UX-X | A | trained | -0.0000 [-0.0029, +0.0031] equivalent | -0.0009 [-0.0050, +0.0031] equivalent | +0.0004 [-0.0025, +0.0035] equivalent | +0.0020 [+0.0008, +0.0032] positive | +0.0007 [-0.0061, +0.0078] equivalent | -0.0002 [-0.0075, +0.0075] equivalent | +0.0018 [+0.0001, +0.0034] positive | +0.0001 [-0.0002, +0.0004] equivalent |
| AZ-Bz | A | trained | +0.0185 [+0.0147, +0.0220] positive | +0.0164 [+0.0115, +0.0212] positive | +0.0207 [+0.0170, +0.0246] positive | +0.0221 [+0.0197, +0.0244] positive | +0.0220 [+0.0150, +0.0290] positive | +0.0200 [+0.0123, +0.0276] positive | +0.0175 [+0.0153, +0.0197] positive | +0.0012 [+0.0008, +0.0017] positive |
| C5 | A | trained | +0.0273 [+0.0206, +0.0341] positive | +0.0322 [+0.0227, +0.0418] positive | +0.0221 [+0.0166, +0.0282] positive | +0.0107 [+0.0077, +0.0137] positive | +0.0475 [+0.0331, +0.0627] positive | +0.0425 [+0.0274, +0.0581] positive | +0.0168 [+0.0131, +0.0205] positive | +0.0015 [+0.0009, +0.0021] positive |
| X-C5 | A | trained | +0.0039 [+0.0013, +0.0067] positive | +0.0038 [-0.0001, +0.0077] equivalent | +0.0042 [+0.0013, +0.0072] positive | -0.0024 [-0.0048, +0.0001] equivalent | +0.0028 [-0.0031, +0.0088] equivalent | +0.0015 [-0.0055, +0.0084] equivalent | -0.0000 [-0.0021, +0.0021] equivalent | +0.0003 [-0.0000, +0.0006] equivalent |
| GZ-P | A | trained | +0.0290 [+0.0212, +0.0374] positive | +0.0299 [+0.0192, +0.0411] positive | +0.0280 [+0.0205, +0.0360] positive | +0.0223 [+0.0188, +0.0260] positive | +0.0416 [+0.0276, +0.0568] positive | +0.0359 [+0.0211, +0.0518] positive | +0.0245 [+0.0204, +0.0287] positive | +0.0018 [+0.0011, +0.0026] positive |
| AZX-GZP | A | trained | +0.0023 [-0.0039, +0.0080] equivalent | +0.0061 [-0.0026, +0.0142] inconclusive | -0.0017 [-0.0072, +0.0035] equivalent | -0.0140 [-0.0176, -0.0108] negative | +0.0087 [-0.0039, +0.0216] inconclusive | +0.0081 [-0.0064, +0.0217] inconclusive | -0.0077 [-0.0114, -0.0043] negative | -0.0001 [-0.0005, +0.0003] equivalent |
| GZ:U-P | A | trained | +0.0003 [-0.0025, +0.0034] equivalent | -0.0006 [-0.0045, +0.0029] equivalent | +0.0007 [-0.0025, +0.0042] equivalent | +0.0036 [+0.0009, +0.0062] positive | -0.0036 [-0.0101, +0.0024] equivalent | -0.0042 [-0.0112, +0.0025] equivalent | +0.0018 [-0.0004, +0.0041] equivalent | +0.0000 [-0.0002, +0.0003] equivalent |
| GZ:P-S_P | A | trained | +0.0284 [+0.0206, +0.0366] positive | +0.0294 [+0.0187, +0.0402] positive | +0.0273 [+0.0200, +0.0350] positive | +0.0215 [+0.0179, +0.0252] positive | +0.0420 [+0.0280, +0.0569] positive | +0.0367 [+0.0223, +0.0523] positive | +0.0240 [+0.0201, +0.0281] positive | +0.0018 [+0.0011, +0.0025] positive |
| AZ-X | A | zeroshot | +0.0295 [+0.0214, +0.0376] positive | +0.0352 [+0.0242, +0.0464] positive | +0.0229 [+0.0161, +0.0301] positive | +0.0050 [+0.0031, +0.0069] positive | +0.0476 [+0.0319, +0.0640] positive | +0.0425 [+0.0260, +0.0600] positive | +0.0148 [+0.0111, +0.0185] positive | +0.0020 [+0.0011, +0.0029] positive |
| AZ-S | A | zeroshot | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent |
| X-S | A | zeroshot | +0.0295 [+0.0214, +0.0376] positive | +0.0352 [+0.0242, +0.0464] positive | +0.0229 [+0.0161, +0.0301] positive | +0.0050 [+0.0031, +0.0069] positive | +0.0476 [+0.0319, +0.0640] positive | +0.0425 [+0.0260, +0.0600] positive | +0.0148 [+0.0111, +0.0185] positive | +0.0020 [+0.0011, +0.0029] positive |
| C1:P-X | A | zeroshot | -0.0032 [-0.0087, +0.0025] equivalent | -0.0070 [-0.0160, +0.0021] inconclusive | -0.0001 [-0.0058, +0.0058] equivalent | +0.0109 [+0.0077, +0.0140] positive | -0.0171 [-0.0304, -0.0040] negative | -0.0148 [-0.0299, +0.0010] inconclusive | +0.0034 [-0.0007, +0.0074] equivalent | +0.0006 [-0.0002, +0.0014] equivalent |
| C1a:P-N | A | zeroshot | +0.0029 [+0.0005, +0.0052] positive | +0.0052 [+0.0014, +0.0092] positive | +0.0010 [-0.0015, +0.0037] equivalent | +0.0021 [+0.0004, +0.0040] positive | +0.0026 [-0.0029, +0.0084] equivalent | +0.0021 [-0.0040, +0.0082] equivalent | +0.0018 [-0.0002, +0.0038] equivalent | +0.0001 [-0.0004, +0.0006] equivalent |
| C1a:N-X | A | zeroshot | -0.0061 [-0.0120, -0.0002] negative | -0.0122 [-0.0211, -0.0030] negative | -0.0011 [-0.0070, +0.0051] equivalent | +0.0088 [+0.0054, +0.0123] positive | -0.0197 [-0.0333, -0.0053] negative | -0.0169 [-0.0326, -0.0001] negative | +0.0016 [-0.0029, +0.0058] equivalent | +0.0005 [-0.0004, +0.0014] equivalent |
| C1t:P-T | A | zeroshot | +0.0009 [-0.0012, +0.0032] equivalent | +0.0036 [-0.0009, +0.0086] equivalent | +0.0009 [-0.0014, +0.0032] equivalent | +0.0022 [+0.0007, +0.0036] positive | +0.0012 [-0.0044, +0.0068] equivalent | +0.0011 [-0.0059, +0.0080] equivalent | +0.0016 [-0.0002, +0.0035] equivalent | -0.0002 [-0.0007, +0.0002] equivalent |
| C1ii:P-X@0.4 | A | zeroshot | +0.0008 [-0.0054, +0.0070] equivalent | -0.0032 [-0.0130, +0.0066] inconclusive | +0.0040 [-0.0027, +0.0108] equivalent | +0.0126 [+0.0087, +0.0167] positive | -0.0183 [-0.0322, -0.0050] negative | -0.0187 [-0.0343, -0.0032] negative | +0.0044 [-0.0001, +0.0087] equivalent | +0.0008 [-0.0000, +0.0016] equivalent |
| C3:UX-X | A | zeroshot | +0.0009 [-0.0018, +0.0041] equivalent | +0.0009 [-0.0032, +0.0054] equivalent | +0.0014 [-0.0014, +0.0043] equivalent | +0.0016 [+0.0004, +0.0028] positive | +0.0030 [-0.0047, +0.0109] equivalent | +0.0021 [-0.0069, +0.0114] equivalent | +0.0014 [-0.0004, +0.0033] equivalent | +0.0001 [-0.0001, +0.0004] equivalent |
| AZ-Bz | A | zeroshot | +0.0300 [+0.0239, +0.0363] positive | +0.0260 [+0.0168, +0.0351] positive | +0.0339 [+0.0268, +0.0410] positive | +0.0321 [+0.0275, +0.0367] positive | +0.0337 [+0.0227, +0.0446] positive | +0.0341 [+0.0222, +0.0464] positive | +0.0254 [+0.0213, +0.0297] positive | +0.0027 [+0.0017, +0.0036] positive |
| C5 | A | zeroshot | +0.0190 [+0.0125, +0.0256] positive | +0.0265 [+0.0169, +0.0367] positive | +0.0118 [+0.0063, +0.0178] positive | +0.0018 [+0.0000, +0.0036] positive | +0.0390 [+0.0238, +0.0560] positive | +0.0319 [+0.0151, +0.0497] positive | +0.0100 [+0.0067, +0.0136] positive | +0.0013 [+0.0006, +0.0021] positive |
| X-C5 | A | zeroshot | +0.0105 [+0.0072, +0.0141] positive | +0.0088 [+0.0031, +0.0147] positive | +0.0111 [+0.0073, +0.0151] positive | +0.0032 [+0.0019, +0.0045] positive | +0.0086 [-0.0016, +0.0185] inconclusive | +0.0106 [-0.0017, +0.0227] inconclusive | +0.0047 [+0.0023, +0.0071] positive | +0.0007 [+0.0002, +0.0013] positive |
| GZ-P | A | zeroshot | +0.0264 [+0.0191, +0.0339] positive | +0.0259 [+0.0162, +0.0354] positive | +0.0243 [+0.0167, +0.0324] positive | +0.0126 [+0.0101, +0.0150] positive | +0.0353 [+0.0237, +0.0475] positive | +0.0333 [+0.0202, +0.0471] positive | +0.0184 [+0.0147, +0.0223] positive | +0.0024 [+0.0014, +0.0034] positive |
| AZX-GZP | A | zeroshot | +0.0031 [-0.0020, +0.0083] equivalent | +0.0093 [+0.0012, +0.0178] positive | -0.0013 [-0.0065, +0.0036] equivalent | -0.0076 [-0.0100, -0.0051] negative | +0.0123 [-0.0013, +0.0253] inconclusive | +0.0093 [-0.0061, +0.0240] inconclusive | -0.0037 [-0.0071, -0.0001] negative | -0.0004 [-0.0011, +0.0003] equivalent |
| GZ:U-P | A | zeroshot | +0.0087 [+0.0038, +0.0141] positive | +0.0094 [+0.0014, +0.0180] positive | +0.0083 [+0.0031, +0.0140] positive | +0.0084 [+0.0051, +0.0116] positive | +0.0023 [-0.0078, +0.0130] inconclusive | -0.0012 [-0.0132, +0.0102] inconclusive | +0.0053 [+0.0018, +0.0090] positive | +0.0005 [-0.0002, +0.0012] equivalent |
| GZ:P-S_P | A | zeroshot | +0.0264 [+0.0191, +0.0339] positive | +0.0259 [+0.0162, +0.0354] positive | +0.0243 [+0.0167, +0.0324] positive | +0.0126 [+0.0101, +0.0150] positive | +0.0353 [+0.0237, +0.0475] positive | +0.0333 [+0.0202, +0.0471] positive | +0.0184 [+0.0147, +0.0223] positive | +0.0024 [+0.0014, +0.0034] positive |
| AZ-X | Bclean | trained | +0.0217 [+0.0180, +0.0256] positive | +0.0252 [+0.0194, +0.0312] positive | +0.0178 [+0.0146, +0.0210] positive | +0.0095 [+0.0083, +0.0108] positive | +0.0397 [+0.0309, +0.0490] positive | +0.0363 [+0.0267, +0.0463] positive | +0.0155 [+0.0135, +0.0175] positive | +0.0014 [+0.0010, +0.0018] positive |
| AZ-S | Bclean | trained | +0.0002 [-0.0003, +0.0008] equivalent | -0.0000 [-0.0008, +0.0008] equivalent | +0.0002 [-0.0004, +0.0007] equivalent | +0.0007 [+0.0003, +0.0010] positive | -0.0004 [-0.0016, +0.0008] equivalent | -0.0010 [-0.0024, +0.0004] equivalent | +0.0003 [-0.0002, +0.0008] equivalent | +0.0001 [-0.0001, +0.0002] equivalent |
| X-S | Bclean | trained | +0.0215 [+0.0178, +0.0254] positive | +0.0252 [+0.0196, +0.0311] positive | +0.0176 [+0.0144, +0.0208] positive | +0.0089 [+0.0076, +0.0102] positive | +0.0401 [+0.0314, +0.0493] positive | +0.0373 [+0.0278, +0.0473] positive | +0.0151 [+0.0131, +0.0173] positive | +0.0014 [+0.0010, +0.0017] positive |
| C1:P-X | Bclean | trained | +0.0004 [-0.0027, +0.0038] equivalent | -0.0031 [-0.0076, +0.0016] equivalent | +0.0052 [+0.0021, +0.0085] positive | +0.0131 [+0.0111, +0.0151] positive | -0.0097 [-0.0176, -0.0024] negative | -0.0102 [-0.0187, -0.0020] negative | +0.0058 [+0.0032, +0.0082] positive | +0.0003 [+0.0000, +0.0007] positive |
| C1a:P-N | Bclean | trained | +0.0054 [+0.0036, +0.0073] positive | +0.0062 [+0.0033, +0.0089] positive | +0.0052 [+0.0033, +0.0070] positive | +0.0045 [+0.0035, +0.0055] positive | +0.0082 [+0.0034, +0.0125] positive | +0.0064 [+0.0013, +0.0112] positive | +0.0051 [+0.0039, +0.0064] positive | +0.0004 [+0.0000, +0.0007] positive |
| C1a:N-X | Bclean | trained | -0.0050 [-0.0083, -0.0016] negative | -0.0093 [-0.0142, -0.0041] negative | -0.0000 [-0.0029, +0.0032] equivalent | +0.0086 [+0.0067, +0.0105] positive | -0.0179 [-0.0254, -0.0104] negative | -0.0166 [-0.0249, -0.0084] negative | +0.0007 [-0.0016, +0.0029] equivalent | -0.0000 [-0.0005, +0.0004] equivalent |
| C1t:P-T | Bclean | trained | +0.0009 [-0.0004, +0.0021] equivalent | +0.0014 [-0.0011, +0.0038] equivalent | +0.0005 [-0.0007, +0.0019] equivalent | +0.0034 [+0.0023, +0.0045] positive | +0.0032 [-0.0001, +0.0065] equivalent | +0.0022 [-0.0017, +0.0059] equivalent | +0.0030 [+0.0016, +0.0044] positive | +0.0002 [-0.0000, +0.0004] equivalent |
| C1ii:P-X@0.4 | Bclean | trained | +0.0001 [-0.0031, +0.0036] equivalent | -0.0042 [-0.0089, +0.0007] equivalent | +0.0056 [+0.0024, +0.0089] positive | +0.0142 [+0.0122, +0.0162] positive | -0.0130 [-0.0205, -0.0056] negative | -0.0133 [-0.0217, -0.0052] negative | +0.0058 [+0.0033, +0.0081] positive | +0.0004 [+0.0001, +0.0007] positive |
| C3:UX-X | Bclean | trained | +0.0012 [-0.0002, +0.0029] equivalent | +0.0004 [-0.0018, +0.0027] equivalent | +0.0018 [+0.0004, +0.0034] positive | +0.0024 [+0.0017, +0.0031] positive | +0.0029 [-0.0011, +0.0069] equivalent | +0.0040 [-0.0003, +0.0087] equivalent | +0.0020 [+0.0011, +0.0030] positive | +0.0000 [-0.0001, +0.0002] equivalent |
| AZ-Bz | Bclean | trained | +0.0139 [+0.0114, +0.0164] positive | +0.0108 [+0.0071, +0.0146] positive | +0.0166 [+0.0140, +0.0194] positive | +0.0256 [+0.0238, +0.0274] positive | +0.0192 [+0.0137, +0.0244] positive | +0.0191 [+0.0127, +0.0249] positive | +0.0188 [+0.0169, +0.0206] positive | +0.0010 [+0.0006, +0.0014] positive |
| C5 | Bclean | trained | +0.0173 [+0.0141, +0.0204] positive | +0.0205 [+0.0153, +0.0255] positive | +0.0133 [+0.0107, +0.0159] positive | +0.0072 [+0.0060, +0.0084] positive | +0.0321 [+0.0239, +0.0405] positive | +0.0284 [+0.0196, +0.0374] positive | +0.0120 [+0.0102, +0.0140] positive | +0.0011 [+0.0007, +0.0014] positive |
| X-C5 | Bclean | trained | +0.0045 [+0.0027, +0.0065] positive | +0.0047 [+0.0015, +0.0084] positive | +0.0045 [+0.0028, +0.0063] positive | +0.0023 [+0.0012, +0.0033] positive | +0.0075 [+0.0019, +0.0138] positive | +0.0079 [+0.0019, +0.0142] positive | +0.0034 [+0.0020, +0.0047] positive | +0.0004 [+0.0001, +0.0006] positive |
| GZ-P | Bclean | trained | +0.0217 [+0.0173, +0.0262] positive | +0.0197 [+0.0136, +0.0261] positive | +0.0246 [+0.0203, +0.0292] positive | +0.0278 [+0.0248, +0.0311] positive | +0.0249 [+0.0155, +0.0346] positive | +0.0190 [+0.0088, +0.0294] positive | +0.0230 [+0.0202, +0.0258] positive | +0.0018 [+0.0013, +0.0023] positive |
| AZX-GZP | Bclean | trained | +0.0001 [-0.0037, +0.0036] equivalent | +0.0055 [+0.0000, +0.0107] positive | -0.0069 [-0.0104, -0.0034] negative | -0.0183 [-0.0212, -0.0155] negative | +0.0147 [+0.0072, +0.0221] positive | +0.0173 [+0.0092, +0.0252] positive | -0.0076 [-0.0099, -0.0051] negative | -0.0004 [-0.0008, -0.0000] negative |
| GZ:U-P | Bclean | trained | +0.0004 [-0.0016, +0.0023] equivalent | -0.0013 [-0.0043, +0.0015] equivalent | +0.0016 [-0.0002, +0.0032] equivalent | +0.0046 [+0.0031, +0.0061] positive | -0.0003 [-0.0047, +0.0039] equivalent | +0.0005 [-0.0043, +0.0049] equivalent | +0.0027 [+0.0014, +0.0040] positive | -0.0001 [-0.0002, +0.0001] equivalent |
| GZ:P-S_P | Bclean | trained | +0.0212 [+0.0169, +0.0258] positive | +0.0199 [+0.0136, +0.0262] positive | +0.0235 [+0.0192, +0.0279] positive | +0.0262 [+0.0233, +0.0294] positive | +0.0251 [+0.0157, +0.0347] positive | +0.0199 [+0.0099, +0.0301] positive | +0.0221 [+0.0195, +0.0247] positive | +0.0018 [+0.0013, +0.0022] positive |
| AZ-X | Bclean | zeroshot | +0.0221 [+0.0183, +0.0259] positive | +0.0249 [+0.0197, +0.0302] positive | +0.0183 [+0.0148, +0.0220] positive | +0.0082 [+0.0069, +0.0095] positive | +0.0354 [+0.0269, +0.0442] positive | +0.0339 [+0.0251, +0.0439] positive | +0.0137 [+0.0117, +0.0159] positive | +0.0017 [+0.0014, +0.0022] positive |
| AZ-S | Bclean | zeroshot | +0.0002 [-0.0003, +0.0008] equivalent | -0.0000 [-0.0009, +0.0009] equivalent | +0.0003 [-0.0002, +0.0009] equivalent | +0.0007 [+0.0002, +0.0011] positive | -0.0001 [-0.0013, +0.0009] equivalent | -0.0002 [-0.0016, +0.0011] equivalent | +0.0006 [+0.0001, +0.0010] positive | +0.0001 [-0.0001, +0.0002] equivalent |
| X-S | Bclean | zeroshot | +0.0218 [+0.0181, +0.0258] positive | +0.0249 [+0.0197, +0.0304] positive | +0.0180 [+0.0145, +0.0217] positive | +0.0075 [+0.0062, +0.0089] positive | +0.0356 [+0.0272, +0.0441] positive | +0.0342 [+0.0253, +0.0441] positive | +0.0132 [+0.0111, +0.0153] positive | +0.0017 [+0.0013, +0.0021] positive |
| C1:P-X | Bclean | zeroshot | +0.0029 [-0.0006, +0.0070] equivalent | -0.0022 [-0.0071, +0.0031] equivalent | +0.0087 [+0.0051, +0.0125] positive | +0.0120 [+0.0098, +0.0141] positive | -0.0077 [-0.0148, -0.0002] negative | -0.0107 [-0.0186, -0.0027] negative | +0.0043 [+0.0021, +0.0066] positive | +0.0005 [+0.0001, +0.0010] positive |
| C1a:P-N | Bclean | zeroshot | +0.0043 [+0.0026, +0.0061] positive | +0.0052 [+0.0027, +0.0077] positive | +0.0037 [+0.0018, +0.0057] positive | +0.0036 [+0.0024, +0.0049] positive | +0.0104 [+0.0067, +0.0141] positive | +0.0092 [+0.0049, +0.0135] positive | +0.0032 [+0.0019, +0.0046] positive | -0.0000 [-0.0003, +0.0003] equivalent |
| C1a:N-X | Bclean | zeroshot | -0.0014 [-0.0051, +0.0027] equivalent | -0.0074 [-0.0125, -0.0019] negative | +0.0050 [+0.0010, +0.0089] positive | +0.0083 [+0.0060, +0.0109] positive | -0.0181 [-0.0255, -0.0107] negative | -0.0199 [-0.0282, -0.0116] negative | +0.0011 [-0.0013, +0.0034] equivalent | +0.0005 [+0.0000, +0.0010] positive |
| C1t:P-T | Bclean | zeroshot | +0.0010 [-0.0004, +0.0025] equivalent | +0.0015 [-0.0008, +0.0038] equivalent | +0.0008 [-0.0006, +0.0024] equivalent | +0.0034 [+0.0024, +0.0044] positive | +0.0028 [-0.0007, +0.0062] equivalent | +0.0011 [-0.0027, +0.0048] equivalent | +0.0027 [+0.0015, +0.0038] positive | +0.0001 [-0.0001, +0.0004] equivalent |
| C1ii:P-X@0.4 | Bclean | zeroshot | +0.0031 [-0.0004, +0.0069] equivalent | -0.0021 [-0.0067, +0.0031] equivalent | +0.0091 [+0.0057, +0.0128] positive | +0.0120 [+0.0100, +0.0141] positive | -0.0080 [-0.0147, -0.0008] negative | -0.0103 [-0.0178, -0.0029] negative | +0.0040 [+0.0019, +0.0061] positive | +0.0005 [+0.0001, +0.0010] positive |
| C3:UX-X | Bclean | zeroshot | +0.0009 [-0.0008, +0.0027] equivalent | +0.0008 [-0.0020, +0.0037] equivalent | +0.0011 [-0.0006, +0.0026] equivalent | +0.0018 [+0.0011, +0.0025] positive | +0.0009 [-0.0033, +0.0052] equivalent | +0.0018 [-0.0032, +0.0065] equivalent | +0.0012 [+0.0002, +0.0023] positive | +0.0000 [-0.0001, +0.0002] equivalent |
| AZ-Bz | Bclean | zeroshot | +0.0200 [+0.0165, +0.0235] positive | +0.0151 [+0.0101, +0.0201] positive | +0.0242 [+0.0207, +0.0280] positive | +0.0312 [+0.0285, +0.0341] positive | +0.0215 [+0.0150, +0.0276] positive | +0.0211 [+0.0139, +0.0279] positive | +0.0234 [+0.0212, +0.0257] positive | +0.0018 [+0.0012, +0.0023] positive |
| C5 | Bclean | zeroshot | +0.0171 [+0.0136, +0.0207] positive | +0.0216 [+0.0166, +0.0271] positive | +0.0122 [+0.0094, +0.0152] positive | +0.0041 [+0.0028, +0.0055] positive | +0.0300 [+0.0220, +0.0387] positive | +0.0277 [+0.0189, +0.0374] positive | +0.0097 [+0.0078, +0.0118] positive | +0.0011 [+0.0007, +0.0015] positive |
| X-C5 | Bclean | zeroshot | +0.0050 [+0.0029, +0.0071] positive | +0.0033 [-0.0001, +0.0066] equivalent | +0.0060 [+0.0038, +0.0083] positive | +0.0041 [+0.0031, +0.0052] positive | +0.0054 [-0.0004, +0.0112] inconclusive | +0.0063 [-0.0005, +0.0131] inconclusive | +0.0040 [+0.0025, +0.0055] positive | +0.0007 [+0.0003, +0.0010] positive |
| GZ-P | Bclean | zeroshot | +0.0263 [+0.0219, +0.0311] positive | +0.0243 [+0.0185, +0.0306] positive | +0.0282 [+0.0238, +0.0329] positive | +0.0211 [+0.0190, +0.0234] positive | +0.0308 [+0.0226, +0.0398] positive | +0.0258 [+0.0169, +0.0356] positive | +0.0207 [+0.0184, +0.0233] positive | +0.0024 [+0.0019, +0.0030] positive |
| AZX-GZP | Bclean | zeroshot | -0.0043 [-0.0081, -0.0007] negative | +0.0006 [-0.0047, +0.0056] equivalent | -0.0099 [-0.0139, -0.0063] negative | -0.0129 [-0.0151, -0.0108] negative | +0.0046 [-0.0026, +0.0119] inconclusive | +0.0082 [+0.0004, +0.0164] positive | -0.0070 [-0.0091, -0.0048] negative | -0.0007 [-0.0011, -0.0003] negative |
| GZ:U-P | Bclean | zeroshot | +0.0016 [-0.0006, +0.0038] equivalent | +0.0020 [-0.0013, +0.0052] equivalent | +0.0021 [-0.0001, +0.0042] equivalent | +0.0037 [+0.0025, +0.0050] positive | +0.0021 [-0.0027, +0.0068] equivalent | +0.0027 [-0.0029, +0.0081] equivalent | +0.0019 [+0.0005, +0.0034] positive | +0.0001 [-0.0002, +0.0004] equivalent |
| GZ:P-S_P | Bclean | zeroshot | +0.0252 [+0.0208, +0.0300] positive | +0.0240 [+0.0182, +0.0299] positive | +0.0273 [+0.0230, +0.0319] positive | +0.0182 [+0.0160, +0.0203] positive | +0.0322 [+0.0240, +0.0409] positive | +0.0274 [+0.0184, +0.0366] positive | +0.0200 [+0.0176, +0.0224] positive | +0.0023 [+0.0018, +0.0029] positive |
| AZ-X | huri | trained | +0.0010 [-0.0023, +0.0048] equivalent | -0.0006 [-0.0098, +0.0090] equivalent | +0.0016 [-0.0016, +0.0054] equivalent | +0.0066 [+0.0019, +0.0116] positive | -0.0011 [-0.0072, +0.0050] equivalent | -0.0014 [-0.0122, +0.0095] inconclusive | +0.0015 [-0.0031, +0.0064] equivalent | +0.0001 [-0.0012, +0.0015] equivalent |
| AZ-S | huri | trained | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent |
| X-S | huri | trained | +0.0010 [-0.0023, +0.0048] equivalent | -0.0006 [-0.0098, +0.0090] equivalent | +0.0016 [-0.0016, +0.0054] equivalent | +0.0066 [+0.0019, +0.0116] positive | -0.0011 [-0.0072, +0.0050] equivalent | -0.0014 [-0.0122, +0.0095] inconclusive | +0.0015 [-0.0031, +0.0064] equivalent | +0.0001 [-0.0012, +0.0015] equivalent |
| C1:P-X | huri | trained | +0.0046 [+0.0005, +0.0084] positive | -0.0049 [-0.0151, +0.0051] inconclusive | +0.0049 [+0.0010, +0.0088] positive | +0.0103 [+0.0052, +0.0154] positive | +0.0069 [+0.0008, +0.0131] positive | +0.0088 [-0.0023, +0.0197] inconclusive | +0.0081 [+0.0030, +0.0131] positive | +0.0010 [-0.0006, +0.0025] equivalent |
| C1a:P-N | huri | trained | +0.0009 [-0.0005, +0.0023] equivalent | -0.0019 [-0.0056, +0.0021] equivalent | +0.0011 [-0.0004, +0.0025] equivalent | +0.0034 [+0.0011, +0.0056] positive | +0.0007 [-0.0012, +0.0026] equivalent | +0.0002 [-0.0030, +0.0035] equivalent | +0.0023 [+0.0006, +0.0039] positive | +0.0007 [-0.0000, +0.0015] equivalent |
| C1a:N-X | huri | trained | +0.0037 [-0.0006, +0.0075] equivalent | -0.0031 [-0.0133, +0.0073] inconclusive | +0.0038 [-0.0005, +0.0077] equivalent | +0.0070 [+0.0017, +0.0121] positive | +0.0062 [+0.0000, +0.0125] positive | +0.0086 [-0.0025, +0.0198] inconclusive | +0.0058 [+0.0008, +0.0108] positive | +0.0002 [-0.0013, +0.0019] equivalent |
| C1t:P-T | huri | trained | -0.0002 [-0.0026, +0.0023] equivalent | -0.0018 [-0.0068, +0.0033] equivalent | -0.0002 [-0.0027, +0.0022] equivalent | -0.0006 [-0.0035, +0.0021] equivalent | -0.0001 [-0.0033, +0.0030] equivalent | -0.0009 [-0.0057, +0.0038] equivalent | -0.0005 [-0.0032, +0.0023] equivalent | -0.0000 [-0.0010, +0.0010] equivalent |
| C1ii:P-X@0.4 | huri | trained | +0.0076 [+0.0014, +0.0142] positive | -0.0116 [-0.0260, +0.0022] inconclusive | +0.0086 [+0.0023, +0.0150] positive | +0.0168 [+0.0084, +0.0255] positive | +0.0062 [-0.0021, +0.0147] inconclusive | +0.0096 [-0.0041, +0.0228] inconclusive | +0.0114 [+0.0043, +0.0185] positive | +0.0015 [-0.0008, +0.0038] equivalent |
| C3:UX-X | huri | trained | +0.0034 [+0.0007, +0.0063] positive | +0.0023 [-0.0036, +0.0084] equivalent | +0.0035 [+0.0007, +0.0066] positive | +0.0037 [+0.0009, +0.0070] positive | +0.0063 [+0.0020, +0.0112] positive | +0.0075 [+0.0013, +0.0141] positive | +0.0054 [+0.0016, +0.0096] positive | +0.0006 [-0.0002, +0.0013] equivalent |
| AZ-Bz | huri | trained | +0.0001 [-0.0035, +0.0039] equivalent | +0.0014 [-0.0078, +0.0111] equivalent | -0.0000 [-0.0036, +0.0037] equivalent | +0.0056 [-0.0004, +0.0117] inconclusive | +0.0012 [-0.0030, +0.0054] equivalent | +0.0006 [-0.0059, +0.0071] equivalent | +0.0011 [-0.0030, +0.0050] equivalent | -0.0014 [-0.0029, +0.0003] equivalent |
| C5 | huri | trained | -0.0008 [-0.0041, +0.0025] equivalent | -0.0032 [-0.0124, +0.0064] inconclusive | -0.0003 [-0.0035, +0.0031] equivalent | +0.0060 [+0.0011, +0.0111] positive | -0.0038 [-0.0098, +0.0021] equivalent | -0.0048 [-0.0150, +0.0052] inconclusive | -0.0007 [-0.0053, +0.0042] equivalent | -0.0009 [-0.0025, +0.0007] equivalent |
| X-C5 | huri | trained | +0.0018 [+0.0001, +0.0038] positive | +0.0026 [-0.0024, +0.0076] equivalent | +0.0019 [+0.0000, +0.0040] positive | +0.0007 [-0.0018, +0.0032] equivalent | +0.0027 [+0.0001, +0.0056] positive | +0.0034 [-0.0016, +0.0085] equivalent | +0.0022 [-0.0003, +0.0049] equivalent | +0.0010 [-0.0002, +0.0024] equivalent |
| GZ-P | huri | trained | +0.0054 [+0.0011, +0.0104] positive | -0.0081 [-0.0201, +0.0045] inconclusive | +0.0067 [+0.0024, +0.0116] positive | +0.0201 [+0.0136, +0.0273] positive | +0.0034 [-0.0030, +0.0100] equivalent | +0.0081 [-0.0029, +0.0196] inconclusive | +0.0091 [+0.0038, +0.0146] positive | +0.0009 [-0.0010, +0.0028] equivalent |
| AZX-GZP | huri | trained | -0.0045 [-0.0092, +0.0002] equivalent | +0.0075 [-0.0037, +0.0191] inconclusive | -0.0051 [-0.0097, -0.0004] negative | -0.0135 [-0.0200, -0.0076] negative | -0.0044 [-0.0111, +0.0018] inconclusive | -0.0095 [-0.0205, +0.0014] inconclusive | -0.0076 [-0.0129, -0.0020] negative | -0.0008 [-0.0026, +0.0011] equivalent |
| GZ:U-P | huri | trained | +0.0023 [-0.0012, +0.0062] equivalent | -0.0009 [-0.0077, +0.0059] equivalent | +0.0028 [-0.0007, +0.0068] equivalent | +0.0043 [+0.0005, +0.0085] positive | +0.0029 [-0.0014, +0.0076] equivalent | +0.0007 [-0.0053, +0.0068] equivalent | +0.0029 [-0.0012, +0.0070] equivalent | +0.0005 [-0.0007, +0.0018] equivalent |
| GZ:P-S_P | huri | trained | +0.0059 [+0.0015, +0.0110] positive | -0.0091 [-0.0216, +0.0030] inconclusive | +0.0072 [+0.0029, +0.0123] positive | +0.0207 [+0.0141, +0.0278] positive | +0.0036 [-0.0030, +0.0104] equivalent | +0.0087 [-0.0023, +0.0204] inconclusive | +0.0097 [+0.0043, +0.0154] positive | +0.0010 [-0.0009, +0.0029] equivalent |
| AZ-X | huri | zeroshot | -0.0006 [-0.0040, +0.0033] equivalent | -0.0089 [-0.0202, +0.0021] inconclusive | +0.0005 [-0.0029, +0.0045] equivalent | +0.0033 [-0.0015, +0.0083] equivalent | -0.0009 [-0.0085, +0.0063] equivalent | -0.0012 [-0.0133, +0.0105] inconclusive | -0.0011 [-0.0063, +0.0041] equivalent | -0.0012 [-0.0030, +0.0007] equivalent |
| AZ-S | huri | zeroshot | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent | +0.0000 [+0.0000, +0.0000] equivalent |
| X-S | huri | zeroshot | -0.0006 [-0.0040, +0.0033] equivalent | -0.0089 [-0.0202, +0.0021] inconclusive | +0.0005 [-0.0029, +0.0045] equivalent | +0.0033 [-0.0015, +0.0083] equivalent | -0.0009 [-0.0085, +0.0063] equivalent | -0.0012 [-0.0133, +0.0105] inconclusive | -0.0011 [-0.0063, +0.0041] equivalent | -0.0012 [-0.0030, +0.0007] equivalent |
| C1:P-X | huri | zeroshot | +0.0082 [+0.0019, +0.0143] positive | +0.0031 [-0.0154, +0.0211] inconclusive | +0.0077 [+0.0012, +0.0139] positive | +0.0133 [+0.0048, +0.0221] positive | +0.0118 [+0.0020, +0.0213] positive | +0.0150 [+0.0001, +0.0305] positive | +0.0125 [+0.0042, +0.0205] positive | +0.0014 [-0.0020, +0.0047] equivalent |
| C1a:P-N | huri | zeroshot | +0.0001 [-0.0031, +0.0033] equivalent | -0.0101 [-0.0201, -0.0014] negative | +0.0008 [-0.0024, +0.0041] equivalent | +0.0042 [+0.0005, +0.0079] positive | -0.0001 [-0.0050, +0.0049] equivalent | -0.0042 [-0.0122, +0.0036] inconclusive | +0.0013 [-0.0029, +0.0056] equivalent | -0.0007 [-0.0029, +0.0014] equivalent |
| C1a:N-X | huri | zeroshot | +0.0081 [+0.0012, +0.0145] positive | +0.0132 [-0.0057, +0.0318] inconclusive | +0.0069 [-0.0002, +0.0137] inconclusive | +0.0091 [+0.0004, +0.0180] positive | +0.0119 [+0.0024, +0.0210] positive | +0.0192 [+0.0039, +0.0351] positive | +0.0112 [+0.0030, +0.0190] positive | +0.0021 [-0.0013, +0.0057] equivalent |
| C1t:P-T | huri | zeroshot | +0.0042 [+0.0010, +0.0076] positive | -0.0054 [-0.0148, +0.0033] inconclusive | +0.0043 [+0.0011, +0.0076] positive | +0.0022 [-0.0009, +0.0053] equivalent | +0.0007 [-0.0032, +0.0051] equivalent | -0.0004 [-0.0065, +0.0059] equivalent | +0.0037 [-0.0002, +0.0078] equivalent | +0.0022 [+0.0004, +0.0042] positive |
| C1ii:P-X@0.4 | huri | zeroshot | +0.0090 [+0.0016, +0.0167] positive | -0.0032 [-0.0221, +0.0159] inconclusive | +0.0092 [+0.0014, +0.0171] positive | +0.0143 [+0.0046, +0.0246] positive | +0.0119 [+0.0011, +0.0220] positive | +0.0180 [+0.0009, +0.0355] positive | +0.0168 [+0.0075, +0.0261] positive | +0.0026 [-0.0014, +0.0066] equivalent |
| C3:UX-X | huri | zeroshot | +0.0049 [+0.0012, +0.0094] positive | +0.0036 [-0.0046, +0.0130] inconclusive | +0.0048 [+0.0011, +0.0090] positive | +0.0028 [+0.0002, +0.0056] positive | +0.0046 [-0.0006, +0.0101] equivalent | +0.0055 [-0.0022, +0.0133] inconclusive | +0.0052 [+0.0008, +0.0097] positive | +0.0013 [-0.0001, +0.0027] equivalent |
| AZ-Bz | huri | zeroshot | +0.0018 [-0.0058, +0.0094] equivalent | -0.0001 [-0.0193, +0.0189] inconclusive | +0.0011 [-0.0065, +0.0088] equivalent | -0.0031 [-0.0141, +0.0079] inconclusive | +0.0011 [-0.0075, +0.0096] equivalent | -0.0022 [-0.0157, +0.0099] inconclusive | -0.0003 [-0.0087, +0.0081] equivalent | -0.0029 [-0.0066, +0.0012] equivalent |
| C5 | huri | zeroshot | -0.0011 [-0.0041, +0.0022] equivalent | -0.0048 [-0.0140, +0.0051] inconclusive | +0.0000 [-0.0029, +0.0034] equivalent | +0.0045 [-0.0002, +0.0092] equivalent | -0.0030 [-0.0084, +0.0027] equivalent | -0.0029 [-0.0133, +0.0078] inconclusive | +0.0006 [-0.0040, +0.0053] equivalent | +0.0009 [-0.0009, +0.0029] equivalent |
| X-C5 | huri | zeroshot | +0.0005 [-0.0016, +0.0026] equivalent | -0.0041 [-0.0142, +0.0045] inconclusive | +0.0005 [-0.0017, +0.0025] equivalent | -0.0012 [-0.0043, +0.0018] equivalent | +0.0021 [-0.0036, +0.0072] equivalent | +0.0017 [-0.0076, +0.0101] equivalent | -0.0017 [-0.0058, +0.0021] equivalent | -0.0021 [-0.0040, -0.0003] negative |
| GZ-P | huri | zeroshot | +0.0044 [+0.0005, +0.0085] positive | -0.0091 [-0.0222, +0.0035] inconclusive | +0.0048 [+0.0007, +0.0090] positive | +0.0102 [+0.0049, +0.0156] positive | +0.0053 [-0.0013, +0.0123] inconclusive | +0.0111 [+0.0001, +0.0218] positive | +0.0038 [-0.0016, +0.0092] equivalent | -0.0013 [-0.0038, +0.0012] equivalent |
| AZX-GZP | huri | zeroshot | -0.0049 [-0.0092, -0.0004] negative | +0.0002 [-0.0139, +0.0146] inconclusive | -0.0043 [-0.0088, +0.0006] equivalent | -0.0069 [-0.0123, -0.0014] negative | -0.0062 [-0.0135, +0.0008] inconclusive | -0.0123 [-0.0248, +0.0000] inconclusive | -0.0049 [-0.0110, +0.0010] equivalent | +0.0001 [-0.0026, +0.0029] equivalent |
| GZ:U-P | huri | zeroshot | +0.0043 [-0.0026, +0.0114] inconclusive | -0.0008 [-0.0134, +0.0124] inconclusive | +0.0050 [-0.0017, +0.0124] inconclusive | +0.0083 [+0.0011, +0.0158] positive | +0.0047 [-0.0033, +0.0131] inconclusive | +0.0043 [-0.0078, +0.0179] inconclusive | +0.0075 [-0.0001, +0.0155] inconclusive | +0.0009 [-0.0027, +0.0048] equivalent |
| GZ:P-S_P | huri | zeroshot | +0.0044 [+0.0005, +0.0085] positive | -0.0091 [-0.0222, +0.0035] inconclusive | +0.0048 [+0.0007, +0.0090] positive | +0.0102 [+0.0049, +0.0156] positive | +0.0053 [-0.0013, +0.0123] inconclusive | +0.0111 [+0.0001, +0.0218] positive | +0.0038 [-0.0016, +0.0092] equivalent | -0.0013 [-0.0038, +0.0012] equivalent |

## Control states (trained)

- A: C2 {H1: pass (trials: pass), H2: pass (trials: pass), H3: pass (trials: pass)}; C5 {'H1': 'pass', 'H2': 'pass', 'H3': 'pass'}
- Bclean: C2 {H1: pass (trials: pass), H2: pass (trials: pass), H3: pass (trials: pass)}; C5 {'H1': 'pass', 'H2': 'pass', 'H3': 'pass'}
- all_B: C2 {H1: pass (trials: pass), H2: pass (trials: pass), H3: pass (trials: pass)}; C5 {'H1': 'pass', 'H2': 'pass', 'H3': 'pass'}

## Data efficiency (cohort A, AZ-X hard AUC)

- state: **inconclusive**; β_DE = -0.0054 [-0.0119, +0.0015] p=0.1129 n=729
- β_DE pooled: -0.0063 [-0.0111, -0.0019] p=0.0120 n=735
- β_DE ap: -0.0114 [-0.0208, -0.0023] p=0.0200 n=735
- β_DE gzp_hard: -0.0018 [-0.0099, +0.0060] p=0.6677 n=729
- F=10: Δ hard +0.0380 [+0.0269, +0.0495] p=0.0010 n=729; text hard 0.7780
- F=25: Δ hard +0.0382 [+0.0275, +0.0496] p=0.0010 n=729; text hard 0.7799
- F=50: Δ hard +0.0358 [+0.0245, +0.0471] p=0.0010 n=729; text hard 0.7884
- F=75: Δ hard +0.0363 [+0.0260, +0.0474] p=0.0010 n=729; text hard 0.7867
- F=100: Δ hard +0.0326 [+0.0217, +0.0433] p=0.0010 n=729; text hard 0.7962
- F10 − F100: +0.0055 [-0.0019, +0.0129] p=0.1519

## Sensitivities (none changes a label)

- all_B: exposed anchors alone (descriptive): H1 +0.0222 [+0.0166, +0.0282] p=0.0010 n=1170; H2 +0.0380 [+0.0255, +0.0506] p=0.0010 n=1170; H3 +0.0237 [+0.0153, +0.0324] p=0.0010 n=1165
- huri_grade1_unrestricted: H1 +0.0003 [-0.0029, +0.0042] p=0.8756 n=768; H2 -0.0016 [-0.0080, +0.0049] p=0.6237 n=768; H3 -0.0028 [-0.0107, +0.0057] p=0.4988 n=415
- huri_new_labels (H3): -0.0039 [-0.0121, +0.0054] p=0.3988 n=377
- fraction-stratified c* (A): H1 +0.0313 [+0.0238, +0.0390] p=0.0010 n=735; H2 +0.0503 [+0.0360, +0.0656] p=0.0010 n=735; H3 +0.0360 [+0.0258, +0.0464] p=0.0010 n=729
- exclude empty-φ anchors (A): H1 +0.0321 [+0.0246, +0.0401] p=0.0010 n=715; H2 +0.0517 [+0.0369, +0.0674] p=0.0010 n=715; H3 +0.0370 [+0.0268, +0.0478] p=0.0010 n=709
- mention: drop flagged grade-2 records (A): H1 +0.0339 [+0.0255, +0.0430] p=0.0010 n=653; H2 +0.0512 [+0.0354, +0.0678] p=0.0010 n=653; H3 +0.0393 [+0.0280, +0.0507] p=0.0010 n=647
- study-bias tertiles A: n_PMID(anchor) tertile 1 (n=453): H1 +0.0323 [+0.0181, +0.0490] p=0.0010 n=219; H2 +0.0348 [+0.0104, +0.0598] p=0.0060 n=219; H3 +0.0301 [+0.0096, +0.0527] p=0.0060 n=214
- study-bias tertiles A: n_PMID(anchor) tertile 2 (n=433): H1 +0.0348 [+0.0226, +0.0488] p=0.0010 n=210; H2 +0.0691 [+0.0424, +0.0997] p=0.0010 n=210; H3 +0.0445 [+0.0279, +0.0628] p=0.0010 n=209
- study-bias tertiles A: n_PMID(anchor) tertile 3 (n=510): H1 +0.0281 [+0.0179, +0.0388] p=0.0010 n=306; H2 +0.0485 [+0.0241, +0.0727] p=0.0010 n=306; H3 +0.0343 [+0.0205, +0.0486] p=0.0010 n=306
- study-bias tertiles A: n_PMID(anchor) top − bottom: H1 -0.0042 [-0.0240, +0.0145] p=0.6647; H2 +0.0138 [-0.0189, +0.0492] p=0.4318; H3 +0.0042 [-0.0212, +0.0295] p=0.7146
- study-bias tertiles A: STRING degree(anchor) tertile 1 (n=209): H1 +0.0608 [+0.0331, +0.0909] p=0.0010 n=99; H2 +0.0801 [+0.0411, +0.1201] p=0.0010 n=99; H3 +0.0674 [+0.0231, +0.1154] p=0.0020 n=93
- study-bias tertiles A: STRING degree(anchor) tertile 2 (n=716): H1 +0.0298 [+0.0171, +0.0432] p=0.0010 n=263; H2 +0.0444 [+0.0172, +0.0733] p=0.0010 n=263; H3 +0.0324 [+0.0158, +0.0500] p=0.0010 n=263
- study-bias tertiles A: STRING degree(anchor) tertile 3 (n=471): H1 +0.0245 [+0.0166, +0.0328] p=0.0010 n=373; H2 +0.0466 [+0.0275, +0.0661] p=0.0010 n=373; H3 +0.0308 [+0.0205, +0.0416] p=0.0010 n=373
- study-bias tertiles A: STRING degree(anchor) top − bottom: H1 -0.0364 [-0.0688, -0.0086] p=0.0080; H2 -0.0334 [-0.0781, +0.0095] p=0.1179; H3 -0.0366 [-0.0843, +0.0075] p=0.1109
- HuRI-supported-positive strata (H3, descriptive): A (n=58) H3 +0.0467 [-0.0001, +0.0973] p=0.0520 n=58; Bclean (n=209) H3 +0.0275 [+0.0084, +0.0478] p=0.0020 n=209; all_B (n=303) H3 +0.0206 [+0.0040, +0.0380] p=0.0120 n=303
- A-pub (15 published checkpoints, descriptive) AZ-X: H1 +0.0271 [+0.0200, +0.0344] p=0.0010 n=735; H2 +0.0390 [+0.0241, +0.0533] p=0.0010 n=735; H3 +0.0287 [+0.0190, +0.0384] p=0.0010 n=729
- A-pub (15 published checkpoints, descriptive) AZ-UX: H1 +0.0280 [+0.0206, +0.0356] p=0.0010 n=735; H2 +0.0404 [+0.0252, +0.0544] p=0.0010 n=735; H3 +0.0290 [+0.0189, +0.0386] p=0.0010 n=729
- A-pub (15 published checkpoints, descriptive) GZ-P: H1 +0.0249 [+0.0174, +0.0331] p=0.0010 n=735; H2 +0.0307 [+0.0160, +0.0453] p=0.0010 n=735; H3 +0.0226 [+0.0122, +0.0330] p=0.0010 n=729
- A-pub (15 published checkpoints, descriptive) GZ-U: H1 +0.0253 [+0.0177, +0.0338] p=0.0010 n=735; H2 +0.0256 [+0.0111, +0.0405] p=0.0010 n=735; H3 +0.0222 [+0.0115, +0.0334] p=0.0010 n=729
- A-pub (15 published checkpoints, descriptive) GZ-U|pooled: H1 +0.0198 [+0.0142, +0.0261] p=0.0010 n=735; H2 +0.0256 [+0.0160, +0.0357] p=0.0010 n=735; H3 +0.0195 [+0.0119, +0.0277] p=0.0010 n=729
- A-pub (15 published checkpoints, descriptive) retrained A − A-pub AZ-X: H1 +0.0042 [+0.0008, +0.0078] p=0.0230; H2 +0.0113 [+0.0046, +0.0190] p=0.0010; H3 +0.0073 [+0.0028, +0.0122] p=0.0040
- A-pub (15 published checkpoints, descriptive) retrained A − A-pub GZ-P: H1 +0.0041 [+0.0003, +0.0078] p=0.0330; H2 +0.0109 [+0.0035, +0.0184] p=0.0040; H3 +0.0074 [+0.0021, +0.0129] p=0.0050

## Pooled A + Bclean (descriptive)

- trained: H1 +0.0238 [+0.0205, +0.0272] p=0.0010 n=[735, 2691]; H2 +0.0419 [+0.0344, +0.0498] p=0.0010 n=[735, 2691]; H3 +0.0275 [+0.0225, +0.0328] p=0.0010 n=[729, 2669]
- zeroshot: H1 +0.0237 [+0.0202, +0.0273] p=0.0010 n=[735, 2691]; H2 +0.0380 [+0.0307, +0.0458] p=0.0010 n=[735, 2691]; H3 +0.0271 [+0.0224, +0.0320] p=0.0010 n=[729, 2669]

## p / CI disagreements

- none

## Near ties, exclusions and counts

- no exclusions
- near tie: A f10_s42 AZ-X|ndcg: w=0.6 top-two gap 6.28e-05
- near tie: A f10_s13 AZ-X|ndcg: w=0.6 top-two gap 3.58e-05
- near tie: A f10_s13 GZ-P: w=0.3 top-two gap 5.27e-05
- near tie: A f25_s42 AZ-P: w=0.2 top-two gap 1.60e-05
- near tie: A f25_s13 GZ-X: w=0.5 top-two gap 9.58e-05
- near tie: A f25_s21 AZ-X: w=0.4 top-two gap 6.57e-05
- near tie: A f25_s21 AZ-UX: w=0.4 top-two gap 1.48e-05
- near tie: A f25_s21 AZ-X+Bz: w=[0.3, 0.1] top-two gap 9.70e-05
- near tie: A f25_s123 GZ-U: w=0.4 top-two gap 1.51e-05
- near tie: A f50_s42 AZ-T: w=0.3 top-two gap 6.03e-05
- near tie: A f50_s42 AZ-X+Bz: w=[0.3, 0.1] top-two gap 4.14e-06
- near tie: A f50_s13 GZ-X: w=0.4 top-two gap 6.03e-05
- near tie: A f50_s21 AZ-T: w=0.3 top-two gap 9.41e-05
- near tie: A f75_s42 GZ-P|pooled: w=0.3 top-two gap 3.31e-05
- near tie: A f75_s13 AZ-UX: w=0.4 top-two gap 5.21e-05
- near tie: A f75_s13 GZ-U: w=0.5 top-two gap 1.89e-05
- near tie: A f75_s21 AZ-X|ndcg: w=0.4 top-two gap 2.85e-05
- near tie: A f75_s21 GZ-U: w=0.5 top-two gap 5.02e-05
- near tie: A f100_s42 AZ-UX: w=0.3 top-two gap 5.92e-05
- near tie: A f100_s42 GZ-P|pooled: w=0.3 top-two gap 7.75e-05
- near tie: A f100_s21 AZ-T: w=0.2 top-two gap 8.93e-05
- near tie: A f100_s21 GZ-P: w=0.3 top-two gap 4.85e-05
- near tie: A f100_s21 GZ-U: w=0.4 top-two gap 6.91e-05
- near tie: A f100_s21 GZ-S_P: w=0.1 top-two gap 6.97e-05
- near tie: A f100_s123 AZ-T: w=0.3 top-two gap 1.06e-05
- near tie: A zeroshot AZ-N: w=0.3 top-two gap 4.38e-05
- Bclean anchors (trained): 4942
- all_B anchors (trained): 7041
- cohort A anchors: 1396
- complete folds: [0, 1, 2, 3, 4]
- HuRI anchors: 768
- empty-φ anchors (A test): 42
