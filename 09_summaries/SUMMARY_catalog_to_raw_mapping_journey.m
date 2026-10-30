%{
=================================================================
SUMMARY_catalog_to_raw_mapping_journey.m
=================================================================
A summary of the attempt to connect the tabledata2 catalog entries (TLS results already fitted by the original authors)
to the raw segments (one of 639).
A pure write-up document - not executable code.


=================================================================
Part 1. Physical background
=================================================================
TLS (two-level system): a microscopic system created by defects in amorphous materials (oxide films, etc.).
It tunnels between two stable states.

Swap spectroscopy: a standard experimental technique that scans two axes, the qubit frequency (tuned by flux) and the TLS frequency
(Stark-shifted by gate voltage), and observes whether an avoided crossing
appears where the two are resonant.

Gate electrodes (alpha=ao4, beta=ao5, gamma=ao3, delta=ao6): four local voltage
application points on the chip. The ultimate goal of the whole study is to infer the TLS position by comparing how strongly
each electrode couples to the TLS (gamma, MHz/V).


=================================================================
Part 2. Data structure (as established so far)
=================================================================
Raw structure: qdats (639 segments, the 4 electrodes cycle in round-robin order:
  raw_index%4==0->ao4, 1->ao5, 2->ao3, 3->ao6); each segment is a
  21 (voltage) x 401 (flux -> frequency) 2D map (dispamp, etc.).

tabledata2 structure (13 rows x 200 columns; 19-20 entries actually filled per file):
rows 0, 2, 9, 11  = coupling strengths (gamma, in Hz/V) of the 4 electrodes
rows 1, 3, 10, 12 = originally guessed to be "symmetry-point voltages", but checking
gave physically impossible values (-151 V, +1240 V, etc.), so
that assumption is definitely wrong. Identity unknown, but since the absolute value
tends to grow as the coupling strength shrinks, they are likely
derived quantities such as "theoretical extrapolated values" (not verified, a next task).
row 4         = TLS eigenfrequency (Hz) - firmly verified (scale matches the raw dip frequency)
row 5         = presumed to be a 0/1 flag
rows 6, 7, 8  = always 0 (unused)


=================================================================
Part 3. Catalog -> raw reverse-mapping journey (v1-v4 + multi-column scan)
=================================================================

--- v1: assumed rows 1/3/10/12 were "symmetry-point voltages" and tried to match directly
    to the raw voltage axis ---
Result: failed. The target voltage -151.26 V lies outside the raw voltage range (-60 to +100 V),
so there were 0 candidates. Moreover these voltage values themselves (-151, +1240, -2585, -276)
are impossible for a cryogenic gate sweep - the assumption itself was definitely wrong.

--- Diagnosis: why rows 1/3/10/12 are not voltages ---
Checking the magnitude pattern showed that the smaller the coupling strength (gamma), the
larger the absolute value. From f=gamma*V, it may be an extrapolated value back-computed as
"the theoretical voltage needed to move this TLS to the symmetry point" (not yet verified).

--- v2: tried reverse mapping with row 4 (frequency) + main electrode (largest coupling strength) ---
Problem found: this file has a structure that "repeats the same frequency window (5.035-5.257 GHz)
while changing only the voltage slice", so the condition "is the frequency inside the window?"
did not work as a filter at all (all 160 ao4 segments passed).
In that state the quality score was computed as the "whole-map minimum", so the noisiest
background (another dark band) unrelated to the target frequency was wrongly adopted as "number one"
(raw_index=503, quality score 6.88).

--- v3: localized the quality score to "near the target frequency" + A/B/C case classification ---
Fix: compute the local quality (near the target frequency only) and the global quality (whole map)
separately, and check whether "minimum at the target point = global minimum of the map".
  Case A: the target frequency itself is the true minimum of this map (a true candidate)
  Case B: the target frequency is unremarkable and another place (e.g. 5.09 GHz) is the true minimum
  Case C: an outlier where only one pixel happens to be corrupted
The A/B separation logic was validated beforehand on synthetic data (it separated them exactly).
Real application to column 4: all of the top 10 ao4 candidates were case B (false) - that is,
raw_index=503 was not "the origin of column 4" but just a noisy background.

--- v4: extended to all 4 electrodes (did not give up because one electrode failed) ---
Reflects the lesson that "the electrode with the largest coupling strength is not necessarily the
electrode most visible in the raw map". Case A search over ao4, ao5, ao3, ao6.
Column 4 result: 0 case-A hits across the 4 electrodes and all 639 segments in total.
So column 4 never appears as the "main dip of the map" anywhere in this file's raw data -
provisionally concluded to be an entry that cannot be confirmed with single-time-point (1 segment) raw data.

--- Multi-column scan: separating whether column 4 is a special case or the method itself is the problem ---
The v4 case-A search was repeated for all 19 catalog entries.
Result: only 3 of 19 (15.8%) were confirmed as case A.
Column 17 (5.097 GHz): 597 case-A hits (close to 160 segments x 4 electrodes) - this
is the "truly dominant background dip" of this file's raw data. It matches exactly the dark horizontal band near 5.09 GHz
found by visual inspection earlier (seg12) - three different methods (visual, pixel tracking, this statistical scan)
independently reached the same conclusion.
Column 11 (5.103 GHz): 46 case-A hits - near the same band, less dominant.
Column 1 (5.073 GHz): 1 case-A hit - at the edge of the band.
The remaining 16 (including column 4): 0 case-A hits.

Decisive check: it is not only large gamma that fails (column 11 has a very large gamma=1377 MHz/V
yet has case-A hits) - what separates success from failure is not the size of the coupling strength
but whether "the target frequency happens to coincide with the strongest background band of that file".


=================================================================
Part 4. Summary of statistical methodology
=================================================================
Local vs global quality scores: an indicator based on the "overall minimum" that is not narrowed to the region of interest
can easily be fooled by other strong signals of no interest (the false number one at raw_index 503) - it must always
be compared with an indicator computed separately inside the region of interest.

A/B/C case classification: instead of judging by a single number (a score), explicitly verify with a
separate condition (target = global minimum?) "whether the score actually came from the point we want".
The classification logic itself was also validated beforehand on synthetic data (exact separation confirmed).

Systematic scan over many entries: the failure of a single entry (column 4) alone cannot distinguish "the method
is wrong" from "this entry is unusual". By repeating the same check over several entries (19), the
success/failure pattern itself is confirmed statistically - this lets the validity of the method (does it respond to real signals?)
and the characteristics of individual entries be judged separately.


=================================================================
Part 5. Final conclusions and next steps
=================================================================
In this file's (400to640) raw data, the only catalog entry directly confirmable as "the darkest dip
within a single time point (single segment)" is column 17 (5.097 GHz). Most entries (84%), including column 4,
are presumed to have been found by the original authors' own algorithm ("quin"), a far more sophisticated method
(stitching several segments together, or statistically boosting weak signals) -
a regime that our simple single-frame minimum search cannot reproduce.

Next step: for column 17, follow the same procedure as for seg12 (visual check ->
avoided-crossing 2D model fit -> MCMC). This is the first opportunity to produce a verified case in which
"the catalog and raw data actually match".

=================================================================
Part 6. Final judgment on column 17 (the last conclusion of the catalog <-> raw mapping journey)
=================================================================
Result of the precise re-verification of raw_index=503 (column 17, ao4) by continuity tracking:

  - Over all 21 voltages the dip frequency stays between 5.0955 and 5.0985 GHz
  - Standard deviation of the dip position: 0.98 MHz (effectively 0 against a window width of ~220 MHz)
  - Estimated slope from a first-order polynomial: 0.356 MHz/V
    (about 1/1300 of the catalog gamma=452 MHz/V - a completely different scale)

--- Established facts (no longer in doubt) ---
  1. Catalog column numbers and raw segment numbers/voltage rows are not directly linked
  2. All raw segments in this file share the same frequency window (5.035-5.257 GHz)
     and differ only in a 1 V span - the condition "the frequency is inside the window"
     hardly works as a filter
  3. What dominates the raw data is a horizontal band near 5.096 GHz
  4. That band is horizontal even when the voltage is changed (standard deviation of about 1 MHz) -
     clearly different from the diagonal predicted by an avoided crossing
  5. The large catalog gamma (hundreds of MHz/V) does not match the raw single-slice map
  6. Neither column 4 nor column 17 is the "darkest dip" of a raw single frame
  7. raw_index=503 is not "the origin of column 17" but merely a representative picture in which
     that horizontal band happens to appear darkest

--- Final interpretation ---
This horizontal band (about 5.096 GHz) must be seen as one of the following two hypotheses:
(Hypothesis 1) A systematic structure - a repeating artifact inherent in the hardware, of the same nature as the other
systematic errors already confirmed (3 flux-based, 5.168 GHz, etc.)
(Hypothesis 2) A genuine TLS that barely moves (very small gamma) - regardless of the catalog's
large gamma (452 MHz/V), this narrow standard deviation (1 MHz) itself
may show that "the TLS at that position hardly responds to this electrode"

At this point there is no additional evidence that decisively separates the two hypotheses. However, since a pattern
like the other systematic errors already confirmed (appearing at the same position across several electrodes/observables/voltages)
keeps being observed (always at a similar position regardless of the electrode),
a little more weight falls on Hypothesis 1 (systematic structure) - but this is not conclusive either,
and cross-validation with other electrodes/files in a later session could make it clearer.

The catalog gamma=452 MHz/V cannot be confirmed in a single raw slice, so it is most likely
a value that can only be obtained by the original authors' way of stitching several segments -
the same conclusion already reached for column 4.

[Part 6 conclusion] The attempt to map the catalog to single raw frames is honestly closed at this point.
What remains is not further raw exploration but an interpretive judgment based on the evidence already gathered,
and that judgment concludes as summarized above:
"hard to confirm, with more weight on a systematic structure".
%}
