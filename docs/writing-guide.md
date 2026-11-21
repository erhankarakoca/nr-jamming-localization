# Writing guide for the paper

Notes for revising the paper (Overleaf project, see [development.md](development.md)). They collect what the IEEE style
guides and well-written papers in our area do, and how the November 2026 revision applied it.
Every source below was opened, and every DOI was resolved through Crossref (18 November 2026).

## 1. Mathematics

1. **Display what the reader has to inspect; keep sizes and simple values inline.** Definitions
   that carry the argument (fractions, sums, matrices, conditions) get their own numbered line;
   $N = 6n$ or $\tau_{\max} = 3\,\mu$s stay in the sentence. *IEEE Mathematics Style Guide.*
2. **Number every display** and refer to it by its number and its role: "the threshold
   (13)", "Equation (13)" at the start of a sentence. *IEEE Editorial Style Manual; Mermin.*
3. **A display is part of a sentence.** Introduce it with words that run into it ("the residual of
   PRB $b$ is"), end it with a comma or period as the sentence requires, and continue with
   "where ...". *IEEE Editorial Style Manual.*
4. **Define every symbol at or right after its first use**, including index ranges and sets. One or
   two terms go into a "where" clause; longer definitions get their own sentence. *IEEE Author
   Center.*
5. **One short notation paragraph** at the end of the introduction for conventions used
   throughout (bold vectors and matrices, $(\cdot)^H$, $\mathcal{CN}$, submatrix notation).
   Method-specific symbols are defined where they are used. *IEEE Editorial Style Manual;
   Bertsekas.*
6. **Propositions state hypotheses and result; the proof explains the mechanism.** Write
   equality in distribution explicitly ($\overset{d}{=}$). A second proposition can say which
   covariance changes and point back to the first proof. After the proof, say what the result is
   used for and when its assumption fails. *IEEE Editorial Style Manual; Bertsekas;
   Akhlaghpasand et al. (below).*
7. **Two-column width.** A display must fit about 252 pt. Break long ones at a relation or an
   operator (`split`), or put two short equations on two lines (`gathered`).

## 2. Structure

- Introduction: the receiver's problem, the closest prior work grouped by what it outputs, the gap,
  then contributions. One verifiable claim per contribution, with parallel verbs (formulate,
  derive, compare, show).
- System model before the problem: resource grid and received signal, then the attacker, then the
  formal problem (labels, hypotheses, operating point, metrics).
- Results: per paragraph, give the condition, then the comparison, then the interpretation.
- Captions stand alone: metric, population of cells, operating point, conditions, what lines and
  bands mean. Figure captions go below figures, table captions above tables.
- Abstract: one paragraph, no equations, citations or references to tables; state the operating
  point of every number. *IEEE Author Center, "Structure your article".*

## 3. Tone

- Concrete subjects and scoped claims: name the comparison that was run instead of "the strongest
  detector", "breaks every detector", "changes nothing", "costs nothing".
- One comparison per sentence; move the remaining numbers into a table.
- No rhetorical questions; no chains of colons, semicolons and parentheses.
- Vary sentence length: a short sentence for the finding, a longer one for its conditions.
- Mark interpretation as interpretation ("presumably because", "the ablation points to").
- *IEEE Authorship Series; Halmos; Peyton Jones.*

## 4. Exemplar papers

| Paper | What to copy |
| --- | --- |
| Edfors et al., "OFDM channel estimation by singular value decomposition", IEEE TCOM 46(7), 1998, [10.1109/26.701321](https://doi.org/10.1109/26.701321) | Model and assumptions before the estimator; derivations in appendices. Closest model for introducing the delay subspace. |
| Akhlaghpasand, Razavizadeh, Björnson, Do, "Jamming detection in massive MIMO systems", IEEE WCL 7(2), 2018, [10.1109/LWC.2017.2769650](https://doi.org/10.1109/LWC.2017.2769650) | Hypotheses, projected observation, false-alarm theorem, detection theorem, analytic vs simulated curves: the sequence our residual section follows. |
| Xu, Xu, Pan, Elkashlan, "Detection of jamming attack in non-coherent massive SIMO systems", IEEE TIFS 14(9), 2019, [10.1109/TIFS.2019.2899484](https://doi.org/10.1109/TIFS.2019.2899484) | Ordinary receiver first, then how the attack changes its statistic; each display followed by its definitions. |
| Al-Naffouri et al., "On the distribution of indefinite quadratic forms in Gaussian random variables", IEEE TCOM 64(1), 2016, [10.1109/TCOMM.2015.2496592](https://doi.org/10.1109/TCOMM.2015.2496592) | Separates the statistical object, its numerical evaluation and the numerical check. |
| Soltani et al., "Deep learning-based channel estimation", IEEE Commun. Lett. 23(4), 2019, [10.1109/LCOMM.2019.2898944](https://doi.org/10.1109/LCOMM.2019.2898944) | Tensor dimensions and loss stated before training; says what each baseline knows. |
| Gao, Jin, Wen, Li, "ComNet: Combination of deep learning and expert knowledge in OFDM receivers", IEEE Commun. Lett. 22(12), 2018, [10.1109/LCOMM.2018.2877965](https://doi.org/10.1109/LCOMM.2018.2877965) | One paragraph per receiver operation, inputs and outputs stated locally. |
| Cammerer et al., "A neural receiver for 5G NR multi-user MIMO", IEEE Globecom Workshops, 2023, [10.1109/GCWKSHPS58843.2023.10464486](https://doi.org/10.1109/GCWKSHPS58843.2023.10464486) | NR terminology subsection before the method; results paragraphs name the varied parameter and the baselines first. |

## 5. References added in this revision

| Key | Where it is cited |
| --- | --- |
| `han2008ofdm` (IEEE TVT 2008) | Introduction and residual: detects and excises jammed OFDM pilots after LS estimation, the closest pilot-level precedent. |
| `kapetanovic2013detection`, `akhlaghpasand2018jamming` | Introduction: pilot-attack detection in multi-antenna systems. |
| `shahriar2015phy` (IEEE COMST 2015) | Introduction: OFDM pilot vulnerability. |
| `uvaydov2024stitching` (INFOCOM 2024) | Introduction: dense spectrum segmentation. |
| `edfors1998ofdm`, `li1998robust` (IEEE TCOM 1998) | Residual: channel subspace and the uniform delay-profile correlation. |
| `turin1960characteristic`, `alnaffouri2016distribution` | Proof of Proposition 1: quadratic forms in complex Gaussian vectors. |
| `bladt2017matrix` (Springer 2017) | Phase-type tail of the residual. |
| `poor1994introduction` (Springer 1994) | Problem statement: fixed false-alarm design. |
| `li2007robust` (IEEE JSAC 2007), `kakizaki2016suppression` (EURASIP JWCN 2016) | Receiver section: interference detection fed into decoding; interference power in the LLRs. |

`li2012robust` excises corrupted pilots and makes the LLRs robust; it is not a hard-erasure
receiver. The hard receiver in the paper is our own benchmark.

## Sources

- IEEE, *Mathematics Style Guide for Authors*:
  <https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/Editing-Mathematics.pdf>
- IEEE, *Editorial Style Manual for Authors*:
  <https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Editorial-Style-Manual-for-Authors.pdf>
- IEEE Author Center, *Write your paper* (conferences):
  <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/>
- IEEE Author Center, *Structure your article*:
  <https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/>
- N. D. Mermin, "What's wrong with these equations?", *Physics Today*, 1989:
  <https://physicstoday.aip.org/opinion/whats-wrong-with-these-equations>
- D. P. Bertsekas, *Ten simple rules for mathematical writing*:
  <https://faculty.engineering.asu.edu/bertsekas/ten-simple-rules-for-mathematical-writing/>
- P. R. Halmos, "How to write mathematics", 1970:
  <https://www.scott-nelson.net/readings/Halmos1973.pdf>
- S. Peyton Jones, *How to write a great research paper*:
  <https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/>
- IEEE Authorship Series, *How to write for technical periodicals & conferences*:
  <https://eps.ieee.org/wp-content/uploads/2025/11/mc-ieee_author_guide_interactive-new.pdf>
