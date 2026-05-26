# Un modello di machine learning può battere il mercato delle scommesse sul tennis?
### Uno studio empirico con dati pubblici gratuiti

*Capolavoro — Liceo Scientifico — [Nome] [Anno scolastico]*

---

## Abstract

Si indaga se un modello predittivo di machine learning, addestrato esclusivamente
su dati pubblici e gratuiti, possa generare un profitto sistematico scommettendo
sulle partite di tennis ATP. Si costruisce una pipeline completa (rating ELO
dinamico, feature anti-leakage, gradient boosting calibrato, quote di chiusura come
feature) e si valuta con metodi statistici rigorosi (validazione walk-forward,
intervalli di confidenza bootstrap). Il modello raggiunge il **67,6% di accuratezza**
ed è ben calibrato, ma il **ROI** su oltre **8.000 scommesse out-of-sample** è
≈ **−1%**, statisticamente indistinguibile dal puro margine del bookmaker. Si
conclude che il mercato del tennis ATP, sulle quote sharp (Pinnacle), è **efficiente**
rispetto all'informazione contenuta nei dati pubblici: un risultato negativo ma
informativo, coerente con la teoria dei mercati efficienti.

**Parole chiave:** machine learning, mercati efficienti, ELO, XGBoost, calibrazione,
walk-forward validation, scommesse sportive.

---

## 1. Introduzione

- **Contesto**: i modelli predittivi sportivi (FiveThirtyEight, Tennis Abstract) e
  l'industria del betting quantitativo.
- **Domanda di ricerca**: con i soli dati pubblici gratuiti, un modello ML può
  predire i vincitori abbastanza bene da *battere il mercato* (non solo da indovinare)?
- **Ipotesi nulla (H0)**: nessun edge — il ROI atteso è ≤ 0 (mercato efficiente).
- **Ipotesi alternativa (H1)**: esiste un sottoinsieme di partite con ROI > 0.
- **Contributo**: distinzione netta tra *accuratezza predittiva* (il modello indovina)
  ed *edge economico* (il modello batte le quote) — concetti spesso confusi.

## 2. Background e lavori correlati

- **Sistema ELO** e la sua applicazione al tennis (specializzazione per superficie).
- **Ipotesi dei mercati efficienti** (Fama) applicata al betting: la quota di
  chiusura come "prezzo" che aggrega tutta l'informazione disponibile.
- **Calibrazione** delle probabilità vs accuratezza.
- **Data leakage** e validazione temporale come requisito metodologico.

## 3. Dati

- **Risultati**: dataset Jeff Sackmann (`tennis_atp`, GitHub), ~63.000 incontri
  2005–2026, con statistiche match-by-match (servizio, ranking, ecc.).
- **Quote**: tennis-data.co.uk, quote di chiusura Pinnacle/Bet365, ~33.000 incontri
  dal 2013 (solo ATP main tour).
- **Pre-processing**: parsing date, pulizia, ordinamento cronologico stretto.

## 4. Metodi

### 4.1 ELO dinamico
Rating generale + per superficie; K-factor decrescente col numero di match giocati.

### 4.2 Feature engineering anti-leakage
**Regola d'oro**: per ogni match le feature usano solo lo stato *prima* dell'incontro;
lo stato si aggiorna *dopo*. Feature: differenze ELO, forma recente (ultimi 10),
H2H, fatica (minuti 14 gg), statistiche di servizio rolling (1ª in %, punti vinti al
servizio, ace%, ecc.), ranking. Randomizzazione deterministica p1/p2 per evitare che
il modello impari "p1 = sempre vincitore".

### 4.3 Modello
XGBoost (gradient boosting) con **split temporale** (mai casuale): train ≤ N−3,
validation N−2, test N−1…N. **Calibrazione isotonica** sul validation set per rendere
affidabili le probabilità (necessario per stimare l'edge).

### 4.4 Closing odds come feature
La probabilità implicita pre-match (overround rimosso) della quota di chiusura viene
aggiunta come feature — pre-match, quindi senza leakage. Assegnata a p1/p2 secondo lo
swap deterministico, non secondo l'esito.

### 4.5 Valutazione dell'edge
- **Backtest value-betting**: si scommette se `prob_modello − 1/quota > soglia`
  (edge), confronto con le quote di chiusura come mercato.
- **Walk-forward validation**: il modello viene ri-addestrato su finestre temporali
  consecutive e testato sempre su dati mai visti — smaschera il cherry-picking.
- **Bootstrap**: intervallo di confidenza al 95% sul ROI (10.000 ricampionamenti).
- **Stake**: Kelly frazionario (¼) con cap, per realismo.

## 5. Risultati

### 5.1 Accuratezza e calibrazione
| Modello | Accuracy test | LogLoss | Brier |
|---|---|---|---|
| Baseline ELO | 63,4% | 0,63 | 0,221 |
| XGBoost (ELO+servizio) | 65,0% | 0,617 | 0,215 |
| + closing odds (calibrato) | **67,6%** | 0,599 | 0,207 |

Calibrazione: quando il modello dice "90%", vince ~91% delle volte (per fascia di
confidenza). La feature *closing odds* dà il maggior salto (+2 punti).

### 5.2 Edge economico (il cuore dello studio)
| Strategia | N. scommesse | ROI | IC 95% | P(ROI>0) |
|---|---|---|---|---|
| edge ≥ 0,04 (walk-forward) | 8.262 | −1,1% | [−2,8%, +0,7%] | 0,11 |
| solo "sicure" (prob ≥ 0,80) | 768 (OOS) | −4,1% | [−7,6%, −0,6%] | 0,01 |
| banda ranking 31–50 | 1.416 | −1,9% | [−6,3%, +2,6%] | 0,21 |
| Grand Slam + Masters, Hard/Grass | 2.572 | +1,0% | [−1,9%, +4,0%] | 0,75 |

**Nessuna strategia ha un IC al 95% interamente sopra zero.** Le "scommesse sicure"
(alta confidenza) sono addirittura una perdita *provata* (IC tutto negativo): hit-rate
82% ma quota media 1,21 → break-even richiede 82,6% → si perde il margine.

### 5.3 La trappola del garden-of-forking-paths
Il segmento "ranking 31–50" appariva a **+1,8%** su 95 scommesse (in-sample); esteso a
1.416 scommesse out-of-sample è regredito a **−1,9%**. Dimostrazione empirica del
perché i pattern trovati a posteriori non sopravvivono.

## 6. Discussione

- L'accuratezza (67,6%) è ottima ma **non implica profitto**: il bookmaker prezza la
  quota intorno allo stesso hit-rate del modello + il proprio margine.
- La relazione chiave: `edge = hit-rate − 1/quota`. Filtrare per hit-rate ignora la
  quota → si selezionano favoriti a quota corta → si perde il margine.
- Il risultato è coerente con l'**efficienza del mercato**: la closing line di Pinnacle
  (margine 2–3%) incorpora già tutta l'informazione dei dati pubblici.

## 7. Limiti

- Solo dati **gratuiti** e **pre-match**: nessuna quota live, nessuna informazione
  privata (infortuni, condizioni last-minute).
- Quote storiche solo ATP main tour (no Challenger, dove i mercati sono meno efficienti).
- In produzione si scommette a 24 h, non sulla closing line → edge reale ancora più basso.

## 8. Conclusioni

Con dati pubblici gratuiti è possibile costruire un predittore di tennis accurato e
ben calibrato (67,6%), ma **non** un sistema di scommesse profittevole sul mercato ATP:
il ROI è statisticamente indistinguibile dal margine del bookmaker. Il mercato è
efficiente rispetto a questa informazione. Il risultato negativo è scientificamente
informativo: quantifica *dove non c'è valore* e illustra la differenza tra predizione
e profitto, e i rischi metodologici (leakage, cherry-picking) del data mining.

**Lavori futuri**: mercati meno efficienti (in-play, WTA, totals/handicap),
integrazione di dati non pubblici, modellazione punto-per-punto (Markov).

---

## Appendice
- A1. Architettura della pipeline (diagramma).
- A2. Lista completa delle feature.
- A3. Codice: repository del progetto (moduli `tennis/`).
- A4. Riproducibilità: `python run_full_pipeline.py` + `python -m walkforward ...`.

## Riferimenti (da completare)
- Sackmann J., *Tennis databases*, GitHub.
- Fama E., *Efficient Capital Markets* (1970).
- Chen & Guestrin, *XGBoost* (2016).
- Kelly J., *A New Interpretation of Information Rate* (1956).
