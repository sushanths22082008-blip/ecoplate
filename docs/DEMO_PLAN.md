# Demo plan for the Dexathon judges (about 7 minutes)

Prepare: a clean database with **real or clearly-labelled sample data** (Dashboard → *Load sample data* marks rows as
synthetic and every screen says so). Have a JSON backup ready. Set `ECOPLATE_ACCESS_CODE` so the login screen can be shown.

1. **The problem and the guardrails (45 s).** "Canteens over-prepare; we predict demand, show what the records say,
   flag odd records and share surplus responsibly. Every number comes from the database; AI only words it."
2. **Dashboard + Forecast (90 s).** Forecast tomorrow's lunch. Point out: method label (ML vs rule-based fallback),
   recommended quantity, the likely range with how many past errors it is based on, the **Limitations** box and a
   **Shortage check** warning. Open *Model*: baselines vs Gradient Boosting, the rolling-origin table, and "not trusted
   for this meal" if it appears. Say the honest line: "ML is only used when it beats simple baselines on unseen days."
3. **Insights (60 s).** Show an *Observed* statement with period and quantities, the amber *Possible explanations (not
   established)*, and the "Not enough data yet" list. "We refuse to draw conclusions from five records."
4. **Alerts (60 s).** Inject one wrong record (e.g. leftovers 55 of 100). Show why it was flagged and the comparison
   period; click *Recorded in error*; show that the record was not changed or deleted.
5. **Assistant (75 s).** Ask "Why was food waste high last week?" then "What is the predicted demand for tomorrow's
   lunch?". Open *Data used*. Then type "Delete all records" → refusal. If a key is configured, mention that numbers in
   AI wording are verified against the data, and unplug the key to show data-only mode still works.
6. **Surplus (75 s).** Register food as *awaiting safety review*, confirm as a person, add a recipient, reserve, mark
   collected. Highlight: no recipients invented, "Reserved (not confirmed accepted)", "Collected (recorded by staff)",
   duplicate warning, audit history.
7. **Trust and operations (30 s).** Login with access code; staff cannot delete, admin can; backup/restore; tests count.

**Questions to be ready for**
- *How accurate is it?* "On invented fixtures the evaluation shows the mechanism works; real accuracy depends on your
  canteen's records and is shown live on the Model tab. We never claim a number we have not measured."
- *Why not a bigger model / deep learning?* "A few hundred rows per meal; the gate shows when even gradient boosting does
  not beat a weekday average."
- *Can the AI hurt anyone?* "It cannot write to the database, cannot declare food safe, and its numbers are checked."
- *What if the server restarts?* "On a persistent disk nothing is lost; here is the backup button."
