# Recommendation: what this washing-machine model is good for

*For a utility team considering appliance-level insights from smart-meter data. Based on 19
UK homes from the REFIT dataset; details in [RESULTS.md](RESULTS.md).*

## What the model can and cannot do

The model reads a home's 1-minute total electricity use and estimates when the washing machine
is running and how much power it draws. On three homes it never saw during development:

- It finds about **7 in 10 washes**, and about **7 in 10 washes it reports are real**.
- Its daily washing-energy estimate is about **one third more accurate than assuming no washing
  at all** (daily error 0.29 vs 0.43 kWh), but it **under-estimates total washing energy: its totals are
  39–69% of the actual**, mainly by missing part of the 2 kW heating phase.
- Accuracy varies a lot between homes. One development home was worse than the "no washing"
  baseline.

**Suitable for:** spotting probable wash events and *when* they usually happen, at the level of
a household's habits over weeks. **Not suitable for:** billing, exact energy accounting, or
automated decisions about an individual customer.

## One opportunity: shifting evening-peak washing

Most washing happens in the morning, but some homes wash in the 16:00–20:00 evening peak a lot:
two of our 19 homes started 38–42% of their washes in that window (median home: 14%). A wash draws
about 2 kW while its heater is on. The energy is small (washing is about 3% of a
home's use, median), so the value is **moving load out of the peak**, not reducing bills.

**Offer:** flag households whose detected washes often start in the evening peak and offer them
simple advice: use the machine's delay timer, or an off-peak tariff where one exists.

**How to tell whether it worked:** run a randomised pilot. Among flagged households, randomly
assign half to receive the advice and half not. Compare the change in the **share of washes
starting 16:00–20:00** between the two groups over the same months (before vs after, treated vs
control). Using wash *timing and counts* rather than the model's kWh estimates avoids its low
energy bias, and the model's detection errors affect both groups in the same way. Put plug
monitors on the washing machines of a small subsample to confirm the detected change is real.

## Limitations

- **Overlapping signatures.** Dishwashers and tumble dryers also heat at about 2 kW and are the
  most common cause of false washes; many false detections coincide with appliances nobody
  monitored. Cold or eco washes without the heater are mostly missed.
- **Missing data.** Even this research dataset has outages: 74–93% of minutes were usable per
  home, and some outages were hidden by forward-filling. Real meter data will need the same
  checks.
- **Household differences.** Error on daily energy ranged from 0.50 to 0.84 across the three
  test homes (lower is better). Results for a small pilot group will vary.
- **UK to India.** The model learned UK machines and routines. Many machines used in India
  (for example semi-automatic or top-loading models) wash cold or have no internal heater, which
  removes the main signal the model relies on. Other heating loads such as water heaters (geysers)
  and air conditioners are common and can look similar, and the time-of-day features encode UK
  routines. The model would need to be retrained and checked on sub-metered Indian homes before
  any use there.

## What changes with 15- or 30-minute smart-meter data

A heating phase (under 40 minutes in 99% of cases in our data) would fall into one or two
readings and be averaged together
with everything else running in the home. The shape that identifies a wash (a sharp 2 kW step,
then low motor and spin activity) disappears, so the machine can no longer be separated reliably
from a kettle, a dishwasher or a shower at minute level. Start times would only be known to within
15–30 minutes, and short or cold washes would be almost invisible. Realistically the goal would
shift from detecting individual washes to estimating washing energy per day or week, or the
likelihood that a home washes in the evening peak, with wider error bands. How much accuracy drops would be measured by re-running the model on
15- and 30-minute aggregates (a planned next step, see README).
