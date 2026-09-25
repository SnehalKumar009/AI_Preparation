# Software Release 4.2: Status

Release 4.2 contains the new curb-climbing controller, required for 5 of the
12 Albers Markt stores (the ones with high curbs in Utrecht).

Status: delayed. Field tests in Utrecht on 5 August showed a 7% failure rate
climbing curbs above 14 cm (target: under 1%). Root cause: the wheel-torque
model underestimates load on wet surfaces.

New estimate: a fix is in review; release candidate 4.2-rc3 is expected on
25 September, with a two-week field validation before general release.

Stores that do NOT need 4.2 (7 of 12) can go live on release 4.1.
