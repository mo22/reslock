# v0.12.0 (Schema v5) ist veröffentlicht, aber nicht ausgerollt

**Stand 2026-08-13, nach dem Release:** `v0.12.0` ist getaggt, das GitHub-Release steht, und
trusted publishing hat nach PyPI geschoben — geprüft: `uvx --from 'reslock==0.12.0'` zieht
das Rad, meldet `reslock, version 0.12.0` und `SCHEMA_VERSION = 5`, und
`SchemaVersionMismatch` / `peek_state_version` / `force_reset_state` sind im
veröffentlichten Paket vorhanden.

**Auf kirk läuft davon weiterhin nichts.** Schritt 1 unten ist erledigt, die Schritte 2–5
sind offen, und bis dahin ist die Änderung dort wirkungslos.

## Was der Rollout verlangt

v4 → v5 ist ein Schemasprung, und mit dem neuen Verhalten ist er ein **erzwungenes
Wartungsfenster**, keine rollende Aktualisierung: sobald eine Datei v5 trägt, verweigern alle
v4-Verbraucher, und umgekehrt.

1. ~~GitHub-Release mit Tag `v0.12.0` anlegen → trusted publishing schiebt nach PyPI
   (nicht `uv publish` von Hand, siehe „Publishing" in AGENTS.md).~~ **Erledigt 2026-08-13**,
   Run 31694829917 grün, auf PyPI verifiziert.
2. Alle Verbraucher auf kirk stoppen — Stand 2026-08-12 waren das **17 Container**
   (2 aiserver + kirk-rpcserver + 14 scriba-Mandanten; die zwei intellex-ocr mounten die
   Datei nur, haben reslock nicht installiert).
3. Überall auf ≥0.12.0 heben.
4. `reslock reset --force` (oder `state.json` löschen) — mit allem gestoppt, sonst hält ein
   laufender Prozess Ressourcen, die die Datei nicht mehr kennt.
5. Starten; Kapazitäten registrieren sich über `set_resources()` neu.

## Warum das nicht schleifen sollte

Solange die Flotte unter 0.12.0 läuft, ist die alte Mine noch scharf: scriba
`backend/.venv` mit reslock 0.6.0 liest die v4-Datei erfolgreich, sieht `version=2`, setzt
still zurück — ein `transact` danach schreibt das zurück. (Gemessen von isidore am
2026-08-13 gegen eine Kopie der echten Datei. Von den ursprünglich drei gemeldeten
Host-Venvs ist das das einzige verbleibende: `~/.local` 0.1.2 lässt sich unter Python 3.12
gar nicht importieren, und `aiserver/.venv` steht seit 2026-08-12 21:57 auf 0.11.1.)

Das Aufräumen dieses Venvs ist Moritz' Entscheidung und **wirkt sofort**, im Gegensatz zum
Rollout, der nur künftige Abweichungen laut macht.

## Vorher zu prüfen

Ob die Verbraucher `pool.acquire()` in ein breites `except Exception` wickeln — das würde den
neuen lauten Fehler wieder stillstellen und den ganzen Zweck aushebeln. Betrifft aiserver,
scriba und kirk-rpcserver; in keinem der drei Repos nachgesehen.
