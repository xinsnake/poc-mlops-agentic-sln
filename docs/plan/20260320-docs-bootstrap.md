# Bootstrap Docs: Kmart Demand Forecasting Agent

Decisions and source references for bootstrapping the Why-What-How documentation framework.

## Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Project name | Kmart Demand Forecasting Agent | User selected from options |
| Scope | ISE engagement only (UC1+UC4 with Kmart) | Workshop context stays in `meeting-notes/` |
| Bootstrap depth | Phase 1 + Phase 2 | Enough info from workshop notes to populate personas, journeys, scenarios |
| Priority order | UC4 first, UC1 second | UC4 stands alone and directly reduces manual labour; UC1 adds corrective layer |
| Phase 3 (Solution) | Deferred | Awaiting Discovery phase architectural decisions |
| Glossary | Deferred | Add when terminology confusion arises (Blue Yonder terms may warrant this early) |

## Execution Plan — Phase 1 + Phase 2

### Phase 1 — Foundation

Create the business area root and core framing documents.

| # | File | Status |
|---|------|--------|
| 1.1 | `docs/business/business.index.md` | To create |
| 1.2 | `docs/business/business-context.md` | To create |
| 1.3 | `docs/business/vision.md` | To create |
| 1.4 | `docs/business/outcomes.md` | To create |
| 1.5 | `docs/business/scope.md` | To create |
| 1.6 | `docs/business/assumptions-and-risks.md` | To create |

**Gate:** Someone reading 1.2–1.5 can explain what the project is, who it serves, what it will do, and what it won't.

### Phase 2 — Who and Why

Define the primary persona, one journey, and derived scenarios.

| # | File | Status |
|---|------|--------|
| 2.1 | `docs/business/personas/personas.index.md` | To create |
| 2.2 | `docs/business/personas/demand-planner.md` | To create |
| 2.3 | `docs/business/journeys/journeys.index.md` | To create |
| 2.4 | `docs/business/journeys/JNY-001-weekly-forecast-exception-review.md` | To create |
| 2.5 | `docs/business/scenarios/scenarios.index.md` | To create |
| 2.6 | `docs/business/scenarios/SCN-001-temporary-anomaly-ringfence.md` | To create |
| 2.7 | `docs/business/scenarios/SCN-002-baseline-shift-realignment.md` | To create |
| 2.8 | `docs/business/scenarios/SCN-003-event-driven-bulk-ringfence.md` | To create |

**Gate:** The primary persona has at least one journey with derived scenarios, understandable without knowing the technical solution.

### Final step

| # | Action | Status |
|---|--------|--------|
| 3.1 | Update `docs/docs.index.md` with links to all created files | To do |

## Sources

All content is synthesised from these meeting notes (read-only — not modified):

| File | Content |
|------|---------|
| `meeting-notes/20260320 note - point in time understanding.md` | Post-workshop status, use case summary, success criteria, blockers, next steps |
| `meeting-notes/20260319 note - kmart workshop.md` | Kmart session detail: use case deep dive, workflow, pain points, success criteria |
| `meeting-notes/20260319 workshop day 2 notes.md` | Day 2 AI-generated summary: co-engineering model, selection criteria |
| `meeting-notes/20260318 workshop day 1 notes.md` | All-division themes, opportunity cards, HMW statements |
| `meeting-notes/20260313 note - point in time understanding.md` | Pre-workshop survey signals, early domain tally |
| `meeting-notes/20260313 note - questionnaire analysis.md` | Division-level survey analysis |
| `meeting-notes/20260227 note - point in time understanding.md` | Early engagement structure and strategic direction |
