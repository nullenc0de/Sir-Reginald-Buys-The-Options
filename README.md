# Asymmetric Options Agent

A fully autonomous trading agent for binary events and lottery ticket options. Designed to run inside **Claude Code** where Claude acts as both the reasoning engine AND executor - automatically researching, analyzing positions, and asking for confirmation before trades.

---

## TL;DR - The Only Command You Need

```
"Run the agent"
```

**That's it.** Claude will:
1. Check your positions for exits needed (moonshots, targets hit)
2. Identify catalysts happening today/tomorrow
3. Run research searches for new plays (if you have capital available)
4. Ask you to confirm any actions before executing

---

## How It Works (Autonomous Mode)

When you say "run the agent", Claude:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DETECT TIME OF DAY                                          │
│     → Determines mode: pre-market, market open, mid-day, etc.   │
├─────────────────────────────────────────────────────────────────┤
│  2. CHECK POSITIONS FOR ACTIONS                                 │
│     → Moonshots (3x+): URGENT SELL                              │
│     → Targets hit (2x+): Sell Tranche 1                         │
│     → Warming (1.5x+): Watch closely                            │
│     → Catalysts today/tomorrow: Alert                           │
├─────────────────────────────────────────────────────────────────┤
│  3. GENERATE RESEARCH TASKS (if research mode)                  │
│     → Pulls search queries from 100x Framework                  │
│     → Pulls searches from Research Framework                    │
│     → Includes seasonal factors                                 │
├─────────────────────────────────────────────────────────────────┤
│  4. CLAUDE EXECUTES                                             │
│     → Runs web searches for FDA dates, short squeezes, etc.     │
│     → Evaluates positions (e.g., "SLV catalyst passed - valid?")│
│     → Prepares exit orders                                      │
│     → ASKS YOU TO CONFIRM before any trades                     │
└─────────────────────────────────────────────────────────────────┘
```

### Example Autonomous Session

```
You: Run the agent

Claude: [Detects it's 4:05 PM ET - After Hours mode]

  📊 PORTFOLIO: +$103 P/L, 6 positions, 8.6% risk

  🟡 APLD at 1.75x - approaching 2x target (earnings Jan 7)
  ⚠️  SLV catalyst PASSED (Jan 1) - need to evaluate

  🔬 RESEARCH MODE: 12 searches to run, $2,532 available

  What should I do?
  [ ] Run research searches
  [ ] Evaluate SLV position
  [ ] Focus on APLD exit
  [ ] All of the above

You: Do it all

Claude: [Runs web searches for FDA PDUFA dates]
        [Searches China silver export controls news]
        [Finds SLV thesis is STRONGER - recommends HOLD]
        [Prepares APLD T1 exit order: 4 contracts @ $1.10]
        [Attempts to place order, reports PDT restriction]

  ✅ Session complete. Your action items:
  - Monday: Place APLD exit order
  - Jan 7: APLD earnings - watch for spike
  - Jan 13: TVTX FDA decision
```

---

## Quick Start (Claude Code)

### 1. Get Alpaca API Keys
1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Go to Paper Trading → API Keys
3. Generate a new key pair

### 2. Start Claude Code
```bash
claude
```

### 3. Tell Claude to Load & Run
```
Load the asymmetric options agent from ~/asymmetric_options_agent.py
Use API key: PKXXX and secret: XXX
Run the agent
```

### 4. Every Time You Check In
```
Run the agent
```

The agent automatically detects the time and runs the appropriate check, pulling dynamic data from the built-in **100x Framework** and **Research Framework**:

| Time (ET) | What Runs | Framework Data Shown |
|-----------|-----------|----------------------|
| Before 9:30 AM | Pre-market prep | Seasonal factors, current search queries, binary event categories |
| 9:30 - 11:00 AM | Full daily briefing | Complete portfolio analysis, catalyst calendar, exit status |
| 11:00 AM - 2:00 PM | Mid-day research | Binary event searches, research framework categories with example queries |
| 2:00 - 3:30 PM | Afternoon check | Tomorrow's catalysts, quick scan suggestions |
| 3:30 - 4:00 PM | Power hour | Execution rules, exit reminders from 100x framework |
| After 4:00 PM | Evening research | Full framework dump: searches, binary events, screening criteria |
| Weekend | Deep research | Seasonal factors, all search queries, binary event deep dives |

**Key:** The `run()` function doesn't show hardcoded prompts - it pulls **live data** from `get_100x_framework()` and `get_research_framework()` which generate date-aware search queries automatically.

---

## Daily Workflow

### It's Simple: Just Run the Agent

**Morning, lunch, afternoon, EOD** - just say:
```
Run the agent
```

It figures out what you need based on the time.

### When You See Something Interesting

Position mooning:
```
APLD is up 3x - execute the exit plan
```

Want more detail:
```
Show me the full briefing
```

Find new plays:
```
Search for lottery tickets for next month
```

### Need Help?
```
Show me the help
```

Shows a quick reference of all commands and concepts.

---

## What to Say to Claude

### Finding New Plays
```
Search for binary event lottery tickets for February
Focus on FDA approvals and earnings with high short interest
```

```
Find second-order effects from the polar vortex weather event
```

### Executing Trades
```
Execute the lottery basket for these events:
- TVTX FDA Jan 13, $250 allocation
- APLD Earnings Jan 7, $300 allocation
```

### Managing Positions
```
Show scaled exit plan for all positions
```

```
APLD hit 2x target - sell tranche 1 (50% of position)
```

### Research
```
Validate my current positions - check if the catalysts are still on track
```

```
What's the short interest on APLD? Is the squeeze thesis still valid?
```

---

## Two Strategies

### 1. Second-Order Effects (Macro Asymmetry)
Markets price **primary events** but miss **second-order consequences**.

**Example:** COVID lockdowns → can't visit family → send flowers → Long 1-800-Flowers = **300% gain**

### 2. Binary Event Lottery Tickets (100x Potential)
Deep OTM options on stocks with **defined binary catalysts** (FDA approvals, earnings + short squeeze, regulatory decisions).

**Example:** Biotech FDA approval → $0.50 option → $50.00 = **100x**

---

## Built-in Frameworks

The agent includes two dynamic frameworks that generate **date-aware** research suggestions. These aren't static - they update based on the current date.

### 100x Framework (`get_100x_framework()`)

The mechanics of finding 100x lottery ticket plays:

| Component | What It Provides |
|-----------|------------------|
| `binary_event_types` | FDA PDUFA, Earnings Squeeze, Regulatory, Conference, M&A - each with specific search queries |
| `current_searches` | Pre-built search strings like "PDUFA calendar January 2026" (auto-updates with current month) |
| `screening_criteria` | Must-have and prefer criteria for filtering candidates |
| `execution_rules` | Entry timing, position sizing, exit rules, dilution warnings |
| `ideal_setup` | Float, short interest, delta, time to expiry targets |

### Research Framework (`get_research_framework()`)

Eight categories for finding second-order effects:

| Category | Goal |
|----------|------|
| `economic_calendar` | Earnings, Fed events, data releases this week |
| `geopolitical_now` | Global events creating obscure catalysts |
| `weather_anomalies` | Extreme weather impacting energy, agriculture, shipping |
| `viral_consumer_trends` | TikTok trends, behavioral shifts |
| `unusual_options_activity` | Where smart money is positioning |
| `seasonal_patterns` | Tax effects, holiday spending, cyclical patterns |
| `sector_flows` | Fund flows, sector rotation |
| `supply_chain` | Disruptions creating winners/losers |

Each category includes `example_searches` that auto-generate with current dates.

### Seasonal Factors (Auto-Detected)

The research framework automatically detects what's seasonally relevant:

| Time Period | Factors Shown |
|-------------|---------------|
| Late Dec - Mid Jan | January Effect, Tax-Loss Reversal, New Year Resolution Spending |
| Early January | Earnings Season Preview |
| May - June | Summer Driving Season |
| June - November | Hurricane Season |
| July - August | Back to School |
| November - December | Holiday Shopping Season |

---

## Scaled Exit Strategy

The agent uses a 3-tranche exit system to capture both protection AND moonshots:

| Tranche | % to Sell | Target | Purpose |
|---------|-----------|--------|---------|
| T1 | 50% | 2x | Recover cost basis |
| T2 | 30% | 4x | Capture big move |
| T3 | 20% | 10x | Moonshot potential |

**Why this works:**
- If it goes to 2x and dies: You break even (sold 50% at 2x = 100% of cost back)
- If it goes to 10x: You still have 20% riding = 2x total return
- If it goes to 100x: That 20% = 20x total return

---

## Catalyst Storage (Hybrid System)

Catalyst data is stored in two places for redundancy:

| Data | Local File | Alpaca Orders |
|------|------------|---------------|
| Ticker | Yes | Yes |
| Catalyst Date | Yes | Yes (in client_order_id) |
| Description | Yes | No (48 char limit) |

**Local file:** `~/.asymmetric_catalysts.json`

### If You Lose the Local File
```
Rebuild my catalyst data from Alpaca order history
```

This recovers dates from order history. You'll need to re-add descriptions manually.

### Backup
```
Backup my catalyst file
```

### View All Catalysts
```
List all stored catalysts
```

---

## Key Functions Reference

### Autonomous System (NEW)
| Function | What It Does |
|----------|-------------|
| `autonomous_run(agent)` | Main autonomous entry point - checks positions, generates research tasks, returns structured actions |
| `check_actions_needed(agent)` | Scans all positions for moonshots, targets hit, catalysts today/tomorrow |
| `get_research_tasks(agent)` | Generates research tasks from 100x and Research frameworks |

### Daily Operations
| Function | What to Ask Claude |
|----------|-------------------|
| `run(agent)` | "Run the agent" - time-aware, shows framework data |
| `daily_briefing(agent)` | "Run daily briefing" |
| `quick_check(agent)` | "Quick check on positions" |
| `end_of_day_review(agent)` | "End of day review" |

### Finding Trades
| Function | What to Ask Claude |
|----------|-------------------|
| `scan_binary_events(agent, events)` | "Scan for lottery tickets" |
| `find_lottery_tickets(agent, ticker, date, catalyst)` | "Find options for TVTX FDA Jan 13" |
| `get_research_framework()` | "Show me the research framework" |
| `get_100x_framework()` | "Explain the 100x lottery ticket theory" |

### Executing & Managing
| Function | What to Ask Claude |
|----------|-------------------|
| `execute_lottery_basket(agent, basket)` | "Execute the basket" |
| `get_scaled_exit_plan(agent)` | "Show scaled exit plan" |
| `place_scaled_exit_orders(agent, tranche=1)` | "Sell tranche 1" |
| `moonshot_check(agent)` | "Any moonshots?" |

### Catalyst Management
| Function | What to Ask Claude |
|----------|-------------------|
| `add_catalyst(ticker, desc, date)` | "Add catalyst: TICKER on DATE for REASON" |
| `list_catalysts()` | "List all catalysts" |
| `remove_catalyst(ticker)` | "Remove TICKER catalyst" |
| `backup_catalysts()` | "Backup catalyst file" |
| `rebuild_catalysts_from_orders(key, secret)` | "Rebuild catalysts from orders" |

---

## FAQs

### How often should I check?

| Situation | Frequency |
|-----------|-----------|
| No catalysts today | Once in morning, once EOD |
| Catalyst today | Every 1-2 hours |
| Position spiking | Watch closely, be ready to sell |

### Should I use stop-losses?

**No.** Options prices fluctuate wildly. A stop-loss will get triggered by normal volatility and sell your position before the catalyst. Let losers expire worthless - that's the cost of playing lottery tickets.

### What if I miss a moonshot?

The scaled exit system helps. Even if you miss the absolute peak:
- T1 at 2x recovers your cost
- T2 at 4x locks in profit
- T3 rides for the moon

### How much should I allocate per trade?

- **Per position:** $200-500 (what you can lose 100%)
- **Total portfolio risk:** Max 20% in lottery tickets
- **Expect:** 80-90% of tickets expire worthless

### What's the win rate?

Realistically: **10-20% of plays win**

But winners can be 5x-100x, so one winner pays for many losers.

### When do I sell a losing position early?

- Stock drifts significantly against you before catalyst (thesis broken)
- News comes out that invalidates the catalyst
- You need the capital for a better opportunity

Otherwise: **Hold until catalyst, then reassess**

### Why paper trading?

Start with paper trading to:
1. Learn the system without risking real money
2. Validate your thesis identification
3. Practice the exit discipline
4. Build confidence

Move to live when you're consistently identifying good setups.

---

## Execution Rules

### Entry
- Buy 30-45 days before catalyst
- Deep OTM (Delta < 0.10, 25-60% OTM)
- Only allocate what you can lose 100%

### Exit - WINNING
- **Sell INTO the spike** using scaled tranches
- Never hold overnight after the catalyst news
- Biotechs often announce dilution post-approval - exit fast!

### Exit - LOSING
- If thesis breaks pre-catalyst, consider cutting
- Otherwise let it ride to catalyst date
- Don't average down on lottery tickets

---

## Binary Event Types

| Type | Description | Where to Find |
|------|-------------|---------------|
| **FDA PDUFA** | Drug approval deadline | biopharmcatalyst.com |
| **Earnings Squeeze** | Beat + high short interest | fintel.io, earningswhispers.com |
| **Regulatory** | Government decisions | congress.gov |
| **Conference** | Major announcements | CES, JPM Healthcare, GTC |
| **Weather** | Commodity impacts | NOAA, weather.com |
| **Geopolitical** | Trade, tariffs, conflict | News sources |

---

## Example Session

```
You: Run the agent

Claude: [Detects it's 11:30 AM - runs mid-day check]

  ☀️ MID-DAY CHECK
  ──────────────────────────────────────

  🟡 APLD250117C00035000: 1.9x warming
  🔴 TVTX250124C00015000: 0.7x underwater

  🎯 APPROACHING TARGETS:
     • APLD: 1.9x (T1 target: 2.0x)

  ☕ MID-DAY = RESEARCH TIME

  💰 $1,500 available for new plays

  📅 SEASONAL EDGE: JANUARY EFFECT
     Small caps and beaten-down stocks often rally as tax-loss selling ends
     Beneficiaries: Small cap value, stocks that dropped 30%+ in prior year...

  🎯 BINARY EVENT CATEGORIES (from 100x framework):
     • Fda Pdufa
       Search: "FDA PDUFA dates January 2026"
     • Earnings Squeeze
       Search: "high short interest stocks earnings January 2026"

  🔬 RESEARCH FRAMEWORK CATEGORIES:
     • Economic Calendar
       Search: "economic calendar January 02 2026"
     • Geopolitical Now
       Search: "breaking geopolitical news today"

You: APLD just hit 2x. Execute tranche 1.

Claude: Selling 4 of 8 APLD contracts at $1.10...
  Order filled. Recovered cost basis.
  Remaining 4 contracts riding for T2 (4x = $2.20)

You: Run the agent

Claude: [Detects it's 7:00 PM - runs evening research session]

  🌙 AFTER HOURS - End of Day Review
  ──────────────────────────────────────

  🟢 Day P/L: +$220.00

  🔬 EVENING RESEARCH SESSION

  🎯 100x FRAMEWORK - TONIGHT'S SEARCHES:
     • "PDUFA calendar January 2026 FDA decisions"
     • "biotech catalyst calendar January 2026"
     • "high short interest earnings January 2026"
     • "short squeeze candidates 2026"

  🔬 BINARY EVENT DEEP DIVES:
     • Fda Pdufa
       Search: "FDA PDUFA dates January 2026"
       Sites: biopharmcatalyst.com, fdatracker.com

  ✅ SCREENING CRITERIA (must have):
     • Defined binary catalyst with SPECIFIC DATE
     • Options available with strikes 30-50% OTM
```

---

## File Structure

```
~/asymmetric_options_agent.py          # Main agent (~3700 lines)
~/asymmetric_options_agent_README.md   # This file
~/.asymmetric_catalysts.json           # Catalyst storage (auto-created)
```

### Key Code Sections

| Lines (approx) | Section |
|----------------|---------|
| 1-250 | Catalyst storage functions |
| 269-575 | Alpaca API client |
| 577-675 | RiskManager |
| 676-750 | ClaudeCodeLLM (Claude Code interface) |
| 751-850 | OptionFinder |
| 851-995 | Agent class |
| 997-1147 | `get_100x_framework()` - Binary event framework |
| 1149-1327 | `get_research_framework()` - Second-order effects framework |
| 1329-1640 | Trade finding and execution functions |
| 1643-1925 | Lottery basket execution |
| 1927-2185 | Profit target and monitoring system |
| 2186-2440 | Scaled exit strategy |
| 2442-2596 | Moonshot check and profit taking |
| 2598-2890 | `daily_briefing()` - Full daily report |
| 2892-2973 | `quick_check()`, `end_of_day_review()` |
| **2975-3270** | **AUTONOMOUS SYSTEM** |
| 2975-3061 | `check_actions_needed()` - Position action scanner |
| 3063-3127 | `get_research_tasks()` - Framework-based research generator |
| 3129-3272 | `autonomous_run()` - Main autonomous entry point |
| 3274-3660 | `run()` - Time-aware display (uses frameworks) |
| 3661-3720 | `help_me()` - Quick reference |

---

## Risk Warning

**This is not investing. This is asymmetric speculation.**

- 80-90% of lottery tickets expire worthless
- You are betting on low-probability, high-payoff events
- Only use money you can lose 100%
- One 10-20x winner can pay for many losers
- Paper trade first until you understand the system

---

## Sources for Binary Events

- **FDA/PDUFA:** biopharmcatalyst.com, fdatracker.com
- **Short Interest:** fintel.io, shortinterest.com
- **Earnings:** earningswhispers.com
- **Unusual Options:** barchart.com/options/unusual-activity
- **Weather:** NOAA Climate Prediction Center

---

**Built for Claude Code** - AI reasoning meets binary event execution.

> "The market prices the disaster, not the adaptation"
> "100x returns are not random - they are structural"
