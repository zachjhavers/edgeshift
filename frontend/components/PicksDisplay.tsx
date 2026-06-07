"use client";

import { useState, useEffect } from "react";
import type { MLBTotalsEvBet, SoccerEvBet } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────────────

export type UnifiedBet = {
  team:           string;
  matchup:        string;
  odds:           number;
  bookLabel:      string;
  model_prob:     number;
  pin_prob:       number | null;
  market_prob:    number;
  edge_vs_market: number;
  ev:             number;
  kelly_pct:      number;
  lm:             number;
};

type Props = {
  date:           string;
  mlbBets:        UnifiedBet[];
  nhlBets:        UnifiedBet[];
  nbaBets:        UnifiedBet[];
  mlbTotalsBets:  MLBTotalsEvBet[];
  soccerBets:     SoccerEvBet[];
};

// ── Helpers ────────────────────────────────────────────────────────────────

function Signal({ lm }: { lm: number }) {
  if (lm === 1)  return <span className="text-xs font-semibold text-[#22d3ee] uppercase tracking-wider">Sharp</span>;
  if (lm === -1) return <span className="text-xs font-semibold text-[#f87171] uppercase tracking-wider">Fading</span>;
  return <span className="text-xs font-semibold text-[#4b5563] uppercase tracking-wider">Neutral</span>;
}

function BetSize({ kelly_pct, bankroll }: { kelly_pct: number; bankroll: number | null }) {
  if (bankroll && bankroll > 0) {
    const amount = (kelly_pct / 100) * bankroll;
    return (
      <div>
        <div className="text-white font-bold text-base">${amount.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</div>
        <div className="text-[#4b5563] text-xs mt-0.5">{kelly_pct.toFixed(2)}% of bankroll</div>
      </div>
    );
  }
  return <div className="text-[#94a3b8] text-sm">{kelly_pct.toFixed(2)}%</div>;
}

function BetCard({ b, bankroll }: { b: UnifiedBet; bankroll: number | null }) {
  return (
    <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-xl font-bold text-white leading-tight">{b.team}</div>
          <div className="text-sm text-[#64748b] mt-1">{b.matchup}</div>
        </div>
        <div className="flex flex-col items-end gap-1.5 shrink-0 ml-4">
          <Signal lm={b.lm} />
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-[#cbd5e1] text-sm font-medium">{b.odds.toFixed(3)}</span>
            <span className="text-[#4b5563] text-xs">at</span>
            <span className="text-[#64748b] text-xs">{b.bookLabel}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-5 gap-x-4 gap-y-3 pt-4 border-t border-[#1a3050]">
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Our Win %</div>
          <div className="text-[#cbd5e1] font-semibold text-sm">{(b.model_prob * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Market Win %</div>
          <div className="text-[#94a3b8] text-sm">
            {b.pin_prob !== null
              ? `${(b.pin_prob * 100).toFixed(1)}%`
              : `${(b.market_prob * 100).toFixed(1)}%`}
          </div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Edge</div>
          <div className="text-[#22d3ee] font-semibold text-sm">+{(b.edge_vs_market * 100).toFixed(1)}pp</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Profit / $100</div>
          <div className="text-[#06b6d4] font-bold text-base">${b.ev.toFixed(0)}</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Bet Size</div>
          <BetSize kelly_pct={b.kelly_pct} bankroll={bankroll} />
        </div>
      </div>
    </div>
  );
}

function TotalsCard({ b, bankroll }: { b: MLBTotalsEvBet; bankroll: number | null }) {
  return (
    <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-xl font-bold text-white leading-tight">{b.label}</div>
          <div className="text-sm text-[#64748b] mt-1">{b.matchup}</div>
        </div>
        <div className="flex flex-col items-end gap-1.5 shrink-0 ml-4">
          <Signal lm={b.line_move_direction} />
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-[#cbd5e1] text-sm font-medium">{b.entry_odds.toFixed(3)}</span>
            <span className="text-[#4b5563] text-xs">at</span>
            <span className="text-[#64748b] text-xs">{b.entry_book}</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 sm:grid-cols-5 gap-x-4 gap-y-3 pt-4 border-t border-[#1a3050]">
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Our P({b.side})</div>
          <div className="text-[#cbd5e1] font-semibold text-sm">{(b.model_prob * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Market P</div>
          <div className="text-[#94a3b8] text-sm">{(b.market_prob * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Edge</div>
          <div className="text-[#22d3ee] font-semibold text-sm">+{(b.edge_vs_market * 100).toFixed(1)}pp</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Profit / $100</div>
          <div className="text-[#06b6d4] font-bold text-base">${b.ev.toFixed(0)}</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Bet Size</div>
          <BetSize kelly_pct={b.kelly_pct} bankroll={bankroll} />
        </div>
      </div>
    </div>
  );
}

// ── Bankroll input ─────────────────────────────────────────────────────────

function BankrollInput({ bankroll, onChange }: { bankroll: number | null; onChange: (v: number | null) => void }) {
  const [raw, setRaw] = useState(bankroll ? String(bankroll) : "");

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const val = e.target.value.replace(/[^0-9]/g, "");
    setRaw(val);
    onChange(val ? Number(val) : null);
  }

  function handleClear() {
    setRaw("");
    onChange(null);
  }

  return (
    <div className="flex items-center gap-2">
      <label className="text-xs font-semibold uppercase tracking-widest text-[#4b5563] shrink-0">
        Bankroll
      </label>
      <div className="relative flex items-center">
        <span className="absolute left-3 text-[#64748b] text-sm pointer-events-none">$</span>
        <input
          type="text"
          inputMode="numeric"
          value={raw}
          onChange={handleChange}
          placeholder="0"
          className="w-28 pl-7 pr-3 py-1.5 rounded-lg bg-[#0a0f1e] border border-[#1a3050] text-white text-sm placeholder-[#4b5563] focus:outline-none focus:border-[#22d3ee]/50 transition-colors"
        />
      </div>
      {bankroll && (
        <button
          onClick={handleClear}
          className="text-[#4b5563] hover:text-[#94a3b8] text-xs transition-colors"
          aria-label="Clear bankroll"
        >
          ✕
        </button>
      )}
      {bankroll && bankroll > 0 && (
        <span className="text-xs text-[#4b5563]">
          Bet sizes shown in dollars
        </span>
      )}
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

function SoccerCard({ b, bankroll }: { b: SoccerEvBet; bankroll: number | null }) {
  return (
    <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-xl font-bold text-white leading-tight">{b.label}</div>
          <div className="text-sm text-[#64748b] mt-1">{b.matchup}</div>
        </div>
        <div className="flex flex-col items-end gap-1.5 shrink-0 ml-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#f59e0b]">
            {b.market === "h2h" ? "Match Result" : "Goals"}
          </span>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-[#cbd5e1] text-sm font-medium">{b.entry_odds.toFixed(3)}</span>
            <span className="text-[#4b5563] text-xs">at</span>
            <span className="text-[#64748b] text-xs">Pinnacle</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 sm:grid-cols-4 gap-x-4 gap-y-3 pt-4 border-t border-[#1a3050]">
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Our Prob</div>
          <div className="text-[#cbd5e1] font-semibold text-sm">{(b.model_prob * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Market Prob</div>
          <div className="text-[#94a3b8] text-sm">{(b.market_prob * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Edge</div>
          <div className="text-[#22d3ee] font-semibold text-sm">+{(b.edge_vs_market * 100).toFixed(1)}pp</div>
        </div>
        <div>
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Profit / $100</div>
          <div className="text-[#06b6d4] font-bold text-base">${b.ev.toFixed(0)}</div>
        </div>
      </div>
    </div>
  );
}

export default function PicksDisplay({ date, mlbBets, nhlBets, nbaBets, mlbTotalsBets, soccerBets }: Props) {
  const [bankroll, setBankroll] = useState<number | null>(null);

  // Persist bankroll to localStorage
  useEffect(() => {
    const stored = localStorage.getItem("edgeshift_bankroll");
    if (stored) setBankroll(Number(stored));
  }, []);

  function handleBankrollChange(val: number | null) {
    setBankroll(val);
    if (val) {
      localStorage.setItem("edgeshift_bankroll", String(val));
    } else {
      localStorage.removeItem("edgeshift_bankroll");
    }
  }

  const noPicksToday = mlbBets.length === 0 && nhlBets.length === 0 && nbaBets.length === 0 && mlbTotalsBets.length === 0 && soccerBets.length === 0;

  return (
    <div>
      {/* Page header */}
      <div className="mb-10 pb-6 border-b border-[#1a3050]">
        <p className="text-xs font-bold uppercase tracking-widest text-[#22d3ee]/60 mb-1">{date}</p>
        <h1 className="text-3xl font-bold text-white mb-3">Today&apos;s Picks</h1>
        <p className="text-[#94a3b8] text-sm leading-relaxed max-w-2xl mb-5">
          We model every game using advanced stats and find bets where the true win probability
          is higher than what the sportsbook odds imply. Every pick below has a{" "}
          <span className="text-white font-medium">mathematical edge</span> — meaning
          if you consistently bet these, you profit long-term.
        </p>
        <BankrollInput bankroll={bankroll} onChange={handleBankrollChange} />
      </div>

      {noPicksToday ? (
        <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-16 text-center">
          <div className="text-white font-semibold text-xl mb-2">No picks today</div>
          <div className="text-[#94a3b8] text-sm leading-relaxed">
            Our models didn&apos;t find any bets with a strong enough edge today.<br />
            Check back tomorrow morning — picks update daily.
          </div>
        </div>
      ) : (
        <div className="space-y-12">

          {/* Baseball */}
          <section>
            <div className="flex items-baseline gap-3 mb-4">
              <h2 className="text-sm font-bold uppercase tracking-widest text-white">Baseball</h2>
              {(mlbBets.length + mlbTotalsBets.length) > 0
                ? <span className="text-sm text-[#64748b]">{mlbBets.length + mlbTotalsBets.length} pick{(mlbBets.length + mlbTotalsBets.length) !== 1 ? "s" : ""}</span>
                : <span className="text-sm text-[#64748b]">No picks today</span>
              }
            </div>
            {mlbBets.length > 0 && (
              <div className="mb-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-[#4b5563] mb-2">Moneyline</p>
                <div className="space-y-3">{mlbBets.map((b, i) => <BetCard key={i} b={b} bankroll={bankroll} />)}</div>
              </div>
            )}
            {mlbTotalsBets.length > 0 && (
              <div className="mb-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-[#4b5563] mb-2">Totals</p>
                <div className="space-y-3">{mlbTotalsBets.map((b, i) => <TotalsCard key={i} b={b} bankroll={bankroll} />)}</div>
              </div>
            )}
            {mlbBets.length === 0 && mlbTotalsBets.length === 0 && (
              <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#64748b]">
                No qualifying MLB bets today.
              </div>
            )}
          </section>

          {/* Hockey */}
          <section>
            <div className="flex items-baseline gap-3 mb-4">
              <h2 className="text-sm font-bold uppercase tracking-widest text-white">Hockey</h2>
              {nhlBets.length > 0
                ? <span className="text-sm text-[#64748b]">{nhlBets.length} pick{nhlBets.length !== 1 ? "s" : ""}</span>
                : <span className="text-sm text-[#64748b]">No picks today</span>
              }
            </div>
            {nhlBets.length > 0
              ? <div className="space-y-3">{nhlBets.map((b, i) => <BetCard key={i} b={b} bankroll={bankroll} />)}</div>
              : <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#64748b]">No qualifying NHL bets today.</div>
            }
          </section>

          {/* Basketball */}
          <section>
            <div className="flex items-baseline gap-3 mb-4">
              <h2 className="text-sm font-bold uppercase tracking-widest text-white">Basketball</h2>
              {nbaBets.length > 0
                ? <span className="text-sm text-[#64748b]">{nbaBets.length} pick{nbaBets.length !== 1 ? "s" : ""}</span>
                : <span className="text-sm text-[#64748b]">No picks today</span>
              }
            </div>
            {nbaBets.length > 0
              ? <div className="space-y-3">{nbaBets.map((b, i) => <BetCard key={i} b={b} bankroll={bankroll} />)}</div>
              : <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#64748b]">No qualifying NBA bets today.</div>
            }
          </section>

          {/* World Cup */}
          {soccerBets.length > 0 && (
            <section>
              <div className="flex items-baseline gap-3 mb-4">
                <h2 className="text-sm font-bold uppercase tracking-widest text-white">World Cup</h2>
                <span className="text-xs font-semibold uppercase tracking-widest text-[#f59e0b]">FIFA 2026</span>
                <span className="text-sm text-[#64748b]">{soccerBets.length} pick{soccerBets.length !== 1 ? "s" : ""}</span>
              </div>
              <div className="space-y-3">{soccerBets.map((b, i) => <SoccerCard key={i} b={b} bankroll={bankroll} />)}</div>
            </section>
          )}

        </div>
      )}

      {/* Legend */}
      <div className="mt-10 pt-6 border-t border-[#1a3050] space-y-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-[#64748b] mb-3">The Numbers</p>
          <dl className="space-y-2.5">
            {[
              { term: "Our Win %",     color: "text-[#94a3b8]", def: "The probability our model gives this team of winning the game." },
              { term: "Market Win %",  color: "text-[#94a3b8]", def: "The win probability implied by the sportsbook odds, with the house cut removed." },
              { term: "Edge",          color: "text-[#22d3ee]", def: "How much higher our win % is than the market's. Anything above 0 means we think you have an advantage." },
              { term: "Profit / $100", color: "text-[#06b6d4]", def: "Expected long-term profit on every $100 wagered. A $7 value means: if you placed this bet 100 times in identical conditions, you'd average $7 profit per bet." },
              { term: "Bet Size",      color: "text-[#94a3b8]", def: "Suggested wager based on your edge. Enter your bankroll above to see dollar amounts. Capped at 5% to keep risk conservative." },
            ].map(({ term, color, def }) => (
              <div key={term} className="flex gap-4 text-sm">
                <dt className={`${color} font-semibold shrink-0 w-28`}>{term}</dt>
                <dd className="text-[#94a3b8] leading-relaxed">{def}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-[#64748b] mb-3">Signals</p>
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 rounded-lg border border-[#1a3050] bg-[#0a0f1e] px-4 py-3">
              <p className="text-[#22d3ee] font-semibold text-xs uppercase tracking-wider mb-1.5">Sharp</p>
              <p className="text-[#94a3b8] text-sm leading-relaxed">The betting line has moved in our direction since opening. Professional bettors appear to agree with our model.</p>
            </div>
            <div className="flex-1 rounded-lg border border-[#1a3050] bg-[#0a0f1e] px-4 py-3">
              <p className="text-[#f87171] font-semibold text-xs uppercase tracking-wider mb-1.5">Fading</p>
              <p className="text-[#94a3b8] text-sm leading-relaxed">The line has shifted slightly against us. The bet still qualifies — large moves against are filtered out automatically.</p>
            </div>
            <div className="flex-1 rounded-lg border border-[#1a3050] bg-[#0a0f1e] px-4 py-3">
              <p className="text-[#64748b] font-semibold text-xs uppercase tracking-wider mb-1.5">Neutral</p>
              <p className="text-[#94a3b8] text-sm leading-relaxed">No significant line movement since opening. The market hasn&apos;t shifted notably either way.</p>
            </div>
          </div>
        </div>

        <div className="pt-2 border-t border-[#1a3050]">
          <p className="text-sm text-[#64748b]">For informational purposes only. EdgeShift picks are not gambling advice.</p>
        </div>
      </div>
    </div>
  );
}
