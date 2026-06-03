import { api } from "@/lib/api";
import type { MLBEvBet, MLBTotalsEvBet, NBAEvBet, NHLEvBet } from "@/lib/api";

export const revalidate = 300;

type UnifiedBet = {
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

function normalizeMLB(b: MLBEvBet): UnifiedBet {
  const bookLabel = (b.entry_book || "")
    .replace("williamhill_us", "Caesars")
    .replace("draftkings",     "DraftKings")
    .replace("fanduel",        "FanDuel")
    .replace("betmgm",         "BetMGM")
    .replace("pinnacle",       "Pinnacle");
  return {
    team:           b.team,
    matchup:        b.matchup,
    odds:           b.entry_odds,
    bookLabel,
    model_prob:     b.model_prob,
    pin_prob:       b.pinnacle_prob,
    market_prob:    b.market_prob,
    edge_vs_market: b.edge_vs_market,
    ev:             b.ev,
    kelly_pct:      b.kelly_pct,
    lm:             b.line_move_direction ?? 0,
  };
}

function normalizeNHL(b: NHLEvBet): UnifiedBet {
  return {
    team:           b.team,
    matchup:        b.matchup,
    odds:           b.odds,
    bookLabel:      b.entry_book_label || b.entry_book,
    model_prob:     b.model_prob,
    pin_prob:       b.pinnacle_prob,
    market_prob:    b.market_prob,
    edge_vs_market: b.edge_vs_market,
    ev:             b.ev,
    kelly_pct:      b.kelly_pct,
    lm:             b.line_move_direction ?? 0,
  };
}

function normalizeNBA(b: NBAEvBet): UnifiedBet {
  return {
    team:           b.team,
    matchup:        b.matchup,
    odds:           b.odds,
    bookLabel:      b.entry_book_label || b.entry_book,
    model_prob:     b.model_prob,
    pin_prob:       b.pinnacle_prob,
    market_prob:    b.market_prob,
    edge_vs_market: b.edge_vs_market,
    ev:             b.ev,
    kelly_pct:      b.kelly_pct,
    lm:             b.line_move_direction ?? 0,
  };
}

function Signal({ lm }: { lm: number }) {
  if (lm === 1)  return <span className="text-xs font-semibold text-[#22d3ee] uppercase tracking-wider">Sharp</span>;
  if (lm === -1) return <span className="text-xs font-semibold text-[#f87171] uppercase tracking-wider">Fading</span>;
  return <span className="text-xs font-semibold text-[#4b5563] uppercase tracking-wider">Neutral</span>;
}

function BetCard({ b }: { b: UnifiedBet }) {
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
          <div className="text-[#94a3b8] text-sm">{b.kelly_pct.toFixed(2)}%</div>
        </div>
      </div>
    </div>
  );
}

function TotalsCard({ b }: { b: MLBTotalsEvBet }) {
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
          <div className="text-[#4b5563] text-xs uppercase tracking-wider mb-1">Pred. Total</div>
          <div className="text-[#94a3b8] text-sm">{b.predicted_total.toFixed(1)} runs</div>
        </div>
      </div>
    </div>
  );
}

export default async function Home() {
  const [mlbResult, nhlResult, nbaResult, mlbTotalsResult] = await Promise.allSettled([
    api.mlb.evBets(),
    api.nhl.evBets(),
    api.nba.evBets(),
    api.mlb.totalsEvBets(),
  ]);

  const mlbBets = mlbResult.status === "fulfilled"
    ? (mlbResult.value.bets ?? []).map(normalizeMLB)
    : [];
  const nhlBets = nhlResult.status === "fulfilled"
    ? (nhlResult.value.bets ?? []).map(normalizeNHL)
    : [];
  const nbaBets = nbaResult.status === "fulfilled"
    ? (nbaResult.value.bets ?? []).map(normalizeNBA)
    : [];
  const mlbTotalsBets = mlbTotalsResult.status === "fulfilled"
    ? (mlbTotalsResult.value.bets ?? [])
    : [];

  const date =
    (mlbResult.status === "fulfilled" && mlbResult.value.date) ||
    (nhlResult.status === "fulfilled" && nhlResult.value.date) ||
    (nbaResult.status === "fulfilled" && nbaResult.value.date) ||
    new Date().toISOString().slice(0, 10);

  const noPicksToday = mlbBets.length === 0 && nhlBets.length === 0 && nbaBets.length === 0 && mlbTotalsBets.length === 0;

  return (
    <div>

      {/* Page header */}
      <div className="mb-10 pb-6 border-b border-[#1a3050]">
        <p className="text-xs font-bold uppercase tracking-widest text-[#22d3ee]/60 mb-1">
          {date}
        </p>
        <h1 className="text-3xl font-bold text-white mb-3">Today&apos;s Picks</h1>
        <p className="text-[#94a3b8] text-sm leading-relaxed max-w-2xl">
          We model every game using advanced stats and find bets where the true win probability
          is higher than what the sportsbook odds imply. Every pick below has a{" "}
          <span className="text-white font-medium">mathematical edge</span> — meaning
          if you consistently bet these, you profit long-term.
        </p>
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
                <div className="space-y-3">{mlbBets.map((b, i) => <BetCard key={i} b={b} />)}</div>
              </div>
            )}

            {mlbTotalsBets.length > 0 && (
              <div className="mb-3">
                <p className="text-xs font-semibold uppercase tracking-widest text-[#4b5563] mb-2">Totals</p>
                <div className="space-y-3">{mlbTotalsBets.map((b, i) => <TotalsCard key={i} b={b} />)}</div>
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
              ? <div className="space-y-3">{nhlBets.map((b, i) => <BetCard key={i} b={b} />)}</div>
              : (
                <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#64748b]">
                  No qualifying NHL bets today.
                </div>
              )
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
              ? <div className="space-y-3">{nbaBets.map((b, i) => <BetCard key={i} b={b} />)}</div>
              : (
                <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#64748b]">
                  No qualifying NBA bets today.
                </div>
              )
            }
          </section>

        </div>
      )}

      {/* Legend */}
      <div className="mt-10 pt-6 border-t border-[#1a3050] space-y-6">

        {/* The numbers */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-[#64748b] mb-3">The Numbers</p>
          <dl className="space-y-2.5">
            {[
              { term: "Our Win %",     color: "text-[#94a3b8]", def: "The probability our model gives this team of winning the game." },
              { term: "Market Win %",  color: "text-[#94a3b8]", def: "The win probability implied by the sportsbook odds, with the house cut removed." },
              { term: "Edge",          color: "text-[#22d3ee]", def: "How much higher our win % is than the market's. Anything above 0 means we think you have an advantage." },
              { term: "Profit / $100", color: "text-[#06b6d4]", def: "Expected long-term profit on every $100 wagered. A $7 value means: if you placed this bet 100 times in identical conditions, you'd average $7 profit per bet." },
              { term: "Bet Size",      color: "text-[#94a3b8]", def: "Suggested portion of your bankroll to wager, based on your edge. Capped at 5% to keep risk conservative." },
            ].map(({ term, color, def }) => (
              <div key={term} className="flex gap-4 text-sm">
                <dt className={`${color} font-semibold shrink-0 w-28`}>{term}</dt>
                <dd className="text-[#94a3b8] leading-relaxed">{def}</dd>
              </div>
            ))}
          </dl>
        </div>

        {/* Signals */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-[#64748b] mb-3">Signals</p>
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 rounded-lg border border-[#1a3050] bg-[#0a0f1e] px-4 py-3">
              <p className="text-[#22d3ee] font-semibold text-xs uppercase tracking-wider mb-1.5">Sharp</p>
              <p className="text-[#94a3b8] text-sm leading-relaxed">
                The betting line has moved in our direction since opening. Professional bettors appear to agree with our model.
              </p>
            </div>
            <div className="flex-1 rounded-lg border border-[#1a3050] bg-[#0a0f1e] px-4 py-3">
              <p className="text-[#f87171] font-semibold text-xs uppercase tracking-wider mb-1.5">Fading</p>
              <p className="text-[#94a3b8] text-sm leading-relaxed">
                The line has shifted slightly against us. The bet still qualifies — large moves against are filtered out automatically.
              </p>
            </div>
            <div className="flex-1 rounded-lg border border-[#1a3050] bg-[#0a0f1e] px-4 py-3">
              <p className="text-[#64748b] font-semibold text-xs uppercase tracking-wider mb-1.5">Neutral</p>
              <p className="text-[#94a3b8] text-sm leading-relaxed">
                No significant line movement since opening. The market hasn&apos;t shifted notably either way.
              </p>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="pt-2 border-t border-[#1a3050]">
          <p className="text-sm text-[#64748b]">
            For informational purposes only. EdgeShift picks are not gambling advice.
          </p>
        </div>

      </div>

    </div>
  );
}
