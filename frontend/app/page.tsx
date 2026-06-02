import { api } from "@/lib/api";
import type { MLBEvBet, NHLEvBet } from "@/lib/api";

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

function Signal({ lm }: { lm: number }) {
  if (lm === 1)  return <span className="text-[#22d3ee] font-medium text-xs whitespace-nowrap">Sharp ▲</span>;
  if (lm === -1) return <span className="text-[#f87171] font-medium text-xs whitespace-nowrap">Fading ▼</span>;
  return <span className="text-[#4b5563]">—</span>;
}

function EvTable({ bets }: { bets: UnifiedBet[] }) {
  return (
    <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#1a3050]">
            <th className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b]">Pick</th>
            <th className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b]">Game</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b]">Odds</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b] whitespace-nowrap">Book</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b] whitespace-nowrap">Our Win %</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b] whitespace-nowrap">Market Win %</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b]">Edge</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b] whitespace-nowrap">Profit / $100</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b] whitespace-nowrap">Bet Size</th>
            <th className="text-right px-4 py-3 text-xs font-semibold uppercase tracking-wider text-[#64748b]">Signal</th>
          </tr>
        </thead>
        <tbody>
          {bets.map((b, i) => (
            <tr key={i} className="border-b border-[#0f1729] last:border-0 hover:bg-[#0d1526] transition-colors">
              <td className="px-4 py-3.5 font-bold text-white text-base">{b.team}</td>
              <td className="px-4 py-3.5 text-[#94a3b8] text-xs">{b.matchup}</td>
              <td className="px-4 py-3.5 text-right text-[#cbd5e1] font-mono">{b.odds.toFixed(3)}</td>
              <td className="px-4 py-3.5 text-right text-[#94a3b8] text-xs">{b.bookLabel}</td>
              <td className="px-4 py-3.5 text-right text-[#cbd5e1]">{(b.model_prob * 100).toFixed(1)}%</td>
              <td className="px-4 py-3.5 text-right text-[#94a3b8]">
                {b.pin_prob !== null
                  ? `${(b.pin_prob * 100).toFixed(1)}%`
                  : `${(b.market_prob * 100).toFixed(1)}%`}
              </td>
              <td className="px-4 py-3.5 text-right font-semibold text-[#22d3ee]">
                +{(b.edge_vs_market * 100).toFixed(1)}pp
              </td>
              <td className="px-4 py-3.5 text-right font-bold text-[#06b6d4] text-base">
                ${b.ev.toFixed(0)}
              </td>
              <td className="px-4 py-3.5 text-right text-[#94a3b8]">{b.kelly_pct.toFixed(2)}%</td>
              <td className="px-4 py-3.5 text-right">
                <Signal lm={b.lm} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default async function Home() {
  const [mlbResult, nhlResult] = await Promise.allSettled([
    api.mlb.evBets(),
    api.nhl.evBets(),
  ]);

  const mlbBets = mlbResult.status === "fulfilled"
    ? (mlbResult.value.bets ?? []).map(normalizeMLB)
    : [];
  const nhlBets = nhlResult.status === "fulfilled"
    ? (nhlResult.value.bets ?? []).map(normalizeNHL)
    : [];

  const date =
    (mlbResult.status === "fulfilled" && mlbResult.value.date) ||
    (nhlResult.status === "fulfilled" && nhlResult.value.date) ||
    new Date().toISOString().slice(0, 10);

  const noPicksToday = mlbBets.length === 0 && nhlBets.length === 0;

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
              <h2 className="text-sm font-bold uppercase tracking-widest text-white">⚾ Baseball</h2>
              {mlbBets.length > 0
                ? <span className="text-sm text-[#64748b]">{mlbBets.length} pick{mlbBets.length !== 1 ? "s" : ""}</span>
                : <span className="text-sm text-[#64748b]">No picks today</span>
              }
            </div>
            {mlbBets.length > 0
              ? <EvTable bets={mlbBets} />
              : (
                <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#64748b]">
                  No qualifying MLB bets today.
                </div>
              )
            }
          </section>

          {/* Hockey */}
          <section>
            <div className="flex items-baseline gap-3 mb-4">
              <h2 className="text-sm font-bold uppercase tracking-widest text-white">🏒 Hockey</h2>
              {nhlBets.length > 0
                ? <span className="text-sm text-[#64748b]">{nhlBets.length} pick{nhlBets.length !== 1 ? "s" : ""}</span>
                : <span className="text-sm text-[#64748b]">No picks today</span>
              }
            </div>
            {nhlBets.length > 0
              ? <EvTable bets={nhlBets} />
              : (
                <div className="rounded-xl border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#64748b]">
                  No qualifying NHL bets today.
                </div>
              )
            }
          </section>

        </div>
      )}

      {/* Column explainer */}
      <div className="mt-10 pt-6 border-t border-[#1a3050]">
        <p className="text-xs font-semibold uppercase tracking-widest text-[#64748b] mb-3">How to read this</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 text-xs text-[#7f8ea3] leading-relaxed">
          <p><span className="text-[#94a3b8] font-medium">Our Win %</span> — The probability our model gives this team of winning.</p>
          <p><span className="text-[#94a3b8] font-medium">Market Win %</span> — The implied probability baked into the sportsbook&apos;s odds (vig-free).</p>
          <p><span className="text-[#22d3ee] font-medium">Edge</span> — How much higher our win % is vs. the market. This is your mathematical advantage.</p>
          <p><span className="text-[#06b6d4] font-medium">Profit / $100</span> — Expected long-term profit on every $100 bet, based on our model.</p>
          <p><span className="text-[#94a3b8] font-medium">Bet Size</span> — Suggested % of bankroll to wager (conservative, capped at 5%).</p>
          <p>
            <span className="text-[#22d3ee] font-medium">Sharp ▲</span> — Betting lines are shifting in our direction;
            professional bettors appear to agree.{" "}
            <span className="text-[#f87171] font-medium">Fading ▼</span> — Lines moving slightly against (still qualifies).
          </p>
        </div>
        <p className="text-xs text-[#374151] mt-4">
          For informational purposes only. EdgeShift picks are not gambling advice.
        </p>
      </div>

    </div>
  );
}
