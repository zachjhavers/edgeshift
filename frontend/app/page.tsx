import { api } from "@/lib/api";
import type { MLBEvBet, NHLEvBet } from "@/lib/api";

export const revalidate = 300;

const LM_ICON: Record<number, string>  = { 1: "↑", [-1]: "↓", 0: "—" };
const LM_COLOR: Record<number, string> = {
  1:    "text-[#22d3ee]",
  [-1]: "text-[#f87171]",
  0:    "text-[#4b5563]",
};

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

function EvTable({ bets }: { bets: UnifiedBet[] }) {
  return (
    <div className="rounded-lg border border-[#1a3050] bg-[#0a0f1e] overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#1a3050] text-xs text-[#4b5563] uppercase tracking-wider">
            <th className="text-left px-4 py-3">Team</th>
            <th className="text-left px-4 py-3">Matchup</th>
            <th className="text-right px-4 py-3">Best Odds</th>
            <th className="text-right px-4 py-3 whitespace-nowrap">At Book</th>
            <th className="text-right px-4 py-3">Model%</th>
            <th className="text-right px-4 py-3 whitespace-nowrap">Pin. Mkt%</th>
            <th className="text-right px-4 py-3">Edge</th>
            <th className="text-right px-4 py-3">EV/100</th>
            <th className="text-right px-4 py-3">Kelly</th>
            <th className="text-right px-4 py-3 whitespace-nowrap" title="Line movement vs opening Pinnacle">LM</th>
          </tr>
        </thead>
        <tbody>
          {bets.map((b, i) => {
            const lm = b.lm;
            return (
              <tr key={i} className="border-b border-[#0f1729] hover:bg-[#0f1729] transition-colors">
                <td className="px-4 py-3 font-semibold text-white">{b.team}</td>
                <td className="px-4 py-3 text-[#6b7280] text-xs">{b.matchup}</td>
                <td className="px-4 py-3 text-right text-[#94a3b8]">{b.odds.toFixed(3)}</td>
                <td className="px-4 py-3 text-right text-[#6b7280] text-xs">{b.bookLabel}</td>
                <td className="px-4 py-3 text-right text-[#94a3b8]">{(b.model_prob * 100).toFixed(1)}%</td>
                <td className="px-4 py-3 text-right text-[#94a3b8]">
                  {b.pin_prob !== null
                    ? `${(b.pin_prob * 100).toFixed(1)}%`
                    : `${(b.market_prob * 100).toFixed(1)}%`}
                </td>
                <td className="px-4 py-3 text-right text-[#22d3ee]">
                  +{(b.edge_vs_market * 100).toFixed(1)}pp
                </td>
                <td className="px-4 py-3 text-right font-semibold text-[#06b6d4]">${b.ev.toFixed(0)}</td>
                <td className="px-4 py-3 text-right text-[#6b7280]">{b.kelly_pct.toFixed(2)}%</td>
                <td className={`px-4 py-3 text-right font-mono ${LM_COLOR[lm] ?? "text-[#4b5563]"}`}>
                  {LM_ICON[lm] ?? "—"}
                </td>
              </tr>
            );
          })}
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
    <div className="max-w-5xl">

      {/* Header */}
      <div className="mb-10 pb-6 border-b border-[#1a3050]">
        <p className="text-xs font-bold uppercase tracking-widest text-[#22d3ee]/60 mb-1">
          {date}
        </p>
        <h1 className="text-3xl font-bold text-white mb-2">Today's Picks</h1>
        <p className="text-sm text-[#6b7280]">
          Positive expected value bets identified by our XGBoost models.
          Pinnacle vig-free edge 4–7pp · EV threshold $15/100 · best odds across all books.
        </p>
      </div>

      {noPicksToday ? (
        <div className="rounded-lg border border-[#1a3050] bg-[#0a0f1e] p-16 text-center">
          <div className="text-white font-semibold text-lg mb-2">No picks today</div>
          <div className="text-sm text-[#6b7280]">
            No qualifying bets found across MLB or NHL.
            Check back tomorrow morning — picks update daily.
          </div>
        </div>
      ) : (
        <div className="space-y-12">

          {/* Baseball */}
          <section>
            <div className="flex items-baseline gap-3 mb-4">
              <h2 className="text-xs font-bold uppercase tracking-widest text-white">⚾ Baseball</h2>
              {mlbBets.length > 0
                ? <span className="text-xs text-[#4b5563]">{mlbBets.length} pick{mlbBets.length !== 1 ? "s" : ""} today</span>
                : <span className="text-xs text-[#4b5563]">No picks today</span>
              }
            </div>
            {mlbBets.length > 0
              ? <EvTable bets={mlbBets} />
              : (
                <div className="rounded-lg border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#4b5563]">
                  No qualifying MLB bets today.
                </div>
              )
            }
          </section>

          {/* Hockey */}
          <section>
            <div className="flex items-baseline gap-3 mb-4">
              <h2 className="text-xs font-bold uppercase tracking-widest text-white">🏒 Hockey</h2>
              {nhlBets.length > 0
                ? <span className="text-xs text-[#4b5563]">{nhlBets.length} pick{nhlBets.length !== 1 ? "s" : ""} today</span>
                : <span className="text-xs text-[#4b5563]">No picks today</span>
              }
            </div>
            {nhlBets.length > 0
              ? <EvTable bets={nhlBets} />
              : (
                <div className="rounded-lg border border-[#1a3050] bg-[#0a0f1e] p-8 text-center text-sm text-[#4b5563]">
                  No qualifying NHL bets today.
                </div>
              )
            }
          </section>

        </div>
      )}

      {/* Legend + footer */}
      <div className="mt-10 pt-6 border-t border-[#1a3050] space-y-1">
        <p className="text-xs text-[#4b5563]">
          <span className="text-[#22d3ee]">↑</span> line moved confirming our model ·{" "}
          <span className="text-[#f87171]">↓</span> line moved against (already filtered ≥3pp) ·{" "}
          <span className="text-[#4b5563]">—</span> neutral ·{" "}
          <strong className="text-[#6b7280]">Pin. Mkt%</strong> = Pinnacle vig-free probability
        </p>
        <p className="text-xs text-[#374151]">
          For informational purposes only. EdgeShift picks are not gambling advice.
        </p>
      </div>

    </div>
  );
}
