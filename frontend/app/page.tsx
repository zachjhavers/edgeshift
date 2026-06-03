import { api } from "@/lib/api";
import type { MLBEvBet, NBAEvBet, NHLEvBet } from "@/lib/api";
import PicksDisplay, { type UnifiedBet } from "@/components/PicksDisplay";

export const revalidate = 300;

function formatBook(key: string): string {
  return key
    .replace("williamhill_us", "Caesars")
    .replace("draftkings",     "DraftKings")
    .replace("fanduel",        "FanDuel")
    .replace("betmgm",         "BetMGM")
    .replace("pinnacle",       "Pinnacle");
}

function normalizeMLB(b: MLBEvBet): UnifiedBet {
  return {
    team:           b.team,
    matchup:        b.matchup,
    odds:           b.entry_odds,
    bookLabel:      formatBook(b.entry_book || ""),
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
    bookLabel:      formatBook(b.entry_book_label || b.entry_book),
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
    bookLabel:      formatBook(b.entry_book_label || b.entry_book),
    model_prob:     b.model_prob,
    pin_prob:       b.pinnacle_prob,
    market_prob:    b.market_prob,
    edge_vs_market: b.edge_vs_market,
    ev:             b.ev,
    kelly_pct:      b.kelly_pct,
    lm:             b.line_move_direction ?? 0,
  };
}

export default async function Home() {
  const today = new Date().toISOString().slice(0, 10);

  const [mlbResult, nhlResult, nbaResult, mlbTotalsResult] = await Promise.allSettled([
    api.mlb.evBets(today),
    api.nhl.evBets(today),
    api.nba.evBets(today),
    api.mlb.totalsEvBets(today),
  ]);

  const mlbBets = mlbResult.status === "fulfilled"
    ? (mlbResult.value.bets ?? []).map(normalizeMLB) : [];
  const nhlBets = nhlResult.status === "fulfilled"
    ? (nhlResult.value.bets ?? []).map(normalizeNHL) : [];
  const nbaBets = nbaResult.status === "fulfilled"
    ? (nbaResult.value.bets ?? []).map(normalizeNBA) : [];
  const mlbTotalsBets = mlbTotalsResult.status === "fulfilled"
    ? (mlbTotalsResult.value.bets ?? []) : [];

  return (
    <PicksDisplay
      date={today}
      mlbBets={mlbBets}
      nhlBets={nhlBets}
      nbaBets={nbaBets}
      mlbTotalsBets={mlbTotalsBets}
    />
  );
}
