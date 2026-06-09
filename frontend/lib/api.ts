// Server-side only — never exposed to the browser bundle
const BASE = process.env.API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  mlb: {
    evBets: (date?: string) =>
      get<MLBEvBetsResponse>(`/api/mlb/ev-bets${date ? `?date=${date}` : ""}`),
    totalsEvBets: (date?: string) =>
      get<MLBTotalsEvBetsResponse>(`/api/mlb/totals-ev-bets${date ? `?date=${date}` : ""}`),
  },
  nhl: {
    evBets: (date?: string) =>
      get<NHLEvBetsResponse>(`/api/nhl/ev-bets${date ? `?date=${date}` : ""}`),
  },
  nba: {
    evBets: (date?: string) =>
      get<NBAEvBetsResponse>(`/api/nba/ev-bets${date ? `?date=${date}` : ""}`),
  },
  soccer: {
    evBets: (date?: string) =>
      get<SoccerEvBetsResponse>(`/api/soccer/ev-bets${date ? `?date=${date}` : ""}`),
  },
  health: () => get<{ status: string }>("/api/health"),
};

// ── MLB ───────────────────────────────────────────────────────────────────────

export interface MLBEvBet {
  date:                  string;
  matchup:               string;
  side:                  "home" | "away";
  team:                  string;
  model_prob:            number;
  market_prob:           number;
  pinnacle_prob:         number | null;
  edge_vs_market:        number;
  entry_odds:            number;
  entry_book:            string;
  ev:                    number;
  kelly_pct:             number;
  line_move_direction:   number;
  line_move_label:       string | null;
  closing_pinnacle_odds: number | null;
  clv_pct:               number | null;
  result:                string;
}

export interface MLBEvBetsResponse {
  date:  string | null;
  total: number;
  bets:  MLBEvBet[];
}

export interface MLBTotalsEvBet {
  date:                string;
  matchup:             string;
  side:                "over" | "under";
  label:               string;
  total_line:          number;
  predicted_total:     number;
  model_prob:          number;
  market_prob:         number;
  pinnacle_prob:       number | null;
  edge_vs_market:      number;
  entry_odds:          number;
  entry_book:          string;
  ev:                  number;
  kelly_pct:           number;
  line_move_direction: number;
  result:              string;
}

export interface MLBTotalsEvBetsResponse {
  date:  string | null;
  total: number;
  bets:  MLBTotalsEvBet[];
}

// ── NHL ───────────────────────────────────────────────────────────────────────

export interface NHLEvBet {
  date:                  string;
  matchup:               string;
  side:                  "home" | "away";
  team:                  string;
  model_prob:            number;
  market_prob:           number;
  pinnacle_prob:         number | null;
  edge_vs_market:        number;
  odds:                  number;
  entry_book:            string;
  entry_book_label:      string;
  ev:                    number;
  kelly_pct:             number;
  line_move_direction:   number;
  line_move_label:       string | null;
  closing_pinnacle_odds: number | null;
  clv_pct:               number | null;
  result:                string;
}

export interface NHLEvBetsResponse {
  date:  string | null;
  total: number;
  bets:  NHLEvBet[];
}

// ── NBA ───────────────────────────────────────────────────────────────────────

export interface NBAEvBet {
  date:                  string;
  matchup:               string;
  side:                  "home" | "away";
  team:                  string;
  game_type:             string;
  model_prob:            number;
  market_prob:           number;
  pinnacle_prob:         number | null;
  edge_vs_market:        number;
  odds:                  number;
  entry_book:            string;
  entry_book_label:      string;
  ev:                    number;
  kelly_pct:             number;
  line_move_direction:   number;
  line_move_label:       string | null;
  closing_pinnacle_odds: number | null;
  clv_pct:               number | null;
  result:                string;
}

export interface NBAEvBetsResponse {
  date:  string | null;
  total: number;
  bets:  NBAEvBet[];
}

// ── Soccer ────────────────────────────────────────────────────────────────────

export interface SoccerEvBet {
  date:           string;
  matchup:        string;
  market:         "h2h" | "totals";
  side:           string;
  label:          string;
  model_prob:     number;
  market_prob:    number;
  pinnacle_prob:  number;
  edge_vs_market: number;
  entry_odds:     number;
  entry_book:     string;
  ev:             number;
  kelly_pct:      number;
  result:         string;
}

export interface SoccerEvBetsResponse {
  date:  string | null;
  total: number;
  bets:  SoccerEvBet[];
}
