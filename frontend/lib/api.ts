// Server-side only — never exposed to the browser bundle
const BASE = process.env.API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const opts = process.env.NODE_ENV === "development"
    ? { cache: "no-store" as const }
    : { next: { revalidate: 300 } };
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  mlb: {
    evBets: (date?: string) =>
      get<MLBEvBetsResponse>(`/api/mlb/ev-bets${date ? `?date=${date}` : ""}`),
  },
  nhl: {
    evBets: (date?: string) =>
      get<NHLEvBetsResponse>(`/api/nhl/ev-bets${date ? `?date=${date}` : ""}`),
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
