import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "EdgeShift — Daily EV Picks";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          background: "#060a14",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "system-ui, sans-serif",
          position: "relative",
        }}
      >
        {/* Subtle grid background */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage:
              "linear-gradient(rgba(6,182,212,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.04) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />

        {/* Logo mark */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#0a0f1e",
            border: "2px solid #1a3050",
            borderRadius: "20px",
            width: "96px",
            height: "96px",
            marginBottom: "32px",
          }}
        >
          {/* SVG chart icon inline */}
          <svg width="56" height="56" viewBox="0 0 32 32" fill="none">
            <path
              d="M4 17 L10 11 L16 14 L22 8"
              stroke="#06b6d4"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M19 6.5 L23 8 L21.5 12"
              stroke="#06b6d4"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M4 21 L10 15 L16 18 L22 12"
              stroke="#22d3ee"
              strokeWidth="1.5"
              strokeOpacity={0.3}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <line
              x1="4" y1="25" x2="28" y2="25"
              stroke="#22d3ee"
              strokeWidth="1"
              strokeOpacity={0.15}
              strokeLinecap="round"
            />
          </svg>
        </div>

        {/* Wordmark */}
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: "4px",
            marginBottom: "16px",
          }}
        >
          <span style={{ fontSize: "72px", fontWeight: 800, color: "#ffffff", letterSpacing: "-2px" }}>
            Edge
          </span>
          <span style={{ fontSize: "72px", fontWeight: 800, color: "#06b6d4", letterSpacing: "-2px" }}>
            Shift
          </span>
        </div>

        {/* Tagline */}
        <div
          style={{
            fontSize: "24px",
            color: "#64748b",
            letterSpacing: "4px",
            textTransform: "uppercase",
            fontWeight: 600,
            marginBottom: "48px",
          }}
        >
          Daily EV Picks
        </div>

        {/* Sport tags */}
        <div style={{ display: "flex", gap: "16px" }}>
          {["MLB", "NHL", "NBA"].map((sport) => (
            <div
              key={sport}
              style={{
                background: "#0a0f1e",
                border: "1px solid #1a3050",
                borderRadius: "8px",
                padding: "8px 20px",
                fontSize: "16px",
                fontWeight: 700,
                color: "#94a3b8",
                letterSpacing: "2px",
              }}
            >
              {sport}
            </div>
          ))}
        </div>
      </div>
    ),
    { ...size }
  );
}
