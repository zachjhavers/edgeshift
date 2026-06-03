import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: 7,
          background: "#0a0f1e",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg width="24" height="24" viewBox="0 0 32 32" fill="none">
          {/* Market line — faint, lower */}
          <path
            d="M4 21 L10 15 L16 18 L22 12"
            stroke="#22d3ee"
            strokeWidth="1.5"
            strokeOpacity={0.3}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {/* Edge line — solid, above market */}
          <path
            d="M4 17 L10 11 L16 14 L22 8"
            stroke="#06b6d4"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {/* Arrowhead */}
          <path
            d="M19 6.5 L23 8 L21.5 12"
            stroke="#06b6d4"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {/* Baseline */}
          <line
            x1="4" y1="25" x2="28" y2="25"
            stroke="#22d3ee"
            strokeWidth="1"
            strokeOpacity={0.15}
            strokeLinecap="round"
          />
        </svg>
      </div>
    ),
    { ...size }
  );
}
