import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EdgeShift — Daily EV Picks",
  description: "Find sports bets with a mathematical edge. Our models analyze MLB, NHL, and NBA games daily and surface positive expected value plays — bets where the true win probability is higher than what the sportsbook odds imply.",
  keywords: ["sports betting", "EV betting", "expected value", "MLB picks", "NHL picks", "NBA picks", "sharp betting", "positive EV"],
  metadataBase: new URL("https://edgeshift.vercel.app"),
  authors: [{ name: "EdgeShift" }],
  openGraph: {
    title: "EdgeShift — Daily EV Picks",
    description: "Mathematical edge-finding for MLB, NHL, and NBA. Our models surface +EV bets every morning based on Pinnacle sharp lines.",
    url: "https://edgeshift.vercel.app",
    siteName: "EdgeShift",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "EdgeShift — Daily EV Picks",
    description: "Mathematical edge-finding for MLB, NHL, and NBA.",
  },
  robots: {
    index: true,
    follow: true,
  },
  icons: {
    apple: "/apple-touch-icon.png",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full flex flex-col bg-[#060a14] text-white">
        {/* Top header */}
        <header className="shrink-0 border-b border-[#1a3050] bg-[#090d1a]">
          <div className="max-w-5xl mx-auto px-4 md:px-6 py-3 flex items-center gap-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo.svg" width={32} height={32} alt="EdgeShift" className="shrink-0" />
            <div className="flex items-baseline gap-2">
              <span className="text-base font-bold text-white">
                Edge<span className="text-[#06b6d4]">Shift</span>
              </span>
              <span className="text-xs font-semibold uppercase tracking-widest text-[#22d3ee]/40">
                Daily Picks
              </span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-5xl mx-auto px-4 md:px-6 py-8">
            {children}
          </div>
        </main>
        <Analytics />
      </body>
    </html>
  );
}
