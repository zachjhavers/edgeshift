import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EdgeShift — Daily EV Picks",
  description: "Find sports bets with a mathematical edge. Our models compare win probabilities to market odds and surface positive expected value plays every day.",
  icons: {
    icon: "/favicon.ico",
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
            <img src="/logo.png" width={28} height={28} alt="EdgeShift" className="rounded-md shrink-0" />
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
      </body>
    </html>
  );
}
