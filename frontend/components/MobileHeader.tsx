export default function MobileHeader() {
  return (
    <header className="md:hidden shrink-0 border-b border-[#1a3050] bg-[#090d1a]">
      <div className="flex items-center gap-2 px-4 py-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/logo.png" width={24} height={24} alt="EdgeShift logo" className="rounded shrink-0" />
        <span className="text-sm font-bold text-white">
          Edge<span className="text-[#06b6d4]">Shift</span>
        </span>
        <span className="text-xs text-[#22d3ee]/40 font-semibold uppercase tracking-widest ml-1">
          EV Calculator
        </span>
      </div>
    </header>
  );
}
