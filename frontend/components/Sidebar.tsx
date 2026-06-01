export default function Sidebar() {
  return (
    <aside className="w-52 shrink-0 flex flex-col border-r border-[#1a3050] bg-[#090d1a] p-5">
      {/* Logo */}
      <div className="mb-8 flex items-center gap-3">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/logo.png" width={34} height={34} alt="EdgeShift logo" className="rounded-md shrink-0" />
        <div>
          <div className="text-[17px] font-bold tracking-tight text-white leading-none">
            Edge<span className="text-[#06b6d4]">Shift</span>
          </div>
          <div className="text-xs font-semibold uppercase tracking-widest text-[#22d3ee]/40 mt-0.5">
            EV Calculator
          </div>
        </div>
      </div>

      <div className="flex-1" />

      <div className="pt-4 border-t border-[#1a3050]">
        <p className="text-xs text-[#374151] leading-relaxed">
          For informational purposes only. Not gambling advice.
        </p>
      </div>
    </aside>
  );
}
