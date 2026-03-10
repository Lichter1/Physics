export default function Tooltip({ text, children }) {
  return (
    <span className="relative group inline-flex items-center">
      {children}
      <span className="ml-1 text-gray-600 cursor-help text-xs">&#9432;</span>
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 border border-dark-border rounded text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10 max-w-xs">
        {text}
      </span>
    </span>
  )
}
