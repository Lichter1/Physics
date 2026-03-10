/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0f1117',
          card: '#1a1d29',
          border: '#2a2d3a',
          hover: '#252836',
        },
        bullish: '#22c55e',
        bearish: '#ef4444',
        neutral: '#eab308',
        alert: '#f97316',
      }
    },
  },
  plugins: [],
}
