/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js,ts,vue}"],
  theme: {
    extend: {},
    colors: {
      "grad1-1": "#3a6186",
      "grad1-2": "#89253e",
      "black-1": "#222831",
      "black-2": "#393E46",
      "white-1": "#EEEEEE",
      "white-2": "#FFFFFF"
    },
    fontFamily: {
      sans: "Sofia Sans Condensed",
      serif: "Castoro Titling"
    }
  },
  plugins: []
}
