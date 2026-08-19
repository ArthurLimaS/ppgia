// @ts-check
const config = {
  semi: true,
  singleQuote: false,
  tabWidth: 4,
  printWidth: 80,

  overrides: [
    {
      // Telas, marcações e documentações ganham mais espaço horizontal
      files: [
        "**/*.tsx",
        "**/*.jsx",
        "**/*.json",
        "**/*.html",
        "**/*.md",
        "**/*.yml",
      ],
      options: {
        tabWidth: 2,
        printWidth: 120,
      },
    },
  ],
};

export default config;
