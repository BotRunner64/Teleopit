import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Teleopit',
  tagline: 'Lightweight, extensible whole-body teleoperation framework for humanoid robots',
  favicon: 'img/favicon.ico',

  url: 'https://BotRunner64.github.io',
  baseUrl: '/Teleopit/',

  organizationName: 'BotRunner64',
  projectName: 'Teleopit',

  onBrokenLinks: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: '/',
          editUrl: 'https://github.com/BotRunner64/Teleopit/tree/master/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'Teleopit',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/BotRunner64/Teleopit',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Getting Started', to: '/getting-started/installation'},
            {label: 'Tutorials', to: '/tutorials/offline-sim2sim'},
            {label: 'Configuration', to: '/configuration/overview'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'GitHub', href: 'https://github.com/BotRunner64/Teleopit'},
            {label: 'ModelScope Models', href: 'https://modelscope.cn/models/BingqianWu/Teleopit-models'},
          ],
        },
      ],
      copyright: `Copyright ${new Date().getFullYear()} Teleopit Contributors. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'yaml', 'python'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
