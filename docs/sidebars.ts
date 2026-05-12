import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/download-assets',
        'getting-started/quick-start',
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/offline-sim2sim',
        'tutorials/pico4-vr',
        'tutorials/sim2real',
        'tutorials/training',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      items: [
        'configuration/overview',
        'configuration/config-reference',
        'configuration/faq',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/architecture',
        'reference/assets',
        'reference/dataset',
        'reference/g1-bridge-sdk',
        'reference/training-troubleshooting',
      ],
    },
    'contributing',
  ],
};

export default sidebars;
