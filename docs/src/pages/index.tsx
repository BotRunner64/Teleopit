import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/getting-started/installation">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

const features = [
  {
    title: 'Sim2Sim & Sim2Real',
    description: 'Run offline BVH playback in MuJoCo simulation, then deploy the same pipeline to Unitree G1 hardware with minimal configuration changes.',
  },
  {
    title: 'VR Teleoperation',
    description: 'Real-time whole-body teleoperation using Pico 4 / Pico 4 Ultra VR headsets with full body tracking support.',
  },
  {
    title: 'Extensible Architecture',
    description: 'Protocol-based component design with InputProvider, Retargeter, Controller, and Robot abstractions. Easy to extend and customize.',
  },
];

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="Home" description={siteConfig.tagline}>
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {features.map((feature, idx) => (
                <div key={idx} className={clsx('col col--4')}>
                  <div className="text--center padding-horiz--md padding-vert--lg">
                    <Heading as="h3">{feature.title}</Heading>
                    <p>{feature.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
