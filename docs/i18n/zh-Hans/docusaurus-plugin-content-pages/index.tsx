import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from '@site/src/pages/index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">轻量、可扩展的人形机器人全身遥操作框架</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/getting-started/installation">
            快速上手
          </Link>
        </div>
      </div>
    </header>
  );
}

const features = [
  {
    title: 'Sim2Sim 与 Sim2Real',
    description: '在 MuJoCo 仿真中运行离线 BVH 动作重播，然后以最小配置改动部署到 Unitree G1 真机。',
  },
  {
    title: 'VR 遥操作',
    description: '使用 Pico 4 / Pico 4 Ultra VR 头显进行实时全身遥操作，支持全身追踪。',
  },
  {
    title: '可扩展架构',
    description: '基于协议的组件设计，提供 InputProvider、Retargeter、Controller 和 Robot 抽象接口，易于扩展和自定义。',
  },
];

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title="首页" description="轻量、可扩展的人形机器人全身遥操作框架">
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
