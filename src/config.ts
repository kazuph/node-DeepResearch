import { setGlobalDispatcher } from 'undici';

interface ModelConfig {
  model: string;
  temperature: number;
  fallbackModel: string;
}

interface ToolConfigs {
  dedup: ModelConfig;
  evaluator: ModelConfig;
  errorAnalyzer: ModelConfig;
  queryRewriter: ModelConfig;
  agent: ModelConfig;
  agentBeastMode: ModelConfig;
}

// プロキシサポートは削除されています。
// Bun 環境では undici の ProxyAgent が使用できないため、ここでのプロキシ設定は無効にしています。

export const GEMINI_API_KEY = process.env.GEMINI_API_KEY as string;
export const JINA_API_KEY = process.env.JINA_API_KEY as string;
export const BRAVE_API_KEY = process.env.BRAVE_API_KEY as string;
export const SEARCH_PROVIDER = BRAVE_API_KEY ? 'brave' : 'duck';

const DEFAULT_MODEL = 'gemini-2.0-flash-exp';
const FALLBACK_MODEL = 'gemini-1.5-flash';

const defaultConfig: ModelConfig = {
  model: DEFAULT_MODEL,
  temperature: 0,
  fallbackModel: FALLBACK_MODEL
};

export const modelConfigs: ToolConfigs = {
  dedup: {
    ...defaultConfig,
    temperature: 0.1
  },
  evaluator: {
    ...defaultConfig,
    model: 'gemini-exp-1206',
    temperature: 0
  },
  errorAnalyzer: {
    ...defaultConfig
  },
  queryRewriter: {
    ...defaultConfig,
    temperature: 0.1
  },
  agent: {
    ...defaultConfig,
    model: 'gemini-exp-1206',
    temperature: 0.7
  },
  agentBeastMode: {
    ...defaultConfig,
    model: 'gemini-exp-1206',
    temperature: 0.7
  }
};

export const STEP_SLEEP = 1000;

if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY not found");
if (!JINA_API_KEY) throw new Error("JINA_API_KEY not found");
