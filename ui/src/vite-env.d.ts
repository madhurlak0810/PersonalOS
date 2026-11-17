/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** The PersonalOS API's base URL; `http://127.0.0.1:8000` when unset. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
