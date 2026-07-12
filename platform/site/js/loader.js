// Integrity-checked artifact loader.
//
// Every artifact fetch in this app goes through loadArtifact() so that a
// tampered or corrupted file under data/ is caught before it is rendered.
// The manifest (data/manifest.json) is the trust root: it lists every
// artifact's expected sha256 and byte size, produced at build time.

import { sha256Hex } from './sha256.js';

export const SUPPORTED_SCHEMA_VERSION = 1;

export class IntegrityError extends Error {
  constructor(relPath, expectedHex, actualHex) {
    super(`Integrity check failed for "${relPath}": expected ${expectedHex}, got ${actualHex}`);
    this.name = 'IntegrityError';
    this.relPath = relPath;
    this.expectedHex = expectedHex;
    this.actualHex = actualHex;
  }
}

export class ArtifactNotInManifestError extends Error {
  constructor(relPath) {
    super(`Artifact "${relPath}" is not listed in manifest.artifacts`);
    this.name = 'ArtifactNotInManifestError';
    this.relPath = relPath;
  }
}

function bufferToHex(buffer) {
  const bytes = new Uint8Array(buffer);
  let hex = '';
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

async function computeHashHex(buffer) {
  // Prefer the native Web Crypto API when available (fast, and available
  // in any secure context: https, or http on localhost).
  if (typeof window !== 'undefined' && window.crypto && window.crypto.subtle) {
    try {
      const digest = await window.crypto.subtle.digest('SHA-256', buffer);
      return bufferToHex(digest);
    } catch (err) {
      // Fall through to the pure-JS implementation below (e.g. insecure
      // origin where crypto.subtle throws/erroring instead of existing).
    }
  }
  return sha256Hex(new Uint8Array(buffer));
}

/**
 * Build a loader bound to a manifest. Returns an async function
 * (relPath, opts) -> parsed JSON (or ArrayBuffer if opts.as === 'buffer')
 * that fetches data/<relPath>, verifies its sha256 against the manifest,
 * and throws IntegrityError / ArtifactNotInManifestError / Error on failure.
 */
export function createLoader(manifest) {
  return async function loadArtifact(relPath, opts = {}) {
    const expected = manifest.artifacts && manifest.artifacts[relPath];
    if (!expected) {
      throw new ArtifactNotInManifestError(relPath);
    }

    const url = `data/${relPath}?v=${encodeURIComponent(manifest.build_version)}`;
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} fetching ${relPath}`);
    }
    const buffer = await res.arrayBuffer();

    const actualHex = await computeHashHex(buffer);
    if (actualHex !== expected.sha256) {
      throw new IntegrityError(relPath, expected.sha256, actualHex);
    }

    if (opts.as === 'buffer') return buffer;

    const text = new TextDecoder('utf-8').decode(buffer);
    return JSON.parse(text);
  };
}
