import { describe, it, expect } from 'vitest';

import { validateIngestionManifest, DocumentSchema, ManifestSchema, validateManifest } from './ingestUtils';

describe('Ingestion Utils', () => {
  describe('DocumentSchema', () => {
    it('should validate a correct document object', () => {
      const doc = { id: 'doc1', path: '/path/to/doc' };
      expect(() => DocumentSchema.parse(doc)).not.toThrow();
    });

    it('should fail validation if id is empty', () => {
      const doc = { id: '', path: '/path/to/doc' };
      expect(() => DocumentSchema.parse(doc)).toThrow();
    });

    it('should fail validation if path is empty', () => {
      const doc = { id: 'doc1', path: '' };
      expect(() => DocumentSchema.parse(doc)).toThrow();
    });
  });

  describe('ManifestSchema', () => {
    it('should validate a correct manifest object', () => {
      const manifest = {
        documents: [
          { id: 'doc1', path: '/path/to/doc1' },
          { id: 'doc2', path: '/path/to/doc2' },
        ],
      };
      expect(() => ManifestSchema.parse(manifest)).not.toThrow();
    });

    it('should fail validation if documents array is malformed', () => {
      const manifest = {
        documents: [
          { id: 'doc1', path: '/path/to/doc1' },
          { id: 'doc2', path: '' },
        ],
      };
      expect(() => ManifestSchema.parse(manifest)).toThrow();
    });
  });

  describe('validateManifest', () => {
    it('should return the manifest if it is valid', () => {
      const manifest = {
        documents: [{ id: 'doc1', path: '/path/to/doc1' }],
      };
      const validatedManifest = validateManifest(manifest);
      expect(validatedManifest).toEqual(manifest);
    });

    it('should throw an error if the manifest is invalid', () => {
      const manifest = {
        documents: [{ id: 'doc1', path: '' }],
      };
      expect(() => validateManifest(manifest)).toThrow();
    });
  });

  describe('validateIngestionManifest (legacy)', () => {
    it('should validate a valid manifest with version', () => {
      const manifest = {
        version: '1.0',
        documents: [{ id: '1', path: '/path/to' }],
      };
      const validated = validateIngestionManifest(manifest);
      expect(validated).toEqual(manifest);
    });
  });
});
