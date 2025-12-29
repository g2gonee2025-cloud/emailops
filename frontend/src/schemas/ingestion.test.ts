/** @vitest-environment jsdom */
import { describe, it, expect } from 'vitest';
import { ManualDocumentSchema, ManualUploadFormSchema, transformToPushDocuments } from './ingestion';

describe('Ingestion Schemas', () => {
  describe('ManualDocumentSchema', () => {
    it('should validate a correct document', () => {
      const doc = { text: 'Some content', metadata: '{"source": "test"}' };
      const result = ManualDocumentSchema.safeParse(doc);
      expect(result.success).toBe(true);
    });

    it('should allow an empty metadata string', () => {
      const doc = { text: 'Some content', metadata: '' };
      const result = ManualDocumentSchema.safeParse(doc);
      expect(result.success).toBe(true);
    });

    it('should fail if text is empty', () => {
      const doc = { text: '', metadata: '{"source": "test"}' };
      const result = ManualDocumentSchema.safeParse(doc);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toBe('Document text cannot be empty.');
      }
    });

    it('should fail if metadata is not valid JSON', () => {
      const doc = { text: 'Some content', metadata: '{"source": test}' }; // Invalid JSON
      const result = ManualDocumentSchema.safeParse(doc);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toBe('Metadata must be a valid JSON string.');
      }
    });
  });

  describe('ManualUploadFormSchema', () => {
    it('should validate a form with multiple correct documents', () => {
      const form = {
        documents: [
          { text: 'Doc 1', metadata: '{"source": "a"}' },
          { text: 'Doc 2', metadata: '' },
        ],
      };
      const result = ManualUploadFormSchema.safeParse(form);
      expect(result.success).toBe(true);
    });

    it('should fail if any document in the array is invalid', () => {
      const form = {
        documents: [
          { text: 'Doc 1', metadata: '{"source": "a"}' },
          { text: '', metadata: '' },
        ],
      };
      const result = ManualUploadFormSchema.safeParse(form);
      expect(result.success).toBe(false);
    });
  });

  describe('transformToPushDocuments', () => {
    it('should transform validated data into the correct API format', () => {
      const form = {
        documents: [
          { text: '  Some content  ', metadata: '{"source": "test"}' },
          { text: 'More content', metadata: '' },
        ],
      };
      const result = transformToPushDocuments(form);
      expect(result).toEqual([
        { text: 'Some content', metadata: { source: 'test' } },
        { text: 'More content', metadata: {} },
      ]);
    });

    it('should filter out documents with only whitespace text', () => {
      const form = {
        documents: [
          { text: 'Valid doc', metadata: '{}' },
          { text: '   ', metadata: '' },
        ],
      };
      const result = transformToPushDocuments(form);
      expect(result.length).toBe(1);
      expect(result[0].text).toBe('Valid doc');
    });
  });
});
