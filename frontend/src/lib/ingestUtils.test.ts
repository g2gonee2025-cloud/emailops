import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { validateIngestionManifest } from './ingestUtils';

describe('ingestionUtils', () => {
  describe('validateIngestionManifest', () => {
    it('should validate a valid manifest', () => {
      const manifest = {
        version: '1.0',
        items: [
          {
            id: '1',
            data: {
              foo: 'bar',
            },
          },
        ],
      };
      const validated = validateIngestionManifest(manifest);
      expect(validated).toEqual(manifest);
    });

    it('should throw a ZodError for an invalid manifest if id is missing', () => {
      const invalidManifest = {
        version: '1.0',
        items: [
          {
            data: 'some data',
          },
        ],
      };
      expect(() => validateIngestionManifest(invalidManifest)).toThrow(z.ZodError);
    });

    it('should throw a ZodError for an invalid manifest if data is missing', () => {
        const invalidManifest = {
          version: '1.0',
          items: [
            {
              id: '1',
            },
          ],
        };
        expect(() => validateIngestionManifest(invalidManifest)).toThrow(z.ZodError);
      });
  });
});
