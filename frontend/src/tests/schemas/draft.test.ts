/** @vitest-environment jsdom */
import { describe, it, expect } from 'vitest';
import {
  ToneSchema,
  DraftFormSchema,
  EmailDraftSchema,
  DraftEmailResponseSchema,
  validateDraftForm,
  validateDraftEmailResponse,
} from '../../schemas/draft';

describe('Draft Schemas', () => {
  // Test ToneSchema
  describe('ToneSchema', () => {
    it('should parse valid tones', () => {
      expect(ToneSchema.parse('professional')).toBe('professional');
      expect(ToneSchema.parse('friendly')).toBe('friendly');
    });

    it('should throw an error for invalid tones', () => {
      expect(() => ToneSchema.parse('invalid-tone')).toThrow();
      expect(() => ToneSchema.parse('')).toThrow();
    });
  });

  // Test DraftFormSchema
  describe('DraftFormSchema', () => {
    const validData = {
      instruction: 'Write a thank you email.',
      threadId: 'thread-123',
      tone: 'formal',
    };

    it('should parse valid form data', () => {
      const result = DraftFormSchema.safeParse(validData);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toEqual(validData);
      }
    });

    it('should use default tone if not provided', () => {
      const dataWithoutTone = { ...validData };
      delete dataWithoutTone.tone;
      const result = DraftFormSchema.safeParse(dataWithoutTone);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.tone).toBe('professional');
      }
    });

    it('should fail if instruction is empty', () => {
      const invalidData = { ...validData, instruction: '' };
      const result = DraftFormSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should allow optional threadId', () => {
        const dataWithoutThreadId = { ...validData };
        delete dataWithoutThreadId.threadId;
        const result = DraftFormSchema.safeParse(dataWithoutThreadId);
        expect(result.success).toBe(true);
      });
  });

  // Test EmailDraftSchema
  describe('EmailDraftSchema', () => {
    const validDraft = {
      subject: 'Thank You',
      body: 'This is a thank you email.',
      to: ['test@example.com'],
      cc: ['cc@example.com'],
    };

    it('should parse a valid email draft', () => {
        const result = EmailDraftSchema.safeParse(validDraft);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data).toEqual(validDraft);
        }
    });

    it('should fail if subject or body is missing', () => {
        const invalidSubject = { ...validDraft, subject: undefined };
        const invalidBody = { ...validDraft, body: undefined };
        expect(EmailDraftSchema.safeParse(invalidSubject).success).toBe(false);
        expect(EmailDraftSchema.safeParse(invalidBody).success).toBe(false);
    });
  });


  // Test DraftEmailResponseSchema
  describe('DraftEmailResponseSchema', () => {
    const validResponse = {
      correlation_id: 'corr-123',
      draft: {
        subject: 'API Response Subject',
        body: 'API response body.',
        to: ['api@example.com'],
      },
      confidence: 0.95,
      iterations: 2,
    };

    it('should parse a valid API response', () => {
        const result = DraftEmailResponseSchema.safeParse(validResponse);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data).toEqual(validResponse);
        }
    });

    it('should fail if draft is invalid', () => {
        const invalidResponse = { ...validResponse, draft: { subject: 'only subject' } };
        const result = DraftEmailResponseSchema.safeParse(invalidResponse);
        expect(result.success).toBe(false);
      });
  });


  // Test Validator Functions
  describe('Validator Functions', () => {
    it('validateDraftForm should return success for valid data', () => {
      const result = validateDraftForm({ instruction: 'Valid' });
      expect(result.success).toBe(true);
    });

    it('validateDraftForm should return error for invalid data', () => {
      const result = validateDraftForm({ instruction: '' });
      expect(result.success).toBe(false);
    });

    it('validateDraftEmailResponse should return success for valid response', () => {
        const validResponse = {
            draft: { subject: 'S', body: 'B' },
            confidence: 1,
            iterations: 1,
        }
        const result = validateDraftEmailResponse(validResponse);
        expect(result.success).toBe(true);
      });

      it('validateDraftEmailResponse should return error for invalid response', () => {
        const result = validateDraftEmailResponse({ draft: {} });
        expect(result.success).toBe(false);
      });
  });
});
