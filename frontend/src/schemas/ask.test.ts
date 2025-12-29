/** @vitest-environment jsdom */

import { describe, it, expect } from 'vitest';
import { v4 as uuidv4 } from 'uuid';
import {
  SourceSchema,
  MessageSchema,
  AskFormSchema,
  validateMessage,
  validateAskForm,
} from './ask';

describe('Ask Schemas', () => {
  // ============================================================================
  // Test SourceSchema
  // ============================================================================
  describe('SourceSchema', () => {
    const validSource = {
      chunk_id: uuidv4(),
      content: 'This is a source.',
      score: 0.85,
    };

    it('should validate a correct source object', () => {
      const result = SourceSchema.safeParse(validSource);
      expect(result.success).toBe(true);
    });

    it('should fail validation for an invalid chunk_id', () => {
      const invalidSource = { ...validSource, chunk_id: 'not-a-uuid' };
      const result = SourceSchema.safeParse(invalidSource);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toBe('Invalid chunk ID');
      }
    });

    it('should fail validation for a score less than 0', () => {
      const invalidSource = { ...validSource, score: -0.1 };
      const result = SourceSchema.safeParse(invalidSource);
      expect(result.success).toBe(false);
    });

    it('should fail validation for a score greater than 1', () => {
      const invalidSource = { ...validSource, score: 1.1 };
      const result = SourceSchema.safeParse(invalidSource);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toBe('Score must be between 0 and 1');
      }
    });
  });

  // ============================================================================
  // Test MessageSchema
  // ============================================================================
  describe('MessageSchema', () => {
    it('should validate a correct user message', () => {
      const validUserMessage = {
        role: 'user',
        content: 'Hello, Cortex!',
        timestamp: new Date(),
      };
      const result = validateMessage(validUserMessage);
      expect(result.success).toBe(true);
    });

    it('should validate a correct assistant message with sources', () => {
      const validAssistantMessage = {
        role: 'assistant',
        content: 'Hello! Here are your sources.',
        sources: [
          {
            chunk_id: uuidv4(),
            content: 'Source 1 content.',
            score: 0.9,
          },
        ],
        timestamp: new Date(),
      };
      const result = validateMessage(validAssistantMessage);
      expect(result.success).toBe(true);
    });

    it('should fail validation for an invalid role', () => {
      const invalidMessage = {
        role: 'admin', // Invalid role
        content: 'Test',
        timestamp: new Date(),
      };
      const result = validateMessage(invalidMessage);
      expect(result.success).toBe(false);
    });

    it('should fail if sources array contains invalid data', () => {
      const invalidMessage = {
        role: 'assistant',
        content: 'Here are invalid sources.',
        sources: [{ chunk_id: 'not-a-uuid', content: 'bad source', score: 99 }],
        timestamp: new Date(),
      };
      const result = validateMessage(invalidMessage);
      expect(result.success).toBe(false);
    });
  });

  // ============================================================================
  // Test AskFormSchema
  // ============================================================================
  describe('AskFormSchema', () => {
    it('should validate correct form input', () => {
      const result = validateAskForm({ input: 'What is the status of Project X?' });
      expect(result.success).toBe(true);
    });

    it('should trim whitespace from input', () => {
      const result = validateAskForm({ input: '  Leading and trailing whitespace  ' });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.input).toBe('Leading and trailing whitespace');
      }
    });

    it('should fail validation for empty input', () => {
      const result = validateAskForm({ input: '' });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toBe('Message cannot be empty');
      }
    });

    it('should fail validation for input with only whitespace', () => {
      const result = validateAskForm({ input: '   ' });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].message).toBe('Message cannot be empty');
      }
    });
  });
});
