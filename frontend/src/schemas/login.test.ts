/** @vitest-environment jsdom */
import { describe, it, expect } from 'vitest';
import { LoginSchema } from './login';

describe('LoginSchema', () => {
  it('should validate a correct login object', () => {
    const validLogin = {
      email: 'test@example.com',
      password: 'password123',
    };
    const result = LoginSchema.safeParse(validLogin);
    expect(result.success).toBe(true);
  });

  it('should fail validation for an invalid email', () => {
    const invalidLogin = {
      email: 'not-an-email',
      password: 'password123',
    };
    const result = LoginSchema.safeParse(invalidLogin);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0].message).toBe('Invalid email address');
    }
  });

  it('should fail validation for an empty email', () => {
    const invalidLogin = {
      email: '',
      password: 'password123',
    };
    const result = LoginSchema.safeParse(invalidLogin);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0].message).toBe('Email is required');
    }
  });

  it('should fail validation for an empty password', () => {
    const invalidLogin = {
      email: 'test@example.com',
      password: '',
    };
    const result = LoginSchema.safeParse(invalidLogin);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0].message).toBe('Password is required');
    }
  });
});
