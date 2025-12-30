import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useAuth } from '../contexts/AuthContext';
import { useLogin } from '../hooks/useLogin';
import { LoginSchema, type LoginForm } from '../schemas/login';
import { ShieldCheck, Loader2, AlertCircle } from 'lucide-react';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { Label } from '../components/ui/Label';
import { Alert, AlertDescription } from '../components/ui/Alert';

export default function LoginView() {
  const { setToken } = useAuth();
  const navigate = useNavigate();
  const { loginAsync, error, isLoading, isSuccess, data } = useLogin();

  const {
    register,
    handleSubmit,
    formState: { errors },
    setError,
  } = useForm<LoginForm>({
    resolver: zodResolver(LoginSchema),
    defaultValues: {
      email: import.meta.env.DEV ? 'testuser@emailops.ai' : '',
      password: import.meta.env.DEV ? 'test' : '',
    }
  });

  const onSubmit = async (formData: LoginForm) => {
    try {
      await loginAsync([formData.email, formData.password]);
    } catch (e) {
      // error is handled by useEffect
    }
  };

  useEffect(() => {
    if (isSuccess && data) {
      setToken(data.access_token);
      navigate('/dashboard');
    }
  }, [isSuccess, data, setToken, navigate]);

  useEffect(() => {
    if (error) {
      const message = error instanceof Error ? error.message : 'An unknown error occurred';
      setError('root.serverError', { type: 'custom', message });
    }
  }, [error, setError]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#0a0a0a] p-4">
      {/* Ambient Background */}
      <div className="fixed inset-0 bg-gradient-radial from-blue-900/20 via-transparent to-transparent opacity-50 pointer-events-none" aria-hidden="true" />
      <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] pointer-events-none mix-blend-overlay" aria-hidden="true" />

      {/* Login Card */}
      <div className="relative w-full max-w-md">
        <div className="backdrop-blur-xl bg-black/40 border border-white/10 rounded-2xl p-8 shadow-2xl">
          {/* Logo */}
          <div className="flex items-center justify-center mb-8">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
              <ShieldCheck className="text-white" size={32} />
            </div>
          </div>

          <h1 className="text-3xl font-bold text-center mb-2 bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
            EmailOps Cortex
          </h1>
          <p className="text-white/40 text-center mb-8">Sign in to continue</p>

          {/* Server Error Alert */}
          {errors.root?.serverError && (
            <Alert variant="destructive" className="mb-6">
              <AlertCircle className="w-5 h-5" />
              <AlertDescription>
                {errors.root.serverError.message}
              </AlertDescription>
            </Alert>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            <div className="space-y-2">
              <Label htmlFor="email" className="text-white/60">Email</Label>
              <Input
                id="email"
                type="email"
                {...register('email')}
                autoComplete="email"
                placeholder="Enter your email"
                aria-invalid={errors.email ? "true" : "false"}
                className="bg-white/5"
              />
              {errors.email && <p className="text-red-400 text-sm mt-1">{errors.email.message}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="password"  className="text-white/60">Password</Label>
              <Input
                id="password"
                type="password"
                {...register('password')}
                autoComplete="current-password"
                placeholder="Enter your password"
                aria-invalid={errors.password ? "true" : "false"}
                 className="bg-white/5"
              />
              {errors.password && <p className="text-red-400 text-sm mt-1">{errors.password.message}</p>}
            </div>

            <Button
              type="submit"
              disabled={isLoading}
              className="w-full !mt-8 bg-blue-600 hover:bg-blue-500 text-white"
              size="lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </Button>
          </form>

        </div>
      </div>
    </div>
  );
}
