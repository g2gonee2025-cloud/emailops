
import { useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { AppConfig, ConfigSchema } from '../../schemas/admin';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Label } from '../ui/Label';
import { Eye, EyeOff, Save } from 'lucide-react';
import { cn } from '../../lib/utils';

interface ConfigPanelProps {
  config: AppConfig;
  onSave: (newConfig: AppConfig) => void;
  isLoading: boolean;
}

const isSensitiveField = (key: string) => {
  const lowerKey = key.toLowerCase();
  return lowerKey.includes('key') || lowerKey.includes('secret') || lowerKey.includes('token');
};

export default function ConfigPanel({ config, onSave, isLoading }: ConfigPanelProps) {
  const [revealedFields, setRevealedFields] = useState<Record<string, boolean>>({});

  const {
    control,
    handleSubmit,
    formState: { errors, isDirty },
  } = useForm<AppConfig>({
    resolver: zodResolver(ConfigSchema),
    defaultValues: config,
  });

  const toggleReveal = (key: string) => {
    setRevealedFields(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const onSubmit = (data: AppConfig) => {
    onSave(data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      <div className="space-y-4">
        {Object.entries(config).map(([key, value]) => {
          const isSensitive = isSensitiveField(key);
          const isRevealed = revealedFields[key];
          const fieldError = errors[key as keyof AppConfig];
          const valueType = typeof value;

          return (
            <div key={key} className="flex flex-col space-y-2">
              <Label htmlFor={key} className="capitalize text-white/60">
                {key.replace(/_/g, ' ')}
              </Label>
              <div className="relative">
                <Controller
                  name={key as keyof AppConfig}
                  control={control}
                  render={({ field }) => (
                    <Input
                      {...field}
                      id={key}
                      type={isSensitive && !isRevealed ? 'password' : valueType === 'number' ? 'number' : 'text'}
                      className={cn('font-mono', fieldError && 'border-red-500')}
                      value={field.value ?? ""}
                      onChange={(e) => field.onChange(e.target.value)}
                    />
                  )}
                />
                {isSensitive && (
                  <button
                    type="button"
                    onClick={() => toggleReveal(key)}
                    className="absolute inset-y-0 right-0 flex items-center px-3 text-white/40 hover:text-white"
                  >
                    {isRevealed ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                )}
              </div>
              {fieldError && <p className="text-sm text-red-400">{fieldError.message}</p>}
            </div>
          );
        })}
      </div>
      <div className="flex justify-end">
        <Button type="submit" disabled={!isDirty || isLoading}>
          <Save className="w-4 h-4 mr-2" />
          {isLoading ? 'Saving...' : 'Save Changes'}
        </Button>
      </div>
    </form>
  );
}
