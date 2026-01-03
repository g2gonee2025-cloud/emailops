
import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Loader2, PenTool, Copy, Check, RefreshCw } from 'lucide-react';
import { DraftFormSchema } from '../schemas/draft';
import type { DraftForm } from '../schemas/draft';
import { useDraftEmail } from '../hooks/useDraft';
import GlassCard from '../components/ui/GlassCard';
import { Button } from '@/components/ui/Button';
import { Textarea } from '@/components/ui/Textarea';
import { Label } from '@/components/ui/Label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert';
import { Skeleton } from '@/components/ui/Skeleton';
import { ConversationSelector } from '@/components/ui/ConversationSelector';
import { useToast } from '../contexts/toastContext';

const TONES = [
  { id: 'professional', label: 'Professional' },
  { id: 'friendly', label: 'Friendly' },
  { id: 'formal', label: 'Formal' },
  { id: 'concise', label: 'Concise' },
];

export default function DraftView() {
  const [copied, setCopied] = useState(false);
  const { addToast } = useToast();
  const { mutate: draftEmail, data, error, isPending, reset: resetMutation } = useDraftEmail();

  const form = useForm<DraftForm>({
    resolver: zodResolver(DraftFormSchema),
    defaultValues: {
      instruction: '',
      threadId: '',
      tone: 'professional',
    },
  });

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    reset: resetForm,
    formState: { errors },
  } = form;

  // eslint-disable-next-line react-hooks/incompatible-library
  const tone = watch('tone');
  const draft = data?.draft;

  useEffect(() => {
    if (error) {
      addToast({
        type: 'error',
        message: 'Error Generating Draft',
        details: error.message || 'An unknown error occurred.',
      });
    }
  }, [error, addToast]);

  const onSubmit = (formData: DraftForm) => {
    draftEmail(formData);
  };

  const handleCopy = async () => {
    if (!draft) return;
    const text = `Subject: ${draft.subject}\n\n${draft.body}`;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReset = () => {
    resetForm();
    resetMutation();
  };

  const renderEmptyState = () => (
    <div className="text-center py-12">
      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 flex items-center justify-center mx-auto mb-6">
        <PenTool className="w-10 h-10 text-emerald-300" />
      </div>
      <h3 className="text-xl font-semibold mb-2">Compose with AI</h3>
      <p className="text-white/40 max-w-md mx-auto">
        Describe what you want to say, and Cortex will draft a polished email for you.
      </p>
    </div>
  );

  const renderDraft = () => (
    <div className="space-y-4 animate-slide-up">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Generated Draft</h2>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleCopy}>
            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
            {copied ? 'Copied!' : 'Copy'}
          </Button>
          <Button variant="outline" onClick={handleReset}>
            <RefreshCw className="w-4 h-4" />
            New Draft
          </Button>
        </div>
      </div>

      <GlassCard className="p-6 space-y-4">
        {/* Subject */}
        <div className="space-y-1">
          <span className="text-xs text-white/40 uppercase tracking-wider">Subject</span>
          <p className="text-lg font-medium">{draft!.subject}</p>
        </div>

        {/* Recipients */}
        {(draft!.to?.length || draft!.cc?.length) && (
          <div className="flex gap-6 text-sm">
            {draft!.to?.length && (
              <div>
                <span className="text-white/40">To: </span>
                <span>{draft!.to.join(', ')}</span>
              </div>
            )}
            {draft!.cc?.length && (
              <div>
                <span className="text-white/40">CC: </span>
                <span>{draft!.cc.join(', ')}</span>
              </div>
            )}
          </div>
        )}

        {/* Body */}
        <div className="pt-4 border-t border-white/10">
          <pre className="whitespace-pre-wrap font-sans text-white/90 leading-relaxed">
            {draft!.body}
          </pre>
        </div>
      </GlassCard>
    </div>
  );

  const renderLoading = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Skeleton className="h-8 w-32" />
        <div className="flex gap-2">
          <Skeleton className="h-10 w-24" />
          <Skeleton className="h-10 w-28" />
        </div>
      </div>
      <GlassCard className="p-6 space-y-4">
        <div className="space-y-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-6 w-3/4" />
        </div>
        <div className="space-y-2 pt-4 border-t border-white/10">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
        </div>
      </GlassCard>
    </div>
  );

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="p-6 border-b border-white/5">
        <h1 className="text-3xl font-display font-semibold tracking-tight bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">
          Draft Email
        </h1>
        <p className="text-white/40 mt-1">Generate professional email drafts using AI</p>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Input Form */}
          {!draft && !isPending && (
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6" noValidate>
              {/* Instruction */}
              <div className="space-y-2">
                <Label htmlFor="instruction" className="text-sm font-medium text-white/60">
                  What should the email say?
                </Label>
                <Textarea
                  id="instruction"
                  {...register('instruction')}
                  placeholder="e.g., Reply to John thanking him for the meeting and propose next Tuesday at 2pm..."
                  className="h-32"
                  disabled={isPending}
                />
                {errors.instruction && <p className="text-red-400 text-sm">{errors.instruction.message}</p>}
              </div>

              {/* Thread Context (Optional) */}
              <div className="space-y-2">
                <Label htmlFor="threadId" className="text-sm font-medium text-white/60">
                  Conversation (optional)
                </Label>
                <ConversationSelector
                  value={watch('threadId') || undefined}
                  onValueChange={(value) => setValue('threadId', value || '', { shouldValidate: true })}
                  placeholder="Select a conversation for context..."
                  disabled={isPending}
                />
              </div>

              {/* Tone Selector */}
              <div className="space-y-2">
                <Label className="text-sm font-medium text-white/60">Tone</Label>
                <div className="flex flex-wrap gap-2">
                  {TONES.map((t) => (
                    <Button
                      key={t.id}
                      type="button"
                      variant={tone === t.id ? 'default' : 'secondary'}
                      onClick={() => setValue('tone', t.id, { shouldValidate: true })}
                    >
                      {t.label}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Submit */}
              <Button type="submit" disabled={isPending} className="w-full py-6">
                {isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <PenTool className="w-5 h-5 mr-2" />
                    Generate Draft
                  </>
                )}
              </Button>
            </form>
          )}

          {/* Error */}
          {error && !isPending && (
             <Alert variant="destructive">
               <AlertTitle>Error</AlertTitle>
               <AlertDescription>
                 {error.message || 'An unexpected error occurred. Please try again.'}
               </AlertDescription>
             </Alert>
          )}

          {/* Loading Skeleton */}
          {isPending && renderLoading()}

          {/* Generated Draft */}
          {draft && !isPending && renderDraft()}

          {/* Empty State */}
          {!draft && !isPending && !watch('instruction') && renderEmptyState()}
        </div>
      </div>
    </div>
  );
}
