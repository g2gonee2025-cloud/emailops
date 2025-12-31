
import { useState, useEffect } from 'react';
import GlassCard from '../components/ui/GlassCard';
import { StatusIndicator } from '../components/ui/StatusIndicator';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Textarea } from '../components/ui/Textarea';
import { api } from '../lib/api';
import type { S3Folder } from '../lib/api';
import type { IngestStatusResponse, PushDocument } from '../lib/api';
import { cn } from '../lib/utils';
import {
  FolderOpen,
  Play,
  RefreshCw,
  Loader2,
  Clock,
  Database,
  FileText,
  Sparkles,
  Upload,
  Plus,
  Trash2,
  CheckCircle
} from 'lucide-react';

interface ActiveJob {
  jobId: string;
  status: IngestStatusResponse;
}

// Manual Upload Section Component
function UploadSection() {
  const [documents, setDocuments] = useState<{ text: string; metadata: string }[]>([{ text: '', metadata: '' }]);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);

  const addDocument = () => {
    setDocuments(prev => [...prev, { text: '', metadata: '' }]);
  };

  const removeDocument = (index: number) => {
    setDocuments(prev => prev.filter((_, i) => i !== index));
  };

  const updateDocument = (index: number, field: 'text' | 'metadata', value: string) => {
    setDocuments(prev => prev.map((doc, i) => i === index ? { ...doc, [field]: value } : doc));
  };

  const handleUpload = async () => {
    const validDocs = documents.filter(d => d.text.trim());
    if (validDocs.length === 0) return;

    setIsUploading(true);
    setResult(null);

    try {
      const pushDocs: PushDocument[] = validDocs.map((d, i) => {
        try {
          return {
            text: d.text.trim(),
            metadata: d.metadata ? JSON.parse(d.metadata) : {},
          };
        } catch (_e) {
          // Note: Using (i + 1) to present a 1-based index to the user
          throw new Error(`Invalid JSON in metadata for Document ${i + 1}.`);
        }
      });

      const response = await api.pushDocuments(pushDocs);
      setResult({ success: true, message: `Ingested ${response.documents_ingested} documents, created ${response.chunks_created} chunks` });
      setDocuments([{ text: '', metadata: '' }]);
    } catch (err) {
      // Sanitize error output to user
      if (err instanceof Error && err.message.startsWith('Invalid JSON')) {
        setResult({ success: false, message: err.message });
      } else {
        setResult({ success: false, message: 'An unexpected error occurred during upload.' });
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <section>
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Upload className="w-5 h-5 text-emerald-300" />
        Manual Upload
      </h2>
      <GlassCard className="p-5 space-y-4">
        {documents.map((doc, i) => (
          <div key={i} className="space-y-2 pb-4 border-b border-white/5 last:border-0 last:pb-0">
            <div className="flex items-center justify-between">
              <span className="text-sm text-white/40">Document {i + 1}</span>
              {documents.length > 1 && (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => removeDocument(i)}
                  className="text-red-400 hover:text-red-300"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              )}
            </div>
            <Textarea
              value={doc.text}
              onChange={(e) => updateDocument(i, 'text', e.target.value)}
              placeholder="Document text content..."
              className="h-24 bg-white/5 border-white/10 focus-visible:ring-emerald-500/30 text-white placeholder-white/30 resize-none"
            />
            <Input
              type="text"
              value={doc.metadata}
              onChange={(e) => updateDocument(i, 'metadata', e.target.value)}
              placeholder='Optional metadata JSON, e.g. {"source": "manual"}'
              className="bg-white/5 border-white/10 focus-visible:ring-emerald-500/30 text-xs text-white placeholder-white/30 font-mono"
            />
          </div>
        ))}

        <div className="flex gap-2">
          <Button
            onClick={addDocument}
            variant="glass"
            className="gap-2 text-sm"
          >
            <Plus className="w-4 h-4" /> Add Document
          </Button>
          <Button
            onClick={handleUpload}
            disabled={isUploading || !documents.some(d => d.text.trim())}
            className="ml-auto gap-2 text-sm font-medium"
          >
            {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            Upload
          </Button>
        </div>

        {result && (
          <div className={cn(
            "p-3 rounded-lg text-sm flex items-center gap-2",
            result.success ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
          )}>
            {result.success && <CheckCircle className="w-4 h-4" />}
            {result.message}
          </div>
        )}
      </GlassCard>
    </section>
  );
}

export default function IngestionView() {
  const [folders, setFolders] = useState<S3Folder[]>([]);
  const [isLoadingFolders, setIsLoadingFolders] = useState(false);
  const [isStartingJob, setIsStartingJob] = useState(false);
  const [ingestionError, setIngestionError] = useState<string | null>(null);
  const [activeJobs, setActiveJobs] = useState<ActiveJob[]>([]);
  const [dryRun, setDryRun] = useState(true);

  const loadFolders = async () => {
    setIsLoadingFolders(true);
    try {
      const response = await api.listS3Folders('Outlook/', 100);
      setFolders(response.folders);
    } catch (error) {
      console.error('Failed to list folders:', error);
    } finally {
      setIsLoadingFolders(false);
    }
  };

  useEffect(() => {
    loadFolders();
  }, []);

  // Poll active jobs
  useEffect(() => {
    const jobsStillRunning = activeJobs.some(
      (job) => job.status.status !== 'completed' && job.status.status !== 'failed'
    );

    if (!jobsStillRunning) return;

    const interval = setInterval(async () => {
      const updatedJobs = await Promise.all(
        activeJobs.map(async (job) => {
          if (job.status.status === 'completed' || job.status.status === 'failed') {
            return job; // Stop polling for this job
          }
          try {
            const status = await api.getIngestionStatus(job.jobId);
            return { jobId: job.jobId, status };
          } catch {
            return job; // Keep current state on poll failure
          }
        })
      );
      setActiveJobs(updatedJobs);
    }, 2000);

    return () => clearInterval(interval);
  }, [activeJobs]);

  const startIngestion = async () => {
    setIsStartingJob(true);
    setIngestionError(null);
    try {
      const response = await api.startIngestion('Outlook/', undefined, dryRun);
      if (!dryRun) {
        setActiveJobs(prev => [...prev, {
          jobId: response.job_id,
          status: {
            job_id: response.job_id,
            status: 'started',
            folders_processed: 0,
            threads_created: 0,
            chunks_created: 0,
            embeddings_generated: 0,
            errors: 0,
            skipped: 0,
            message: response.message,
          }
        }]);
      }
    } catch (error) {
      // Sanitize potential PII from backend error messages
      setIngestionError('Failed to start ingestion. Please check console for details.');
      console.error('Failed to start ingestion:', error);
    } finally {
      setIsStartingJob(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'healthy';
      case 'running': return 'warning';
      case 'failed': return 'critical';
      default: return 'inactive';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="p-6 border-b border-white/5">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-display font-semibold tracking-tight bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">
              Ingestion Pipeline
            </h1>
            <p className="text-white/40 mt-1">Manage data ingestion from S3/Spaces</p>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-white/60">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
                className="rounded border-white/20 bg-white/5 text-emerald-400 focus:ring-emerald-500/50"
              />
              Dry Run
            </label>
            <Button
              onClick={startIngestion}
              disabled={isStartingJob}
              className="gap-2 font-medium"
            >
              {isStartingJob ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {dryRun ? 'Preview' : 'Start Ingestion'}
            </Button>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {/* Ingestion Error */}
        {ingestionError && (
          <div className="bg-red-500/10 text-red-400 border border-red-500/20 p-4 rounded-lg text-sm">
            {ingestionError}
          </div>
        )}

        {/* Manual Upload Section */}
        <UploadSection />

        {/* Active Jobs */}
        {activeJobs.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5 text-blue-400" />
              Active Jobs
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {activeJobs.map((job) => (
                <GlassCard key={job.jobId} className="p-5">
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-sm text-white/60">{job.jobId.substring(0, 8)}...</span>
                    <StatusIndicator status={getStatusColor(job.status.status) as 'healthy' | 'warning' | 'critical' | 'inactive'} />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <span className="text-xs text-white/40">Folders</span>
                      <div className="text-xl font-bold flex items-center gap-2">
                        <FolderOpen className="w-4 h-4 text-yellow-400" />
                        {job.status.folders_processed}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <span className="text-xs text-white/40">Threads</span>
                      <div className="text-xl font-bold flex items-center gap-2">
                        <FileText className="w-4 h-4 text-blue-400" />
                        {job.status.threads_created}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <span className="text-xs text-white/40">Chunks</span>
                      <div className="text-xl font-bold flex items-center gap-2">
                        <Database className="w-4 h-4 text-emerald-300" />
                        {job.status.chunks_created}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <span className="text-xs text-white/40">Embeddings</span>
                      <div className="text-xl font-bold flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-green-400" />
                        {job.status.embeddings_generated}
                      </div>
                    </div>
                  </div>

                  {job.status.errors > 0 && (
                    <div className="mt-4 p-2 rounded bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                      {job.status.errors} errors encountered
                    </div>
                  )}
                </GlassCard>
              ))}
            </div>
          </section>
        )}

        {/* S3 Folders */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-yellow-400" />
              S3 Folders ({folders.length})
            </h2>
            <Button
              onClick={loadFolders}
              disabled={isLoadingFolders}
              variant="ghost"
              size="icon"
              className="text-white/60 hover:text-white"
            >
              <RefreshCw className={cn("w-4 h-4", isLoadingFolders && "animate-spin")} />
            </Button>
          </div>

          {isLoadingFolders && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-emerald-300" />
            </div>
          )}

          {!isLoadingFolders && folders.length === 0 && (
            <GlassCard className="p-8 text-center">
              <FolderOpen className="w-12 h-12 text-white/20 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-white/60 mb-2">No Folders Found</h3>
              <p className="text-white/40 text-sm">
                No conversation folders found in S3. Check your bucket configuration.
              </p>
            </GlassCard>
          )}

          {!isLoadingFolders && folders.length > 0 && (
            <div className="rounded-xl border border-white/10 overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="bg-white/5 border-b border-white/10">
                    <th className="px-4 py-3 text-left text-xs font-medium text-white/40 uppercase tracking-wider">Folder</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-white/40 uppercase tracking-wider">Size</th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-white/40 uppercase tracking-wider">Modified</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {folders.map((folder, i) => (
                    <tr key={i} className="hover:bg-white/5 transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <FolderOpen className="w-4 h-4 text-yellow-400/60" />
                          <span className="font-mono text-sm">{folder.folder}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right text-sm text-white/60">
                        {folder.size_bytes ? `${(folder.size_bytes / 1024).toFixed(1)} KB` : '-'}
                      </td>
                      <td className="px-4 py-3 text-right text-sm text-white/60">
                        {folder.last_modified ? new Date(folder.last_modified).toLocaleDateString() : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
