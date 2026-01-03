import { useState } from 'react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/Select';
import { Button } from '../ui/Button';
import { Plus } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import { logger } from '../../lib/logger';

export interface EmailTemplate {
  id: string;
  name: string;
  subject: string;
  body: string;
}

interface TemplateSelectorProps {
  templates: EmailTemplate[];
  onSelect: (template: EmailTemplate) => void;
  onCreateNew?: () => void;
  className?: string;
}

export default function TemplateSelector({
  templates,
  onSelect,
  onCreateNew,
  className,
}: TemplateSelectorProps) {
  const [selectedId, setSelectedId] = useState<string>('');

  const handleValueChange = (value: string) => {
    logger.debug('TemplateSelector: Template selection changed', { selectedId: value });
    setSelectedId(value);
    const template = templates.find((t) => t.id === value);
    if (template) {
      logger.info('TemplateSelector: Template selected', {
        id: template.id,
        name: template.name,
        subject: template.subject,
      });
      onSelect(template);
    } else {
      logger.warn('TemplateSelector: Selected template not found', { selectedId: value });
    }
  };

  return (
    <GlassCard className={className} data-testid="template-selector">
      <div className="p-4 flex gap-3 items-center">
        <Select value={selectedId} onValueChange={handleValueChange}>
          <SelectTrigger className="w-full bg-white/5 border-white/10 text-white placeholder:text-white/30">
            <SelectValue placeholder="Load a template..." />
          </SelectTrigger>
          <SelectContent>
            {templates.map((template) => (
              <SelectItem key={template.id} value={template.id}>
                {template.name}
              </SelectItem>
            ))}
            {templates.length === 0 && (
              <div className="p-2 text-sm text-center text-white/40">
                No templates saved
              </div>
            )}
          </SelectContent>
        </Select>

        {onCreateNew && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onCreateNew}
            title="Create new template"
            className="shrink-0"
          >
            <Plus className="w-4 h-4" />
          </Button>
        )}
      </div>
    </GlassCard>
  );
}
