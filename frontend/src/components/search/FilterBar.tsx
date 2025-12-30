import { useSearchParams } from 'react-router-dom';
import { Label } from '../ui/Label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/Select';

const FILTER_PARAM = 'filetype';

const FILE_TYPES = ['email', 'pdf', 'document', 'csv'];

export function FilterBar() {
  const [searchParams, setSearchParams] = useSearchParams();

  const handleFilterChange = (value: string) => {
    setSearchParams((prev) => {
      if (value === 'all') {
        prev.delete(FILTER_PARAM);
      } else {
        prev.set(FILTER_PARAM, value);
      }
      return prev;
    });
  };

  const currentFilter = searchParams.get(FILTER_PARAM) || 'all';

  return (
    <div className="flex items-center gap-4">
      <div className="grid w-full max-w-sm items-center gap-1.5">
        <Label htmlFor="filetype-filter" className="text-white/60">
          File Type
        </Label>
        <Select onValueChange={handleFilterChange} value={currentFilter} name="filetype-filter">
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by type..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            {FILE_TYPES.map((type) => (
              <SelectItem key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
