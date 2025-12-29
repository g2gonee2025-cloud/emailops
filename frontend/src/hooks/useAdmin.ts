import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

const adminKeys = {
  all: ['admin'] as const,
  status: () => [...adminKeys.all, 'status'] as const,
  config: () => [...adminKeys.all, 'config'] as const,
  doctor: () => [...adminKeys.all, 'doctor'] as const,
};

export function useAdmin() {
  const queryClient = useQueryClient();

  const { data: status, isLoading: isStatusLoading, error: statusError } = useQuery({
    queryKey: adminKeys.status(),
    queryFn: api.fetchStatus,
  });

  const { data: config, isLoading: isConfigLoading, error: configError } = useQuery({
    queryKey: adminKeys.config(),
    queryFn: api.fetchConfig,
  });

  const { mutate: runDoctor, isPending: isDoctorRunning, error: doctorError, data: doctorReport } = useMutation({
    mutationFn: api.runDoctor,
    onSuccess: (data) => {
      queryClient.setQueryData(adminKeys.doctor(), data);
    },
  });

  return {
    status,
    isStatusLoading,
    statusError,
    config,
    isConfigLoading,
    configError,
    runDoctor,
    isDoctorRunning,
    doctorError,
    doctorReport,
  };
}
