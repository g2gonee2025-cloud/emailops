import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, DoctorReport, SystemStatus, SystemConfig } from '../lib/api';

export const useAdmin = () => {
  const queryClient = useQueryClient();

  const {
    data: status,
    isLoading: isStatusLoading,
    error: statusError,
  } = useQuery<SystemStatus, Error>({
    queryKey: ['systemStatus'],
    queryFn: () => api.fetchStatus(),
  });

  const {
    data: config,
    isLoading: isConfigLoading,
    error: configError,
  } = useQuery<SystemConfig, Error>({
    queryKey: ['systemConfig'],
    queryFn: () => api.fetchConfig(),
  });

  const {
    mutate: runDoctor,
    isPending: isDoctorRunning,
    data: doctorReport,
    error: doctorError,
  } = useMutation<DoctorReport, Error>({
    mutationFn: () => api.runDoctor(),
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
    doctorReport,
    doctorError,
  };
};
