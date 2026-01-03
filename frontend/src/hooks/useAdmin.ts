import { useQuery, useMutation } from '@tanstack/react-query';
import { api, type DoctorReport, type StatusData as SystemStatus, type SystemConfig } from '../lib/api';
import { logger } from '../lib/logger';

export const useAdmin = () => {

  const {
    data: status,
    isLoading: isStatusLoading,
    error: statusError,
  } = useQuery<SystemStatus, Error>({
    queryKey: ['systemStatus'],
    queryFn: () => {
      logger.debug('useAdmin: Fetching system status');
      return api.fetchStatus();
    },
  });

  const {
    data: config,
    isLoading: isConfigLoading,
    error: configError,
  } = useQuery<SystemConfig, Error>({
    queryKey: ['systemConfig'],
    queryFn: () => {
      logger.debug('useAdmin: Fetching system config');
      return api.fetchConfig();
    },
  });

  const {
    mutate: runDoctor,
    isPending: isDoctorRunning,
    data: doctorReport,
    error: doctorError,
  } = useMutation<DoctorReport, Error>({
    mutationFn: () => {
      logger.info('useAdmin: Running system diagnostics');
      return api.runDoctor();
    },
    onSuccess: (data) => {
      logger.info('useAdmin: Diagnostics completed', {
        overall_status: data.overall_status,
        checks_count: data.checks.length,
      });
    },
    onError: (error) => {
      logger.error('useAdmin: Diagnostics failed', {
        error: error.message,
      });
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
    doctorReport,
    doctorError,
  };
};
