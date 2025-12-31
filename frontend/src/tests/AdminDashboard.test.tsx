
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import AdminDashboard from '../views/AdminDashboard';
import { useAdmin } from '../hooks/useAdmin';
import { useToast } from '../contexts/toastContext';
import { useAuth } from '../contexts/AuthContext';

vi.mock('../hooks/useAdmin');
vi.mock('../contexts/toastContext');
vi.mock('../contexts/AuthContext');

const mockUseAdmin = useAdmin as Mock;
const mockUseToast = useToast as Mock;
const mockUseAuth = useAuth as Mock;

const defaultAdminState = {
    status: null,
    isStatusLoading: false,
    statusError: null,
    config: null,
    isConfigLoading: false,
    configError: null,
    runDoctor: vi.fn(),
    isDoctorRunning: false,
    doctorReport: null,
    doctorError: null,
};

describe('AdminDashboard', () => {
    const mockToast = vi.fn();

    beforeEach(() => {
        vi.resetAllMocks();
        mockUseToast.mockReturnValue({ addToast: mockToast });
        mockUseAdmin.mockReturnValue(defaultAdminState);
        mockUseAuth.mockReturnValue({ token: 'test-token' });
    });

    it('should render the dashboard title', () => {
        render(<AdminDashboard />);
        expect(screen.getByText('System Administration')).toBeInTheDocument();
    });

    // Loading States
    it('displays loading skeletons when status is loading', () => {
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, isStatusLoading: true });
        render(<AdminDashboard />);
        expect(screen.getAllByLabelText('Loading...').length).toBeGreaterThan(0);
    });

    it('displays loading skeletons when config is loading', () => {
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, isConfigLoading: true });
        render(<AdminDashboard />);
        expect(screen.getAllByLabelText('Loading...').length).toBeGreaterThan(0);
    });

    it('displays "Running..." when doctor is running', () => {
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, isDoctorRunning: true });
        render(<AdminDashboard />);
        expect(screen.getByText('Running...')).toBeInTheDocument();
    });

    // Error States
    it('displays an error message and calls toast when status fails to load', () => {
        const error = new Error('Failed to fetch status');
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, statusError: error });
        render(<AdminDashboard />);
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.getByText(error.message)).toBeInTheDocument();
        expect(mockToast).toHaveBeenCalledWith({
            type: "error",
            message: "Error loading status",
            details: error.message,
        });
    });

    it('displays an error message and calls toast when config fails to load', () => {
        const error = new Error('Failed to fetch config');
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, configError: error });
        render(<AdminDashboard />);
        expect(screen.getByText('Error')).toBeInTheDocument();
        expect(screen.getByText(error.message)).toBeInTheDocument();
        expect(mockToast).toHaveBeenCalledWith({
            type: "error",
            message: "Error loading configuration",
            details: error.message,
        });
    });

    it('displays an error message and calls toast when doctor fails', () => {
        const error = new Error('Diagnostics failed');
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, doctorError: error });
        render(<AdminDashboard />);
        expect(screen.getByText('Diagnostics Failed')).toBeInTheDocument();
        expect(mockToast).toHaveBeenCalledWith({
            type: "error",
            message: "Error running diagnostics",
            details: error.message,
        });
    });

    // Success States
    it('displays status information correctly', () => {
        const status = { service: 'TestService', env: 'test', status: 'OK' };
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, status });
        render(<AdminDashboard />);
        expect(screen.getByText('TestService')).toBeInTheDocument();
        expect(screen.getByText('test')).toBeInTheDocument();
        expect(screen.getByText('OK')).toBeInTheDocument();
    });

    it('displays configuration information correctly', () => {
        const config = { api_url: 'http://test.com', max_retries: 5 };
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, config });
        render(<AdminDashboard />);
        expect(screen.getByText('api url')).toBeInTheDocument();

        expect(screen.getByLabelText(/max retries/i)).toHaveValue(5);
    });

    it('displays doctor report correctly', () => {
        const doctorReport = {
            overall_status: 'healthy',
            checks: [
                { name: 'Database', status: 'pass', message: 'Connected' },
                { name: 'API', status: 'warn', message: 'High latency' },
            ],
        };
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, doctorReport });
        render(<AdminDashboard />);
        expect(screen.getByText('Overall Status')).toBeInTheDocument();
        expect(screen.getByText('healthy')).toBeInTheDocument();
        expect(screen.getByText('Database')).toBeInTheDocument();
        expect(screen.getByText('Connected')).toBeInTheDocument();
        expect(screen.getByText('API')).toBeInTheDocument();
        expect(screen.getByText('High latency')).toBeInTheDocument();
    });

    // Interaction Test
    it('calls runDoctor when the "Run Diagnostics" button is clicked', () => {
        const runDoctor = vi.fn();
        mockUseAdmin.mockReturnValue({ ...defaultAdminState, runDoctor });
        render(<AdminDashboard />);
        const button = screen.getByText('Run Diagnostics');
        fireEvent.click(button);
        expect(runDoctor).toHaveBeenCalledTimes(1);
    });
});
