import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { MemoryRouter } from 'react-router-dom';

// Component and hooks to test
import PredictionForm from '../../components/PredictionForm';
import { usePrediction } from '../../hooks/usePrediction';
import * as predictionService from '../../services/predictionService';

// Mock the hooks and services
vi.mock('../../hooks/usePrediction');
vi.mock('../../services/predictionService');

// Mock data
const mockPredictionData = {
  Age: 65,
  Sex: 'Male',
  BMI: 28.5,
  Smoking: true,
  HighBP: true,
  HighChol: true,
  Diabetes: false,
  PhysActivity: true,
  HvyAlcoholConsump: false,
  PhysHlth: 5,
  MentHlth: 2,
  NoDocbcCost: false,
  GenHlth: 'Good',
  DiffWalk: false,
  Stroke: false
};

const mockPredictionResult = {
  prediction: 1,
  probability: 0.75,
  model_version: 'v1.0.0',
  explanation: {
    top_risk_factors: ['HighBP', 'Age', 'BMI'],
    risk_level: 'High'
  },
  timestamp: '2024-01-15T12:00:00Z',
  correlation_id: 'test-correlation-123'
};

// Test wrapper component
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

  const theme = createTheme();

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <MemoryRouter>
          {children}
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

// Custom render function with wrapper
const renderWithWrapper = (component: React.ReactElement) => {
  return render(component, { wrapper: TestWrapper });
};

describe('PredictionForm Component', () => {
  const mockUsePrediction = vi.mocked(usePrediction);
  const mockPredictionService = vi.mocked(predictionService);

  // Default mock implementation
  const defaultMockPrediction = {
    predict: vi.fn(),
    isLoading: false,
    error: null,
    data: null,
    reset: vi.fn()
  };

  beforeEach(() => {
    // Reset all mocks before each test
    vi.clearAllMocks();
    
    // Set up default mock implementations
    mockUsePrediction.mockReturnValue(defaultMockPrediction);
    mockPredictionService.predictHeartDisease.mockResolvedValue(mockPredictionResult);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Form Rendering', () => {
    it('renders the prediction form with all required fields', () => {
      renderWithWrapper(<PredictionForm />);

      // Check for main form elements
      expect(screen.getByRole('heading', { name: /cardiovascular risk assessment/i })).toBeInTheDocument();
      expect(screen.getByRole('form')).toBeInTheDocument();
      
      // Check for key input fields
      expect(screen.getByLabelText(/age/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/sex/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/bmi/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/smoking/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/high blood pressure/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/high cholesterol/i)).toBeInTheDocument();
      
      // Check for submit button
      expect(screen.getByRole('button', { name: /predict risk/i })).toBeInTheDocument();
    });

    it('renders form fields with correct initial values', () => {
      renderWithWrapper(<PredictionForm />);

      // Check default values
      const ageInput = screen.getByLabelText(/age/i) as HTMLInputElement;
      expect(ageInput.value).toBe('');
      
      const bmiInput = screen.getByLabelText(/bmi/i) as HTMLInputElement;
      expect(bmiInput.value).toBe('');
      
      // Boolean fields should be unchecked by default
      const smokingCheckbox = screen.getByLabelText(/smoking/i) as HTMLInputElement;
      expect(smokingCheckbox.checked).toBe(false);
    });

    it('renders with proper accessibility attributes', () => {
      renderWithWrapper(<PredictionForm />);

      // Check for proper labeling
      const ageInput = screen.getByLabelText(/age/i);
      expect(ageInput).toHaveAttribute('aria-required', 'true');
      
      const form = screen.getByRole('form');
      expect(form).toHaveAttribute('noValidate');
      
      // Check for fieldsets
      const personalInfoFieldset = screen.getByRole('group', { name: /personal information/i });
      expect(personalInfoFieldset).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('validates required fields on submit', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const submitButton = screen.getByRole('button', { name: /predict risk/i });
      await user.click(submitButton);

      // Should show validation errors for required fields
      await waitFor(() => {
        expect(screen.getByText(/age is required/i)).toBeInTheDocument();
        expect(screen.getByText(/sex is required/i)).toBeInTheDocument();
        expect(screen.getByText(/bmi is required/i)).toBeInTheDocument();
      });
    });

    it('validates age range', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const ageInput = screen.getByLabelText(/age/i);
      
      // Test invalid low age
      await user.clear(ageInput);
      await user.type(ageInput, '10');
      await user.tab(); // Trigger blur event

      await waitFor(() => {
        expect(screen.getByText(/age must be between 18 and 100/i)).toBeInTheDocument();
      });

      // Test invalid high age
      await user.clear(ageInput);
      await user.type(ageInput, '150');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/age must be between 18 and 100/i)).toBeInTheDocument();
      });
    });

    it('validates BMI range', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const bmiInput = screen.getByLabelText(/bmi/i);
      
      // Test invalid low BMI
      await user.clear(bmiInput);
      await user.type(bmiInput, '5');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/bmi must be between 10 and 50/i)).toBeInTheDocument();
      });

      // Test invalid high BMI
      await user.clear(bmiInput);
      await user.type(bmiInput, '60');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/bmi must be between 10 and 50/i)).toBeInTheDocument();
      });
    });

    it('validates health days range', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const physHlthInput = screen.getByLabelText(/physical health days/i);
      
      // Test invalid high value
      await user.clear(physHlthInput);
      await user.type(physHlthInput, '35');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/physical health days must be between 0 and 30/i)).toBeInTheDocument();
      });
    });

    it('clears validation errors when fields are corrected', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const ageInput = screen.getByLabelText(/age/i);
      
      // First create an error
      await user.type(ageInput, '10');
      await user.tab();

      await waitFor(() => {
        expect(screen.getByText(/age must be between 18 and 100/i)).toBeInTheDocument();
      });

      // Then correct it
      await user.clear(ageInput);
      await user.type(ageInput, '65');
      await user.tab();

      await waitFor(() => {
        expect(screen.queryByText(/age must be between 18 and 100/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Form Interactions', () => {
    it('updates form state when user inputs data', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      // Fill in form fields
      await user.type(screen.getByLabelText(/age/i), '65');
      await user.selectOptions(screen.getByLabelText(/sex/i), 'Male');
      await user.type(screen.getByLabelText(/bmi/i), '28.5');
      await user.click(screen.getByLabelText(/smoking/i));

      // Verify values are updated
      expect(screen.getByDisplayValue('65')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Male')).toBeInTheDocument();
      expect(screen.getByDisplayValue('28.5')).toBeInTheDocument();
      expect(screen.getByLabelText(/smoking/i)).toBeChecked();
    });

    it('handles checkbox interactions correctly', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const smokingCheckbox = screen.getByLabelText(/smoking/i);
      const highBPCheckbox = screen.getByLabelText(/high blood pressure/i);

      // Initially unchecked
      expect(smokingCheckbox).not.toBeChecked();
      expect(highBPCheckbox).not.toBeChecked();

      // Click to check
      await user.click(smokingCheckbox);
      await user.click(highBPCheckbox);

      expect(smokingCheckbox).toBeChecked();
      expect(highBPCheckbox).toBeChecked();

      // Click to uncheck
      await user.click(smokingCheckbox);
      expect(smokingCheckbox).not.toBeChecked();
    });

    it('handles select dropdown interactions', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const sexSelect = screen.getByLabelText(/sex/i);
      const genHlthSelect = screen.getByLabelText(/general health/i);

      // Select options
      await user.selectOptions(sexSelect, 'Female');
      await user.selectOptions(genHlthSelect, 'Excellent');

      expect(screen.getByDisplayValue('Female')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Excellent')).toBeInTheDocument();
    });

    it('resets form when reset button is clicked', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      // Fill in some data
      await user.type(screen.getByLabelText(/age/i), '65');
      await user.click(screen.getByLabelText(/smoking/i));

      // Verify data is there
      expect(screen.getByDisplayValue('65')).toBeInTheDocument();
      expect(screen.getByLabelText(/smoking/i)).toBeChecked();

      // Reset form
      const resetButton = screen.getByRole('button', { name: /reset/i });
      await user.click(resetButton);

      // Verify form is reset
      expect(screen.queryByDisplayValue('65')).not.toBeInTheDocument();
      expect(screen.getByLabelText(/smoking/i)).not.toBeChecked();
    });
  });

  describe('Form Submission', () => {
    it('submits form with valid data', async () => {
      const user = userEvent.setup();
      const mockPredict = vi.fn().mockResolvedValue(mockPredictionResult);
      
      mockUsePrediction.mockReturnValue({
        ...defaultMockPrediction,
        predict: mockPredict
      });

      renderWithWrapper(<PredictionForm />);

      // Fill in valid form data
      await user.type(screen.getByLabelText(/age/i), '65');
      await user.selectOptions(screen.getByLabelText(/sex/i), 'Male');
      await user.type(screen.getByLabelText(/bmi/i), '28.5');
      await user.click(screen.getByLabelText(/smoking/i));
      await user.click(screen.getByLabelText(/high blood pressure/i));

      // Submit form
      const submitButton = screen.getByRole('button', { name: /predict risk/i });
      await user.click(submitButton);

      // Verify the prediction was called with correct data
      await waitFor(() => {
        expect(mockPredict).toHaveBeenCalledWith({
          Age: 65,
          Sex: 'Male',
          BMI: 28.5,
          Smoking: true,
          HighBP: true,
          HighChol: false,
          Diabetes: false,
          PhysActivity: false,
          HvyAlcoholConsump: false,
          PhysHlth: 0,
          MentHlth: 0,
          NoDocbcCost: false,
          GenHlth: '',
          DiffWalk: false,
          Stroke: false
        });
      });
    });

    it('prevents submission with invalid data', async () => {
      const user = userEvent.setup();
      const mockPredict = vi.fn();
      
      mockUsePrediction.mockReturnValue({
        ...defaultMockPrediction,
        predict: mockPredict
      });

      renderWithWrapper(<PredictionForm />);

      // Try to submit without filling required fields
      const submitButton = screen.getByRole('button', { name: /predict risk/i });
      await user.click(submitButton);

      // Should not call predict function
      expect(mockPredict).not.toHaveBeenCalled();
    });

    it('shows loading state during submission', async () => {
      const user = userEvent.setup();
      
      mockUsePrediction.mockReturnValue({
        ...defaultMockPrediction,
        isLoading: true
      });

      renderWithWrapper(<PredictionForm />);

      // Should show loading state
      const submitButton = screen.getByRole('button', { name: /predicting.../i });
      expect(submitButton).toBeDisabled();
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('displays error message when prediction fails', async () => {
      const errorMessage = 'Prediction service temporarily unavailable';
      
      mockUsePrediction.mockReturnValue({
        ...defaultMockPrediction,
        error: new Error(errorMessage)
      });

      renderWithWrapper(<PredictionForm />);

      // Should display error message
      await waitFor(() => {
        expect(screen.getByText(new RegExp(errorMessage, 'i'))).toBeInTheDocument();
      });

      // Should show error alert
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('allows retrying after error', async () => {
      const user = userEvent.setup();
      const mockPredict = vi.fn();
      const mockReset = vi.fn();
      
      mockUsePrediction.mockReturnValue({
        ...defaultMockPrediction,
        predict: mockPredict,
        reset: mockReset,
        error: new Error('Network error')
      });

      renderWithWrapper(<PredictionForm />);

      // Should show retry button
      const retryButton = screen.getByRole('button', { name: /try again/i });
      await user.click(retryButton);

      // Should call reset function
      expect(mockReset).toHaveBeenCalled();
    });
  });

  describe('Results Display', () => {
    it('displays prediction results when available', async () => {
      mockUsePrediction.mockReturnValue({
        ...defaultMockPrediction,
        data: mockPredictionResult
      });

      renderWithWrapper(<PredictionForm />);

      // Should display results
      await waitFor(() => {
        expect(screen.getByText(/prediction result/i)).toBeInTheDocument();
        expect(screen.getByText(/high risk/i)).toBeInTheDocument();
        expect(screen.getByText(/75%/)).toBeInTheDocument();
      });
    });

    it('displays risk factors explanation', async () => {
      mockUsePrediction.mockReturnValue({
        ...defaultMockPrediction,
        data: mockPredictionResult
      });

      renderWithWrapper(<PredictionForm />);

      // Should display top risk factors
      await waitFor(() => {
        expect(screen.getByText(/top risk factors/i)).toBeInTheDocument();
        expect(screen.getByText(/HighBP/)).toBeInTheDocument();
        expect(screen.getByText(/Age/)).toBeInTheDocument();
        expect(screen.getByText(/BMI/)).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('supports keyboard navigation', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const ageInput = screen.getByLabelText(/age/i);
      const sexSelect = screen.getByLabelText(/sex/i);
      const submitButton = screen.getByRole('button', { name: /predict risk/i });

      // Test tab navigation
      ageInput.focus();
      expect(ageInput).toHaveFocus();

      await user.tab();
      expect(sexSelect).toHaveFocus();

      // Navigate to submit button
      await user.keyboard('{Tab}');
      await user.keyboard('{Tab}');
      await user.keyboard('{Tab}');
      // Continue tabbing to reach submit button...
    });

    it('has proper ARIA labels and descriptions', () => {
      renderWithWrapper(<PredictionForm />);

      // Check for ARIA labels
      expect(screen.getByLabelText(/age/i)).toHaveAttribute('aria-required', 'true');
      expect(screen.getByLabelText(/sex/i)).toHaveAttribute('aria-required', 'true');
      expect(screen.getByLabelText(/bmi/i)).toHaveAttribute('aria-required', 'true');

      // Check for fieldset grouping
      const personalInfoGroup = screen.getByRole('group', { name: /personal information/i });
      expect(personalInfoGroup).toBeInTheDocument();
    });

    it('announces form validation errors to screen readers', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      const ageInput = screen.getByLabelText(/age/i);
      
      // Trigger validation error
      await user.type(ageInput, '10');
      await user.tab();

      await waitFor(() => {
        const errorElement = screen.getByText(/age must be between 18 and 100/i);
        expect(errorElement).toHaveAttribute('role', 'alert');
      });
    });
  });

  describe('Responsive Design', () => {
    it('adapts to mobile viewport', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      renderWithWrapper(<PredictionForm />);

      // Should still render all essential elements
      expect(screen.getByRole('form')).toBeInTheDocument();
      expect(screen.getByLabelText(/age/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /predict risk/i })).toBeInTheDocument();
    });
  });

  describe('Integration with External Services', () => {
    it('calls the prediction service with correct parameters', async () => {
      const user = userEvent.setup();
      renderWithWrapper(<PredictionForm />);

      // Fill and submit form
      await user.type(screen.getByLabelText(/age/i), '65');
      await user.selectOptions(screen.getByLabelText(/sex/i), 'Male');
      await user.type(screen.getByLabelText(/bmi/i), '28.5');
      
      const submitButton = screen.getByRole('button', { name: /predict risk/i });
      await user.click(submitButton);

      // Verify service was called
      await waitFor(() => {
        expect(mockPredictionService.predictHeartDisease).toHaveBeenCalledWith(
          expect.objectContaining({
            Age: 65,
            Sex: 'Male',
            BMI: 28.5
          })
        );
      });
    });
  });
});