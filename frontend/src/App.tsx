import React, { useEffect, useState, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';
import { Toaster } from 'react-hot-toast';
import { HelmetProvider } from 'react-helmet-async';
import { Provider as ReduxProvider } from 'react-redux';
import { motion, AnimatePresence } from 'framer-motion';

// Store and API
import { store } from './store/store';
import { useAppStore } from './store/appStore';

// Components
import Layout from './components/Layout';
import LoadingSpinner from './components/common/LoadingSpinner';
import ErrorBoundary from './components/common/ErrorBoundary';
import NotificationProvider from './components/common/NotificationProvider';

// Pages (Lazy loaded for code splitting)
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const PredictionForm = React.lazy(() => import('./pages/PredictionForm'));
const BatchPrediction = React.lazy(() => import('./pages/BatchPrediction'));
const ModelManagement = React.lazy(() => import('./pages/ModelManagement'));
const DataManagement = React.lazy(() => import('./pages/DataManagement'));
const Analytics = React.lazy(() => import('./pages/Analytics'));
const Settings = React.lazy(() => import('./pages/Settings'));
const Documentation = React.lazy(() => import('./pages/Documentation'));
const NotFound = React.lazy(() => import('./pages/NotFound'));

// Hooks and utilities
import { useHealthCheck } from './hooks/useHealthCheck';
import { useThemeMode } from './hooks/useThemeMode';
import { apiClient } from './services/api';

// Types
import { AppTheme } from './types';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
});

// Application theme configuration
const createAppTheme = (mode: 'light' | 'dark'): AppTheme => {
  const baseTheme = createTheme({
    palette: {
      mode,
      primary: {
        main: mode === 'light' ? '#1976d2' : '#90caf9',
        light: mode === 'light' ? '#42a5f5' : '#e3f2fd',
        dark: mode === 'light' ? '#1565c0' : '#42a5f5',
        contrastText: '#ffffff',
      },
      secondary: {
        main: mode === 'light' ? '#dc004e' : '#f48fb1',
        light: mode === 'light' ? '#ff5983' : '#fce4ec',
        dark: mode === 'light' ? '#9a0036' : '#ad1457',
        contrastText: '#ffffff',
      },
      error: {
        main: mode === 'light' ? '#d32f2f' : '#f44336',
        light: mode === 'light' ? '#ef5350' : '#e57373',
        dark: mode === 'light' ? '#c62828' : '#d32f2f',
      },
      warning: {
        main: mode === 'light' ? '#ed6c02' : '#ff9800',
        light: mode === 'light' ? '#ff9800' : '#ffb74d',
        dark: mode === 'light' ? '#e65100' : '#f57c00',
      },
      success: {
        main: mode === 'light' ? '#2e7d32' : '#4caf50',
        light: mode === 'light' ? '#4caf50' : '#81c784',
        dark: mode === 'light' ? '#1b5e20' : '#388e3c',
      },
      info: {
        main: mode === 'light' ? '#0288d1' : '#29b6f6',
        light: mode === 'light' ? '#03a9f4' : '#4fc3f7',
        dark: mode === 'light' ? '#01579b' : '#0277bd',
      },
      background: {
        default: mode === 'light' ? '#fafafa' : '#121212',
        paper: mode === 'light' ? '#ffffff' : '#1e1e1e',
      },
      text: {
        primary: mode === 'light' ? 'rgba(0, 0, 0, 0.87)' : 'rgba(255, 255, 255, 0.87)',
        secondary: mode === 'light' ? 'rgba(0, 0, 0, 0.6)' : 'rgba(255, 255, 255, 0.6)',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontWeight: 700,
        fontSize: '2.5rem',
        lineHeight: 1.2,
      },
      h2: {
        fontWeight: 600,
        fontSize: '2rem',
        lineHeight: 1.3,
      },
      h3: {
        fontWeight: 600,
        fontSize: '1.5rem',
        lineHeight: 1.4,
      },
      h4: {
        fontWeight: 500,
        fontSize: '1.25rem',
        lineHeight: 1.4,
      },
      h5: {
        fontWeight: 500,
        fontSize: '1.125rem',
        lineHeight: 1.5,
      },
      h6: {
        fontWeight: 500,
        fontSize: '1rem',
        lineHeight: 1.5,
      },
      body1: {
        fontSize: '1rem',
        lineHeight: 1.6,
      },
      body2: {
        fontSize: '0.875rem',
        lineHeight: 1.5,
      },
      button: {
        fontWeight: 600,
        fontSize: '0.875rem',
        textTransform: 'none' as const,
      },
    },
    shape: {
      borderRadius: 12,
    },
    spacing: 8,
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            padding: '8px 24px',
            fontSize: '0.875rem',
            fontWeight: 600,
            textTransform: 'none',
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.15)',
            },
          },
          contained: {
            boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.12)',
            '&:hover': {
              boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.15)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.12), 0px 1px 2px rgba(0, 0, 0, 0.24)',
            '&:hover': {
              boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.15)',
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: 12,
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: {
            '& .MuiOutlinedInput-root': {
              borderRadius: 8,
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 20,
            fontWeight: 500,
          },
        },
      },
    },
  });

  return baseTheme;
};

// Main App Component
const App: React.FC = () => {
  const { themeMode, toggleTheme } = useThemeMode();
  const { healthStatus, checkHealth } = useHealthCheck();
  const { setAppReady, appReady } = useAppStore();
  
  const [theme, setTheme] = useState<AppTheme>(createAppTheme(themeMode));

  // Update theme when mode changes
  useEffect(() => {
    setTheme(createAppTheme(themeMode));
  }, [themeMode]);

  // Initialize application
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check API health
        await checkHealth();
        
        // Initialize API client
        await apiClient.initialize();
        
        // Set app as ready
        setAppReady(true);
        
        console.log('🚀 Cardiovascular Prediction System initialized successfully');
      } catch (error) {
        console.error('❌ Failed to initialize application:', error);
        // App can still work in offline mode
        setAppReady(true);
      }
    };

    initializeApp();
  }, [checkHealth, setAppReady]);

  // Show loading screen during initialization
  if (!appReady) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh',
            flexDirection: 'column',
            gap: '24px',
          }}
        >
          <LoadingSpinner size="large" />
          <div style={{ textAlign: 'center', color: theme.palette.text.secondary }}>
            <h2>Cardiovascular Prediction System</h2>
            <p>Initializing application...</p>
          </div>
        </div>
      </ThemeProvider>
    );
  }

  return (
    <ErrorBoundary>
      <HelmetProvider>
        <QueryClientProvider client={queryClient}>
          <ReduxProvider store={store}>
            <ThemeProvider theme={theme}>
              <LocalizationProvider dateAdapter={AdapterDateFns}>
                <CssBaseline />
                <NotificationProvider>
                  <Router>
                    <AnimatePresence mode="wait">
                      <motion.div
                        key="app-container"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        style={{ minHeight: '100vh' }}
                      >
                        <Layout themeMode={themeMode} onThemeToggle={toggleTheme}>
                          <Suspense
                            fallback={
                              <div style={{ 
                                display: 'flex', 
                                justifyContent: 'center', 
                                alignItems: 'center', 
                                height: '400px' 
                              }}>
                                <LoadingSpinner />
                              </div>
                            }
                          >
                            <Routes>
                              {/* Main Dashboard */}
                              <Route path="/" element={<Dashboard />} />
                              <Route path="/dashboard" element={<Navigate to="/" replace />} />
                              
                              {/* Prediction Routes */}
                              <Route path="/predict" element={<PredictionForm />} />
                              <Route path="/predict/single" element={<Navigate to="/predict" replace />} />
                              <Route path="/predict/batch" element={<BatchPrediction />} />
                              
                              {/* Data Management */}
                              <Route path="/data" element={<DataManagement />} />
                              <Route path="/data/upload" element={<Navigate to="/data" replace />} />
                              <Route path="/data/datasets" element={<Navigate to="/data" replace />} />
                              
                              {/* Model Management */}
                              <Route path="/models" element={<ModelManagement />} />
                              <Route path="/models/performance" element={<Navigate to="/models" replace />} />
                              <Route path="/models/training" element={<Navigate to="/models" replace />} />
                              
                              {/* Analytics */}
                              <Route path="/analytics" element={<Analytics />} />
                              <Route path="/analytics/predictions" element={<Navigate to="/analytics" replace />} />
                              <Route path="/analytics/models" element={<Navigate to="/analytics" replace />} />
                              
                              {/* System */}
                              <Route path="/settings" element={<Settings />} />
                              <Route path="/docs" element={<Documentation />} />
                              <Route path="/documentation" element={<Navigate to="/docs" replace />} />
                              
                              {/* Health Check Route */}
                              <Route 
                                path="/health" 
                                element={
                                  <div style={{ padding: '24px', textAlign: 'center' }}>
                                    <h2>System Health</h2>
                                    <pre style={{ textAlign: 'left', background: '#f5f5f5', padding: '16px', borderRadius: '8px' }}>
                                      {JSON.stringify(healthStatus, null, 2)}
                                    </pre>
                                  </div>
                                } 
                              />
                              
                              {/* 404 Route */}
                              <Route path="*" element={<NotFound />} />
                            </Routes>
                          </Suspense>
                        </Layout>
                      </motion.div>
                    </AnimatePresence>
                  </Router>
                  
                  {/* Global Notifications */}
                  <Toaster
                    position="top-right"
                    toastOptions={{
                      duration: 4000,
                      style: {
                        background: theme.palette.background.paper,
                        color: theme.palette.text.primary,
                        border: `1px solid ${theme.palette.divider}`,
                        borderRadius: '12px',
                        boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.15)',
                      },
                      success: {
                        iconTheme: {
                          primary: theme.palette.success.main,
                          secondary: theme.palette.success.contrastText,
                        },
                      },
                      error: {
                        iconTheme: {
                          primary: theme.palette.error.main,
                          secondary: theme.palette.error.contrastText,
                        },
                      },
                    }}
                  />
                  
                  {/* Development Tools */}
                  {process.env.NODE_ENV === 'development' && (
                    <ReactQueryDevtools 
                      initialIsOpen={false} 
                      position="bottom-right"
                    />
                  )}
                </NotificationProvider>
              </LocalizationProvider>
            </ThemeProvider>
          </ReduxProvider>
        </QueryClientProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
};

export default App;