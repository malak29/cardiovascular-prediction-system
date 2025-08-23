import React from 'react';
import { createRoot } from 'react-dom/client';
import { StrictMode } from 'react';

// Main App Component
import App from './App';

// Performance monitoring
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';
import reportWebVitals from './reportWebVitals';

// Global styles
import './styles/index.css';
import './styles/globals.css';

// Error tracking and monitoring
const setupErrorTracking = () => {
  // Global error handler
  window.addEventListener('error', (event) => {
    console.error('Global error caught:', {
      message: event.message,
      filename: event.filename,
      line: event.lineno,
      column: event.colno,
      error: event.error,
      timestamp: new Date().toISOString(),
    });
    
    // In production, send to error tracking service
    if (process.env.NODE_ENV === 'production' && process.env.REACT_APP_SENTRY_DSN) {
      // Sentry or other error tracking would go here
      console.log('Error would be sent to tracking service in production');
    }
  });

  // Unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', {
      reason: event.reason,
      promise: event.promise,
      timestamp: new Date().toISOString(),
    });
    
    // Prevent default browser behavior
    event.preventDefault();
  });
};

// Performance monitoring setup
const setupPerformanceMonitoring = () => {
  // Web Vitals monitoring
  const reportWebVital = (metric: any) => {
    console.log('Web Vital:', {
      name: metric.name,
      value: metric.value,
      id: metric.id,
      delta: metric.delta,
      timestamp: new Date().toISOString(),
    });
    
    // In production, send to analytics service
    if (process.env.NODE_ENV === 'production') {
      // Analytics tracking would go here
      // Example: gtag('event', metric.name, { value: metric.value });
    }
  };

  // Collect Core Web Vitals
  getCLS(reportWebVital);
  getFID(reportWebVital);
  getFCP(reportWebVital);
  getLCP(reportWebVital);
  getTTFB(reportWebVital);
};

// Application configuration
const setupApplicationConfig = () => {
  // Set global application configuration
  (window as any).__CVD_APP_CONFIG__ = {
    version: '1.0.0',
    buildDate: process.env.REACT_APP_BUILD_DATE || new Date().toISOString(),
    environment: process.env.REACT_APP_ENVIRONMENT || 'development',
    apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
    features: {
      darkMode: true,
      notifications: true,
      analytics: process.env.NODE_ENV === 'production',
      devTools: process.env.NODE_ENV === 'development',
    },
  };

  // Log application startup information
  console.log('🏥 Cardiovascular Disease Prediction System');
  console.log('📊 Version:', (window as any).__CVD_APP_CONFIG__.version);
  console.log('🌍 Environment:', (window as any).__CVD_APP_CONFIG__.environment);
  console.log('🔗 API URL:', (window as any).__CVD_APP_CONFIG__.apiUrl);
  
  if (process.env.NODE_ENV === 'development') {
    console.log('🛠️ Development mode - additional debugging enabled');
    console.log('📋 Available routes:');
    console.log('  🏠 Dashboard: /');
    console.log('  🔮 Prediction: /predict');
    console.log('  📊 Batch Prediction: /predict/batch');
    console.log('  🗄️ Data Management: /data');
    console.log('  🤖 Model Management: /models');
    console.log('  📈 Analytics: /analytics');
    console.log('  ⚙️ Settings: /settings');
    console.log('  📚 Documentation: /docs');
  }
};

// Service Worker registration (for PWA capabilities)
const registerServiceWorker = () => {
  if (process.env.NODE_ENV === 'production' && 'serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker
        .register('/service-worker.js')
        .then((registration) => {
          console.log('✅ Service Worker registered successfully:', registration.scope);
          
          // Check for updates
          registration.addEventListener('updatefound', () => {
            const newWorker = registration.installing;
            if (newWorker) {
              newWorker.addEventListener('statechange', () => {
                if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                  // New content available
                  console.log('🔄 New app version available');
                  
                  // Show update notification
                  if (window.confirm('A new version is available. Reload to update?')) {
                    window.location.reload();
                  }
                }
              });
            }
          });
        })
        .catch((error) => {
          console.error('❌ Service Worker registration failed:', error);
        });
    });
  }
};

// Initialize application
const initializeApplication = async () => {
  try {
    // Setup error tracking
    setupErrorTracking();
    
    // Setup performance monitoring
    setupPerformanceMonitoring();
    
    // Setup application configuration
    setupApplicationConfig();
    
    // Register service worker
    registerServiceWorker();
    
    // Get root container
    const container = document.getElementById('root');
    if (!container) {
      throw new Error('Root container not found');
    }

    // Create React root and render app
    const root = createRoot(container);
    
    root.render(
      <StrictMode>
        <App />
      </StrictMode>
    );

    // Log successful initialization
    console.log('✅ Application initialized successfully');
    
    // Report startup performance
    const startupTime = performance.now();
    console.log(`⚡ Application startup time: ${startupTime.toFixed(2)}ms`);
    
  } catch (error) {
    console.error('❌ Application initialization failed:', error);
    
    // Show error fallback
    const container = document.getElementById('root');
    if (container) {
      container.innerHTML = `
        <div style="
          display: flex;
          justify-content: center;
          align-items: center;
          height: 100vh;
          font-family: 'Inter', sans-serif;
          background: #fafafa;
          color: #333;
          text-align: center;
          padding: 24px;
        ">
          <div>
            <h1 style="color: #d32f2f; margin-bottom: 16px;">
              ⚠️ Application Failed to Load
            </h1>
            <p style="margin-bottom: 24px; max-width: 500px;">
              The Cardiovascular Disease Prediction System could not be initialized. 
              Please check your connection and try refreshing the page.
            </p>
            <button 
              onclick="window.location.reload()" 
              style="
                background: #1976d2;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
              "
            >
              🔄 Reload Application
            </button>
            <details style="margin-top: 24px; text-align: left; max-width: 600px;">
              <summary style="cursor: pointer; font-weight: 600;">Technical Details</summary>
              <pre style="
                background: #f5f5f5;
                padding: 16px;
                border-radius: 8px;
                margin-top: 12px;
                overflow: auto;
                font-size: 12px;
              ">${error}</pre>
            </details>
          </div>
        </div>
      `;
    }
  }
};

// Start the application
initializeApplication();

// Report web vitals for performance monitoring
reportWebVitals((metric) => {
  // Log performance metrics
  console.log('📊 Web Vital:', metric.name, metric.value);
  
  // In production, send to analytics
  if (process.env.NODE_ENV === 'production') {
    // Analytics reporting would go here
    // Example: gtag('event', metric.name, { value: metric.value });
  }
});

// Export for testing
export { initializeApplication };