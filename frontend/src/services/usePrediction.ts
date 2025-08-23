import { useState, useCallback, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

// Types
import {
  PredictionRequest,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  PredictionStatistics,
  ModelInfo,
  ModelPerformance,
  AsyncState,
} from '../types';

// Services
import predictionService from '../services/predictionService';
import apiClient from '../services/api';

// Utilities
import { debounce } from 'lodash';
import { trackEvent } from '../utils/analytics';

/**
 * Prediction hook options
 */
interface UsePredictionOptions {
  enableRealTimeUpdates?: boolean;
  cacheResults?: boolean;
  autoValidate?: boolean;
  onSuccess?: (result: PredictionResponse) => void;
  onError?: (error: Error) => void;
}

/**
 * Prediction hook return type
 */
interface UsePredictionReturn {
  // Single prediction
  makePrediction: (request: PredictionRequest) => Promise<PredictionResponse>;
  predictionResult: PredictionResponse | null;
  
  // Batch prediction
  makeBatchPrediction: (request: BatchPredictionRequest) => Promise<BatchPredictionResponse>;
  batchResult: BatchPredictionResponse | null;
  
  // CSV upload prediction
  uploadCsvPrediction: (
    file: File,
    options?: { modelVersion?: string; includeConfidence?: boolean },
    onProgress?: (progress: number) => void
  ) => Promise<Blob>;
  
  // State
  loading: boolean;
  error: string | null;
  progress: number;
  
  // History and analytics
  predictionHistory: any | null;
  statistics: PredictionStatistics | null;
  
  // Actions
  clearError: () => void;
  clearResults: () => void;
  refreshHistory: () => void;
  
  // Validation
  validatePatientData: (data: any) => {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    recommendations: string[];
  };
}

/**
 * Main prediction hook
 */
export const usePrediction = (options: UsePredictionOptions = {}): UsePredictionReturn => {
  const {
    enableRealTimeUpdates = false,
    cacheResults = true,
    autoValidate = true,
    onSuccess,
    onError,
  } = options;

  // React Query client for cache management
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  // Local state
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  // Refs for cleanup
  const abortControllerRef = useRef<AbortController | null>(null);

  // Query keys
  const QUERY_KEYS = {
    PREDICTION_HISTORY: 'prediction-history',
    PREDICTION_STATS: 'prediction-statistics',
    AVAILABLE_MODELS: 'available-models',
    MODEL_PERFORMANCE: 'model-performance',
  };

  // ========================================
  // PREDICTION MUTATIONS
  // ========================================

  // Single prediction mutation
  const singlePredictionMutation = useMutation({
    mutationFn: async (request: PredictionRequest): Promise<PredictionResponse> => {
      // Track analytics
      trackEvent('prediction_started', {
        type: 'single',
        model_version: request.model_version,
        include_confidence: request.include_confidence,
        include_features: request.include_features,
      });

      const result = await predictionService.makePrediction(request);
      
      // Track completion
      trackEvent('prediction_completed', {
        type: 'single',
        risk_score: result.risk_score,
        risk_category: result.risk_category,
        processing_time_ms: result.prediction_time_ms,
      });

      return result;
    },
    onSuccess: (result) => {
      setPredictionResult(result);
      setError(null);
      
      // Invalidate related queries
      queryClient.invalidateQueries(QUERY_KEYS.PREDICTION_HISTORY);
      queryClient.invalidateQueries(QUERY_KEYS.PREDICTION_STATS);
      
      // Success callback
      onSuccess?.(result);
      
      console.log('✅ Single prediction completed successfully');
    },
    onError: (error: any) => {
      const errorMessage = error.message || 'Prediction failed';
      setError(errorMessage);
      
      // Track error
      trackEvent('prediction_failed', {
        type: 'single',
        error_message: errorMessage,
      });
      
      // Error callback
      onError?.(error);
      
      console.error('❌ Single prediction failed:', error);
    },
  });

  // Batch prediction mutation
  const batchPredictionMutation = useMutation({
    mutationFn: async (request: BatchPredictionRequest): Promise<BatchPredictionResponse> => {
      trackEvent('prediction_started', {
        type: 'batch',
        batch_size: request.patients_data.length,
        model_version: request.model_version,
      });

      const result = await predictionService.makeBatchPrediction(request);
      
      trackEvent('prediction_completed', {
        type: 'batch',
        batch_size: result.total_requested,
        successful_count: result.successful_count,
        failed_count: result.failed_count,
        processing_time_ms: result.processing_time_ms,
      });

      return result;
    },
    onSuccess: (result) => {
      setBatchResult(result);
      setError(null);
      
      queryClient.invalidateQueries(QUERY_KEYS.PREDICTION_HISTORY);
      queryClient.invalidateQueries(QUERY_KEYS.PREDICTION_STATS);
      
      console.log('✅ Batch prediction completed successfully');
      
      // Show success notification with summary
      toast.success(
        `Batch prediction completed! ${result.successful_count}/${result.total_requested} successful predictions`,
        { duration: 6000 }
      );
    },
    onError: (error: any) => {
      const errorMessage = error.message || 'Batch prediction failed';
      setError(errorMessage);
      
      trackEvent('prediction_failed', {
        type: 'batch',
        error_message: errorMessage,
      });
      
      onError?.(error);
      
      console.error('❌ Batch prediction failed:', error);
      toast.error(errorMessage);
    },
  });

  // CSV upload prediction mutation
  const csvUploadMutation = useMutation({
    mutationFn: async ({
      file,
      options,
      onProgress,
    }: {
      file: File;
      options?: { modelVersion?: string; includeConfidence?: boolean };
      onProgress?: (progress: number) => void;
    }): Promise<Blob> => {
      trackEvent('prediction_started', {
        type: 'csv_upload',
        file_size_mb: file.size / (1024 * 1024),
        file_name: file.name,
      });

      const result = await predictionService.uploadCsvForPrediction(file, options, onProgress);
      
      trackEvent('prediction_completed', {
        type: 'csv_upload',
        file_size_mb: file.size / (1024 * 1024),
      });

      return result;
    },
    onSuccess: () => {
      setError(null);
      queryClient.invalidateQueries(QUERY_KEYS.PREDICTION_HISTORY);
      queryClient.invalidateQueries(QUERY_KEYS.PREDICTION_STATS);
      
      console.log('✅ CSV prediction completed successfully');
    },
    onError: (error: any) => {
      const errorMessage = error.message || 'CSV prediction failed';
      setError(errorMessage);
      
      trackEvent('prediction_failed', {
        type: 'csv_upload',
        error_message: errorMessage,
      });
      
      onError?.(error);
      
      console.error('❌ CSV prediction failed:', error);
    },
  });

  // ========================================
  // QUERIES FOR DATA FETCHING
  // ========================================

  // Prediction history query
  const {
    data: predictionHistory,
    refetch: refreshHistory,
    isLoading: historyLoading,
  } = useQuery({
    queryKey: [QUERY_KEYS.PREDICTION_HISTORY],
    queryFn: async () => {
      const response = await predictionService.getPredictionHistory({
        limit: 100,
        offset: 0,
      });
      return response;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
    enabled: enableRealTimeUpdates,
    refetchInterval: enableRealTimeUpdates ? 30000 : false, // 30 seconds
  });

  // Statistics query
  const {
    data: statistics,
    isLoading: statisticsLoading,
  } = useQuery({
    queryKey: [QUERY_KEYS.PREDICTION_STATS, '30d'],
    queryFn: async () => {
      return await predictionService.getPredictionStatistics('30d');
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
    enabled: enableRealTimeUpdates,
  });

  // ========================================
  // CALLBACK FUNCTIONS
  // ========================================

  // Make single prediction
  const makePrediction = useCallback(async (request: PredictionRequest): Promise<PredictionResponse> => {
    // Clear previous error
    setError(null);
    setProgress(0);

    // Validate input if auto-validation is enabled
    if (autoValidate) {
      const validation = predictionService.validatePatientData(request.patient_data);
      if (!validation.isValid) {
        const error = new Error(`Invalid patient data: ${validation.errors.join(', ')}`);
        setError(error.message);
        throw error;
      }
    }

    // Start progress tracking
    setProgress(25);

    try {
      const result = await singlePredictionMutation.mutateAsync(request);
      setProgress(100);
      
      // Reset progress after a short delay
      setTimeout(() => setProgress(0), 1000);
      
      return result;
    } catch (error) {
      setProgress(0);
      throw error;
    }
  }, [autoValidate, singlePredictionMutation]);

  // Make batch prediction
  const makeBatchPrediction = useCallback(async (request: BatchPredictionRequest): Promise<BatchPredictionResponse> => {
    setError(null);
    setProgress(0);

    // Validate batch
    if (request.patients_data.length === 0) {
      const error = new Error('Batch cannot be empty');
      setError(error.message);
      throw error;
    }

    setProgress(10);

    try {
      const result = await batchPredictionMutation.mutateAsync(request);
      setProgress(100);
      
      setTimeout(() => setProgress(0), 1000);
      
      return result;
    } catch (error) {
      setProgress(0);
      throw error;
    }
  }, [batchPredictionMutation]);

  // Upload CSV for prediction
  const uploadCsvPrediction = useCallback(async (
    file: File,
    options: { modelVersion?: string; includeConfidence?: boolean } = {},
    onProgress?: (progress: number) => void
  ): Promise<Blob> => {
    setError(null);
    
    // Combined progress tracking
    const combinedProgress = (uploadProgress: number) => {
      setProgress(uploadProgress);
      onProgress?.(uploadProgress);
    };

    try {
      const result = await csvUploadMutation.mutateAsync({
        file,
        options,
        onProgress: combinedProgress,
      });
      
      setTimeout(() => setProgress(0), 1000);
      
      return result;
    } catch (error) {
      setProgress(0);
      throw error;
    }
  }, [csvUploadMutation]);

  // Clear error state
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Clear all results
  const clearResults = useCallback(() => {
    setPredictionResult(null);
    setBatchResult(null);
    setError(null);
    setProgress(0);
    
    console.log('🧹 Prediction results cleared');
  }, []);

  // Validate patient data
  const validatePatientData = useCallback((data: any) => {
    return predictionService.validatePatientData(data);
  }, []);

  // ========================================
  // EFFECT HOOKS
  // ========================================

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cancel any ongoing requests
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      // Clean up state
      setProgress(0);
    };
  }, []);

  // Error timeout (auto-clear errors after 10 seconds)
  useEffect(() => {
    if (error) {
      const timeout = setTimeout(() => {
        setError(null);
      }, 10000);

      return () => clearTimeout(timeout);
    }
  }, [error]);

  // ========================================
  // COMPUTED VALUES
  // ========================================

  // Overall loading state
  const loading = singlePredictionMutation.isLoading || 
                  batchPredictionMutation.isLoading || 
                  csvUploadMutation.isLoading ||
                  historyLoading ||
                  statisticsLoading;

  return {
    // Core prediction functions
    makePrediction,
    makeBatchPrediction,
    uploadCsvPrediction,
    
    // Results
    predictionResult,
    batchResult,
    
    // State
    loading,
    error,
    progress,
    
    // Data
    predictionHistory,
    statistics,
    
    // Actions
    clearError,
    clearResults,
    refreshHistory,
    validatePatientData,
  };
};

/**
 * Hook for managing prediction history
 */
export const usePredictionHistory = (filters: {
  limit?: number;
  offset?: number;
  timeframe?: string;
  riskCategory?: string;
} = {}) => {
  const [localFilters, setLocalFilters] = useState(filters);
  
  const {
    data: history,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['prediction-history', localFilters],
    queryFn: async () => {
      const startDate = localFilters.timeframe ? 
        new Date(Date.now() - parseInt(localFilters.timeframe) * 24 * 60 * 60 * 1000) : 
        undefined;

      return await predictionService.getPredictionHistory({
        limit: localFilters.limit || 50,
        offset: localFilters.offset || 0,
        startDate,
        riskCategory: localFilters.riskCategory,
      });
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const updateFilters = useCallback((newFilters: typeof filters) => {
    setLocalFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  return {
    history,
    loading: isLoading,
    error,
    refetch,
    filters: localFilters,
    updateFilters,
  };
};

/**
 * Hook for model management
 */
export const useModels = () => {
  const queryClient = useQueryClient();

  // Available models query
  const {
    data: models,
    isLoading: modelsLoading,
    error: modelsError,
    refetch: refetchModels,
  } = useQuery({
    queryKey: ['available-models'],
    queryFn: async () => {
      const response = await apiClient.getAvailableModels();
      return response.data;
    },
    staleTime: 15 * 60 * 1000, // 15 minutes
  });

  // Model performance query (for active model)
  const activeModel = models?.models?.find(m => m.is_default);
  
  const {
    data: performance,
    isLoading: performanceLoading,
  } = useQuery({
    queryKey: ['model-performance', activeModel?.version],
    queryFn: async () => {
      if (!activeModel) return null;
      const response = await apiClient.getModelPerformance(activeModel.version);
      return response.data;
    },
    enabled: !!activeModel,
    staleTime: 30 * 60 * 1000, // 30 minutes
  });

  // Model retraining mutation
  const retrainingMutation = useMutation({
    mutationFn: async (params: Record<string, any> = {}) => {
      const response = await apiClient.triggerModelRetraining(params);
      return response.data;
    },
    onSuccess: () => {
      toast.success('Model retraining started successfully');
      queryClient.invalidateQueries(['available-models']);
    },
    onError: (error: any) => {
      toast.error(`Retraining failed: ${error.message}`);
    },
  });

  return {
    models: models?.models || [],
    defaultModel: models?.default_model,
    modelTypes: models?.model_types || [],
    totalModels: models?.total_models || 0,
    activeModelPerformance: performance,
    
    // State
    loading: modelsLoading || performanceLoading,
    error: modelsError,
    
    // Actions
    refetchModels,
    triggerRetraining: retrainingMutation.mutate,
    retrainingLoading: retrainingMutation.isLoading,
  };
};

/**
 * Hook for real-time prediction monitoring
 */
export const usePredictionMonitoring = (enabled: boolean = false) => {
  const [realtimeStats, setRealtimeStats] = useState<any>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');

  useEffect(() => {
    if (!enabled) return;

    // In a real implementation, this would set up WebSocket connection
    // For now, we'll simulate with polling
    
    const interval = setInterval(async () => {
      try {
        setConnectionStatus('connecting');
        
        const stats = await predictionService.getPredictionStatistics('1h');
        setRealtimeStats(stats);
        setConnectionStatus('connected');
        
      } catch (error) {
        console.error('Real-time monitoring error:', error);
        setConnectionStatus('disconnected');
      }
    }, 30000); // Poll every 30 seconds

    return () => {
      clearInterval(interval);
      setConnectionStatus('disconnected');
    };
  }, [enabled]);

  return {
    realtimeStats,
    connectionStatus,
    isConnected: connectionStatus === 'connected',
  };
};

/**
 * Hook for prediction performance analytics
 */
export const usePredictionAnalytics = (timeframe: string = '30d') => {
  const {
    data: analytics,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['prediction-analytics', timeframe],
    queryFn: async () => {
      const [stats, history] = await Promise.all([
        predictionService.getPredictionStatistics(timeframe),
        predictionService.getPredictionHistory({ limit: 1000 }),
      ]);

      // Calculate additional analytics
      const enhancedAnalytics = {
        ...stats,
        trends: this.calculateTrends(history?.predictions || []),
        insights: this.generateInsights(stats),
      };

      return enhancedAnalytics;
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
  });

  const calculateTrends = useCallback((predictions: any[]) => {
    if (predictions.length < 2) return null;

    // Group by date
    const dailyStats = predictions.reduce((acc, pred) => {
      const date = new Date(pred.created_at).toDateString();
      if (!acc[date]) {
        acc[date] = { total: 0, highRisk: 0, averageScore: 0, scores: [] };
      }
      
      acc[date].total += 1;
      acc[date].scores.push(pred.risk_score);
      
      if (pred.risk_category === 'high') {
        acc[date].highRisk += 1;
      }
      
      return acc;
    }, {} as any);

    // Calculate averages
    Object.keys(dailyStats).forEach(date => {
      const day = dailyStats[date];
      day.averageScore = day.scores.reduce((sum: number, score: number) => sum + score, 0) / day.scores.length;
      day.highRiskPercentage = (day.highRisk / day.total) * 100;
    });

    return dailyStats;
  }, []);

  const generateInsights = useCallback((stats: PredictionStatistics) => {
    const insights: string[] = [];

    if (stats.risk_distribution_percentages.high > 25) {
      insights.push('High percentage of patients classified as high-risk');
    }

    if (stats.risk_statistics.average_risk_score > 0.6) {
      insights.push('Patient population shows elevated cardiovascular risk');
    }

    if (Object.keys(stats.model_usage).length > 1) {
      insights.push('Multiple models are being used for predictions');
    }

    return insights;
  }, []);

  return {
    analytics,
    loading: isLoading,
    error,
    refetch,
    timeframe,
  };
};

/**
 * Hook for prediction feedback
 */
export const usePredictionFeedback = () => {
  const queryClient = useQueryClient();

  const feedbackMutation = useMutation({
    mutationFn: async ({
      predictionId,
      feedback,
    }: {
      predictionId: string;
      feedback: any;
    }) => {
      // This would call a feedback API endpoint
      console.log('Submitting feedback:', { predictionId, feedback });
      
      // Mock API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return { success: true };
    },
    onSuccess: () => {
      toast.success('Feedback submitted successfully');
      queryClient.invalidateQueries(['prediction-history']);
    },
    onError: (error: any) => {
      toast.error(`Failed to submit feedback: ${error.message}`);
    },
  });

  return {
    submitFeedback: feedbackMutation.mutate,
    loading: feedbackMutation.isLoading,
    error: feedbackMutation.error,
  };
};

/**
 * Hook for managing prediction preferences
 */
export const usePredictionPreferences = () => {
  const [preferences, setPreferences] = useState(() => {
    // Load from localStorage
    const stored = localStorage.getItem('cvd_prediction_preferences');
    return stored ? JSON.parse(stored) : {
      defaultModelVersion: null,
      includeConfidenceByDefault: true,
      includeFeaturesbyDefault: true,
      autoSaveResults: true,
      showAdvancedOptions: false,
    };
  });

  // Save preferences to localStorage
  const updatePreferences = useCallback((newPreferences: Partial<typeof preferences>) => {
    const updated = { ...preferences, ...newPreferences };
    setPreferences(updated);
    localStorage.setItem('cvd_prediction_preferences', JSON.stringify(updated));
    
    console.log('💾 Prediction preferences updated:', updated);
  }, [preferences]);

  return {
    preferences,
    updatePreferences,
  };
};

// Default export
export default usePrediction;