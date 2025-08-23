import { useState, useCallback, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';

// Types
import {
  Dataset,
  DataValidationResult,
  DataSyncJob,
  FileUploadState,
  AsyncState,
} from '../types';

// Services
import apiClient from '../services/api';

// Utilities
import { trackEvent } from '../utils/analytics';
import { formatFileSize, getFileExtension } from '../utils/file';

/**
 * Data hook options
 */
interface UseDataOptions {
  enableAutoRefresh?: boolean;
  refreshInterval?: number;
  maxFileSize?: number;
  allowedExtensions?: string[];
}

/**
 * Upload progress tracking
 */
interface UploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'validating' | 'completed' | 'error';
  error?: string;
  result?: any;
}

/**
 * Data validation options
 */
interface ValidationOptions {
  type: 'quick' | 'full' | 'schema_only';
  showWarnings: boolean;
  autoFix: boolean;
}

/**
 * Data hook return type
 */
interface UseDataReturn {
  // Dataset management
  datasets: Dataset[];
  datasetDetails: Dataset | null;
  
  // Upload functionality
  uploadDataset: (file: File, metadata: any, onProgress?: (progress: number) => void) => Promise<any>;
  validateFile: (file: File, options?: ValidationOptions) => Promise<DataValidationResult>;
  uploadProgress: UploadProgress | null;
  
  // CDC synchronization
  syncCdcData: (options?: any) => Promise<DataSyncJob>;
  syncJobStatus: any | null;
  checkSyncStatus: (jobId: string) => Promise<any>;
  
  // Data operations
  exportDataset: (datasetId: string, format?: string) => Promise<void>;
  deleteDataset: (datasetId: string) => Promise<boolean>;
  getDatasetDetails: (datasetId: string, includeSample?: boolean) => Promise<Dataset>;
  
  // Statistics and analytics
  dataStatistics: any | null;
  refreshStatistics: () => void;
  
  // State management
  loading: boolean;
  error: string | null;
  clearError: () => void;
  
  // Utilities
  getSupportedFormats: () => string[];
  validateFileBeforeUpload: (file: File) => { valid: boolean; errors: string[] };
}

/**
 * Main data management hook
 */
export const useData = (options: UseDataOptions = {}): UseDataReturn => {
  const {
    enableAutoRefresh = false,
    refreshInterval = 30000, // 30 seconds
    maxFileSize = 50 * 1024 * 1024, // 50MB
    allowedExtensions = ['csv', 'json', 'parquet'],
  } = options;

  const queryClient = useQueryClient();

  // Local state
  const [datasetDetails, setDatasetDetails] = useState<Dataset | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [syncJobStatus, setSyncJobStatus] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Query keys
  const QUERY_KEYS = {
    DATASETS: 'datasets',
    DATASET_DETAILS: 'dataset-details',
    DATA_STATISTICS: 'data-statistics',
    SYNC_JOBS: 'sync-jobs',
  };

  // ========================================
  // QUERIES
  // ========================================

  // Datasets list query
  const {
    data: datasets = [],
    isLoading: datasetsLoading,
    error: datasetsError,
    refetch: refetchDatasets,
  } = useQuery({
    queryKey: [QUERY_KEYS.DATASETS],
    queryFn: async () => {
      const response = await apiClient.getDatasets();
      return response.data || [];
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
    refetchInterval: enableAutoRefresh ? refreshInterval : false,
  });

  // Data statistics query
  const {
    data: dataStatistics,
    isLoading: statisticsLoading,
    refetch: refetchStatistics,
  } = useQuery({
    queryKey: [QUERY_KEYS.DATA_STATISTICS],
    queryFn: async () => {
      const response = await apiClient.getDataStatistics();
      return response.data;
    },
    staleTime: 15 * 60 * 1000, // 15 minutes
  });

  // ========================================
  // MUTATIONS
  // ========================================

  // Dataset upload mutation
  const uploadMutation = useMutation({
    mutationFn: async ({
      file,
      metadata,
      onProgress,
    }: {
      file: File;
      metadata: any;
      onProgress?: (progress: number) => void;
    }) => {
      // Track upload start
      trackEvent('dataset_upload_started', {
        file_size_mb: file.size / (1024 * 1024),
        file_type: getFileExtension(file.name),
        dataset_type: metadata.datasetType,
      });

      // Update progress state
      setUploadProgress({
        file,
        progress: 0,
        status: 'uploading',
      });

      const result = await apiClient.uploadDataset(file, metadata, (progress) => {
        setUploadProgress(prev => prev ? { ...prev, progress } : null);
        onProgress?.(progress);
      });

      // Update to processing status
      setUploadProgress(prev => prev ? { ...prev, status: 'processing', progress: 100 } : null);

      // Track completion
      trackEvent('dataset_upload_completed', {
        dataset_id: result.data?.dataset_id,
        file_size_mb: file.size / (1024 * 1024),
        validation_score: result.data?.validation_results?.quality_score,
      });

      return result.data;
    },
    onSuccess: (result) => {
      setUploadProgress(prev => prev ? { 
        ...prev, 
        status: 'completed', 
        result 
      } : null);

      // Refresh datasets list
      queryClient.invalidateQueries([QUERY_KEYS.DATASETS]);
      queryClient.invalidateQueries([QUERY_KEYS.DATA_STATISTICS]);

      toast.success('Dataset uploaded successfully!');
      
      // Clear upload progress after delay
      setTimeout(() => setUploadProgress(null), 3000);
    },
    onError: (error: any) => {
      setUploadProgress(prev => prev ? { 
        ...prev, 
        status: 'error', 
        error: error.message 
      } : null);

      setError(error.message);

      trackEvent('dataset_upload_failed', {
        error_message: error.message,
      });

      toast.error(`Upload failed: ${error.message}`);
    },
  });

  // File validation mutation
  const validationMutation = useMutation({
    mutationFn: async ({ file, options }: { file: File; options: ValidationOptions }) => {
      const response = await apiClient.validateDataset(file, options.type);
      return response.data;
    },
  });

  // CDC sync mutation
  const cdcSyncMutation = useMutation({
    mutationFn: async (options: any = {}) => {
      trackEvent('cdc_sync_started', options);
      
      const response = await apiClient.syncCdcData(options);
      return response.data;
    },
    onSuccess: (result) => {
      setSyncJobStatus(result);
      queryClient.invalidateQueries([QUERY_KEYS.DATASETS]);
      
      toast.success('CDC data synchronization started');
      
      trackEvent('cdc_sync_triggered', {
        job_id: result.sync_job_id,
      });
    },
    onError: (error: any) => {
      setError(error.message);
      toast.error(`CDC sync failed: ${error.message}`);
      
      trackEvent('cdc_sync_failed', {
        error_message: error.message,
      });
    },
  });

  // Dataset deletion mutation
  const deleteMutation = useMutation({
    mutationFn: async (datasetId: string) => {
      const response = await apiClient.deleteDataset(datasetId);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries([QUERY_KEYS.DATASETS]);
      queryClient.invalidateQueries([QUERY_KEYS.DATA_STATISTICS]);
      
      toast.success('Dataset deleted successfully');
    },
    onError: (error: any) => {
      toast.error(`Delete failed: ${error.message}`);
    },
  });

  // ========================================
  // CALLBACK FUNCTIONS
  // ========================================

  // Upload dataset with progress tracking
  const uploadDataset = useCallback(async (
    file: File,
    metadata: {
      datasetName: string;
      datasetType: string;
      description?: string;
    },
    onProgress?: (progress: number) => void
  ) => {
    // Pre-upload validation
    const validation = validateFileBeforeUpload(file);
    if (!validation.valid) {
      throw new Error(validation.errors.join(', '));
    }

    return await uploadMutation.mutateAsync({ file, metadata, onProgress });
  }, [uploadMutation]);

  // Validate file
  const validateFile = useCallback(async (
    file: File,
    options: ValidationOptions = {
      type: 'full',
      showWarnings: true,
      autoFix: false,
    }
  ): Promise<DataValidationResult> => {
    try {
      return await validationMutation.mutateAsync({ file, options });
    } catch (error: any) {
      throw new Error(`Validation failed: ${error.message}`);
    }
  }, [validationMutation]);

  // Sync CDC data
  const syncCdcData = useCallback(async (options: any = {}): Promise<DataSyncJob> => {
    return await cdcSyncMutation.mutateAsync(options);
  }, [cdcSyncMutation]);

  // Check sync job status
  const checkSyncStatus = useCallback(async (jobId: string) => {
    try {
      const response = await apiClient.getSyncJobStatus(jobId);
      setSyncJobStatus(response.data);
      return response.data;
    } catch (error: any) {
      console.error('Failed to check sync status:', error);
      throw error;
    }
  }, []);

  // Get dataset details
  const getDatasetDetails = useCallback(async (
    datasetId: string,
    includeSample: boolean = false
  ): Promise<Dataset> => {
    try {
      const response = await apiClient.getDatasetDetails(datasetId, includeSample);
      const details = response.data;
      
      setDatasetDetails(details);
      return details;
    } catch (error: any) {
      console.error('Failed to get dataset details:', error);
      throw error;
    }
  }, []);

  // Export dataset
  const exportDataset = useCallback(async (
    datasetId: string,
    format: string = 'csv'
  ): Promise<void> => {
    try {
      trackEvent('dataset_export_started', { dataset_id: datasetId, format });
      
      const blob = await apiClient.exportDataset(datasetId, format, true);
      
      // Create download
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `dataset_${datasetId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      trackEvent('dataset_export_completed', { dataset_id: datasetId, format });
      toast.success('Dataset exported successfully');

    } catch (error: any) {
      console.error('Export failed:', error);
      toast.error(`Export failed: ${error.message}`);
      
      trackEvent('dataset_export_failed', {
        dataset_id: datasetId,
        format,
        error_message: error.message,
      });
    }
  }, []);

  // Delete dataset
  const deleteDataset = useCallback(async (datasetId: string): Promise<boolean> => {
    try {
      await deleteMutation.mutateAsync(datasetId);
      
      trackEvent('dataset_deleted', { dataset_id: datasetId });
      return true;
    } catch (error: any) {
      console.error('Delete failed:', error);
      return false;
    }
  }, [deleteMutation]);

  // Clear error state
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Get supported file formats
  const getSupportedFormats = useCallback((): string[] => {
    return allowedExtensions;
  }, [allowedExtensions]);

  // Validate file before upload
  const validateFileBeforeUpload = useCallback((file: File): { valid: boolean; errors: string[] } => {
    const errors: string[] = [];

    // Check file size
    if (file.size > maxFileSize) {
      errors.push(`File size (${formatFileSize(file.size)}) exceeds maximum allowed size (${formatFileSize(maxFileSize)})`);
    }

    if (file.size === 0) {
      errors.push('File cannot be empty');
    }

    // Check file extension
    const extension = getFileExtension(file.name);
    if (!allowedExtensions.includes(extension)) {
      errors.push(`File type '${extension}' is not supported. Allowed types: ${allowedExtensions.join(', ')}`);
    }

    // Check file name
    if (file.name.length > 255) {
      errors.push('File name is too long (maximum 255 characters)');
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }, [maxFileSize, allowedExtensions]);

  // ========================================
  // COMPUTED VALUES
  // ========================================

  const loading = datasetsLoading || 
                  statisticsLoading || 
                  uploadMutation.isLoading || 
                  validationMutation.isLoading ||
                  cdcSyncMutation.isLoading ||
                  deleteMutation.isLoading;

  // ========================================
  // EFFECT HOOKS
  // ========================================

  // Auto-refresh sync job status
  useEffect(() => {
    if (syncJobStatus?.sync_job_id && syncJobStatus?.status === 'running') {
      const interval = setInterval(async () => {
        try {
          await checkSyncStatus(syncJobStatus.sync_job_id);
        } catch (error) {
          console.error('Failed to check sync status:', error);
        }
      }, 5000); // Check every 5 seconds

      return () => clearInterval(interval);
    }
  }, [syncJobStatus, checkSyncStatus]);

  // Clear error after timeout
  useEffect(() => {
    if (error) {
      const timeout = setTimeout(() => {
        setError(null);
      }, 8000);

      return () => clearTimeout(timeout);
    }
  }, [error]);

  return {
    // Data
    datasets,
    datasetDetails,
    dataStatistics,
    
    // Upload
    uploadDataset,
    validateFile,
    uploadProgress,
    
    // CDC sync
    syncCdcData,
    syncJobStatus,
    checkSyncStatus,
    
    // Operations
    exportDataset,
    deleteDataset,
    getDatasetDetails,
    
    // Statistics
    refreshStatistics: refetchStatistics,
    
    // State
    loading,
    error: error || datasetsError?.message || null,
    clearError,
    
    // Utilities
    getSupportedFormats,
    validateFileBeforeUpload,
  };
};

/**
 * Hook for dataset upload with advanced features
 */
export const useDatasetUpload = () => {
  const [uploads, setUploads] = useState<Map<string, UploadProgress>>(new Map());
  const [dragActive, setDragActive] = useState(false);

  // Add upload to tracking
  const addUpload = useCallback((file: File): string => {
    const uploadId = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    setUploads(prev => new Map(prev).set(uploadId, {
      file,
      progress: 0,
      status: 'pending',
    }));

    return uploadId;
  }, []);

  // Update upload progress
  const updateUpload = useCallback((uploadId: string, update: Partial<UploadProgress>) => {
    setUploads(prev => {
      const newMap = new Map(prev);
      const existing = newMap.get(uploadId);
      
      if (existing) {
        newMap.set(uploadId, { ...existing, ...update });
      }
      
      return newMap;
    });
  }, []);

  // Remove upload from tracking
  const removeUpload = useCallback((uploadId: string) => {
    setUploads(prev => {
      const newMap = new Map(prev);
      newMap.delete(uploadId);
      return newMap;
    });
  }, []);

  // Handle file drop
  const handleFileDrop = useCallback((files: FileList | File[]) => {
    const fileArray = Array.from(files);
    
    fileArray.forEach(file => {
      const uploadId = addUpload(file);
      
      // Start upload process
      setTimeout(() => {
        updateUpload(uploadId, { status: 'uploading' });
      }, 100);
    });

    setDragActive(false);
  }, [addUpload, updateUpload]);

  // Drag and drop handlers
  const dragHandlers = {
    onDragEnter: useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(true);
    }, []),
    
    onDragLeave: useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
    }, []),
    
    onDragOver: useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
    }, []),
    
    onDrop: useCallback((e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      
      const files = e.dataTransfer.files;
      if (files) {
        handleFileDrop(files);
      }
    }, [handleFileDrop]),
  };

  return {
    uploads: Array.from(uploads.values()),
    dragActive,
    dragHandlers,
    addUpload,
    updateUpload,
    removeUpload,
    handleFileDrop,
  };
};

/**
 * Hook for data quality monitoring
 */
export const useDataQuality = (datasetId?: string) => {
  const [qualityReport, setQualityReport] = useState<any | null>(null);

  const {
    data: qualityData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['data-quality', datasetId],
    queryFn: async () => {
      if (!datasetId) return null;
      
      // This would call a data quality analysis endpoint
      // For now, return mock data
      return {
        overall_score: 0.85,
        completeness: 0.92,
        accuracy: 0.78,
        consistency: 0.88,
        issues: [
          { type: 'missing_values', field: 'cholesterol', count: 120 },
          { type: 'outliers', field: 'age', count: 15 },
        ],
        recommendations: [
          'Consider imputation for missing cholesterol values',
          'Review age outliers for data entry errors',
        ],
      };
    },
    enabled: !!datasetId,
    staleTime: 30 * 60 * 1000, // 30 minutes
  });

  useEffect(() => {
    if (qualityData) {
      setQualityReport(qualityData);
    }
  }, [qualityData]);

  return {
    qualityReport,
    loading: isLoading,
    error,
    refreshQuality: refetch,
  };
};

/**
 * Hook for CDC data synchronization monitoring
 */
export const useCdcSync = () => {
  const [activeSyncJobs, setActiveSyncJobs] = useState<Map<string, any>>(new Map());

  const startSync = useCallback(async (options: any = {}) => {
    try {
      const syncResult = await apiClient.syncCdcData(options);
      const jobId = syncResult.data?.sync_job_id;

      if (jobId) {
        setActiveSyncJobs(prev => new Map(prev).set(jobId, {
          ...syncResult.data,
          startTime: Date.now(),
        }));

        // Start monitoring this job
        const monitor = setInterval(async () => {
          try {
            const statusResponse = await apiClient.getSyncJobStatus(jobId);
            const status = statusResponse.data;

            setActiveSyncJobs(prev => {
              const newMap = new Map(prev);
              newMap.set(jobId, { ...status, startTime: prev.get(jobId)?.startTime });
              return newMap;
            });

            // Stop monitoring if job is complete
            if (status.status === 'completed' || status.status === 'failed') {
              clearInterval(monitor);
              
              // Remove from active jobs after delay
              setTimeout(() => {
                setActiveSyncJobs(prev => {
                  const newMap = new Map(prev);
                  newMap.delete(jobId);
                  return newMap;
                });
              }, 5000);
            }

          } catch (error) {
            console.error('Sync status check failed:', error);
            clearInterval(monitor);
          }
        }, 3000); // Check every 3 seconds

        return syncResult.data;
      }

      throw new Error('No job ID returned from sync request');

    } catch (error: any) {
      console.error('CDC sync failed:', error);
      throw error;
    }
  }, []);

  return {
    activeSyncJobs: Array.from(activeSyncJobs.values()),
    startSync,
    hasActiveSyncs: activeSyncJobs.size > 0,
  };
};

/**
 * Hook for data transformation and processing
 */
export const useDataProcessing = () => {
  const [processingJobs, setProcessingJobs] = useState<any[]>([]);

  const startProcessing = useCallback(async (
    datasetId: string,
    processingConfig: any
  ) => {
    try {
      // This would call a data processing API endpoint
      const jobId = `proc_${Date.now()}`;
      
      const newJob = {
        job_id: jobId,
        dataset_id: datasetId,
        status: 'running',
        progress: 0,
        config: processingConfig,
        started_at: new Date().toISOString(),
      };

      setProcessingJobs(prev => [...prev, newJob]);

      // Simulate processing progress
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 20;
        
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          
          setProcessingJobs(prev => prev.map(job => 
            job.job_id === jobId 
              ? { ...job, status: 'completed', progress: 100, completed_at: new Date().toISOString() }
              : job
          ));

          toast.success('Data processing completed');
        } else {
          setProcessingJobs(prev => prev.map(job => 
            job.job_id === jobId ? { ...job, progress } : job
          ));
        }
      }, 1000);

      return newJob;

    } catch (error: any) {
      console.error('Data processing failed:', error);
      throw error;
    }
  }, []);

  return {
    processingJobs,
    startProcessing,
    hasActiveJobs: processingJobs.some(job => job.status === 'running'),
  };
};

// Default export
export default useData;