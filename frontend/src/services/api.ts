import axios, { 
  AxiosInstance, 
  AxiosRequestConfig, 
  AxiosResponse, 
  AxiosError,
  InternalAxiosRequestConfig 
} from 'axios';
import { v4 as uuidv4 } from 'uuid';

// Types
import { 
  ApiResponse, 
  ApiError, 
  RequestConfig, 
  PredictionRequest,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  ModelInfo,
  ModelPerformance,
  Dataset,
  HealthStatus,
  PredictionStatistics
} from '../types';

// Utilities
import { getStoredAuth, clearStoredAuth, storeAuth } from '../utils/auth';
import { createCorrelationId, isNetworkError, shouldRetry } from '../utils/api';

/**
 * API Client Configuration
 */
interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  retryDelay: number;
  cacheTimeout: number;
}

/**
 * Request Cache Entry
 */
interface CacheEntry<T = any> {
  data: T;
  timestamp: number;
  ttl: number;
}

/**
 * Main API Client Class
 */
class ApiClient {
  private axiosInstance: AxiosInstance;
  private config: ApiClientConfig;
  private requestCache = new Map<string, CacheEntry>();
  private retryQueue = new Set<string>();

  constructor(config: Partial<ApiClientConfig> = {}) {
    this.config = {
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
      timeout: 30000, // 30 seconds
      retries: 3,
      retryDelay: 1000, // 1 second
      cacheTimeout: 5 * 60 * 1000, // 5 minutes
      ...config,
    };

    this.axiosInstance = this.createAxiosInstance();
    this.setupInterceptors();
  }

  /**
   * Initialize API client
   */
  async initialize(): Promise<void> {
    try {
      console.log('🔗 Initializing API client:', this.config.baseURL);
      
      // Check API health
      await this.healthCheck();
      
      // Set up authentication if available
      const auth = getStoredAuth();
      if (auth?.token) {
        this.setAuthToken(auth.token);
      }

      console.log('✅ API client initialized successfully');
    } catch (error) {
      console.error('❌ API client initialization failed:', error);
      throw error;
    }
  }

  /**
   * Create axios instance with default configuration
   */
  private createAxiosInstance(): AxiosInstance {
    return axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });
  }

  /**
   * Setup request and response interceptors
   */
  private setupInterceptors(): void {
    // Request interceptor
    this.axiosInstance.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add correlation ID for request tracing
        const correlationId = createCorrelationId();
        config.headers['X-Correlation-ID'] = correlationId;

        // Add timestamp for performance monitoring
        config.metadata = { startTime: Date.now(), correlationId };

        // Log request in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`🌐 API Request [${correlationId}]:`, {
            method: config.method?.toUpperCase(),
            url: config.url,
            data: config.data,
          });
        }

        return config;
      },
      (error: AxiosError) => {
        console.error('❌ Request configuration error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.axiosInstance.interceptors.response.use(
      (response: AxiosResponse) => {
        // Calculate request duration
        const duration = Date.now() - (response.config.metadata?.startTime || 0);
        const correlationId = response.config.metadata?.correlationId;

        // Log response in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`✅ API Response [${correlationId}]:`, {
            status: response.status,
            duration: `${duration}ms`,
            data: response.data,
          });
        }

        // Add performance metadata
        response.metadata = {
          correlationId,
          duration,
          timestamp: new Date().toISOString(),
        };

        return response;
      },
      async (error: AxiosError) => {
        const correlationId = error.config?.metadata?.correlationId;
        
        console.error(`❌ API Error [${correlationId}]:`, {
          status: error.response?.status,
          message: error.message,
          url: error.config?.url,
          data: error.response?.data,
        });

        // Handle specific error cases
        if (error.response?.status === 401) {
          // Unauthorized - clear auth and redirect
          clearStoredAuth();
          window.location.href = '/login';
          return Promise.reject(error);
        }

        if (error.response?.status === 403) {
          // Forbidden - show permission error
          const customError = new Error('You do not have permission to perform this action.');
          customError.name = 'PermissionError';
          return Promise.reject(customError);
        }

        // Retry logic for network errors
        if (shouldRetry(error) && error.config && !this.retryQueue.has(error.config.url || '')) {
          return this.retryRequest(error);
        }

        // Transform error for consistent handling
        const apiError: ApiError = {
          error: error.response?.data?.error || 'Request failed',
          message: error.response?.data?.message || error.message,
          correlation_id: correlationId,
          details: error.response?.data?.details,
          timestamp: new Date().toISOString(),
        };

        return Promise.reject(apiError);
      }
    );
  }

  /**
   * Retry failed requests with exponential backoff
   */
  private async retryRequest(error: AxiosError): Promise<AxiosResponse> {
    const config = error.config!;
    const url = config.url || '';
    
    this.retryQueue.add(url);

    try {
      // Exponential backoff
      const delay = this.config.retryDelay * Math.pow(2, (config as any).retryCount || 0);
      await new Promise(resolve => setTimeout(resolve, delay));

      // Increment retry count
      (config as any).retryCount = ((config as any).retryCount || 0) + 1;

      // Retry the request
      const response = await this.axiosInstance.request(config);
      this.retryQueue.delete(url);
      return response;

    } catch (retryError) {
      this.retryQueue.delete(url);
      throw retryError;
    }
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string): void {
    this.axiosInstance.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  /**
   * Clear authentication token
   */
  clearAuthToken(): void {
    delete this.axiosInstance.defaults.headers.common['Authorization'];
  }

  /**
   * Generic request method with caching
   */
  private async request<T = any>(
    config: RequestConfig,
    useCache: boolean = false
  ): Promise<ApiResponse<T>> {
    try {
      // Check cache first
      if (useCache && config.method === 'GET') {
        const cacheKey = this.generateCacheKey(config);
        const cached = this.getFromCache<T>(cacheKey);
        
        if (cached) {
          console.log(`💾 Cache hit for: ${config.url}`);
          return { data: cached };
        }
      }

      // Make request
      const response = await this.axiosInstance.request<T>({
        method: config.method,
        url: config.url,
        data: config.data,
        params: config.params,
        headers: config.headers,
        timeout: config.timeout || this.config.timeout,
        onUploadProgress: config.onUploadProgress,
      });

      // Cache GET requests
      if (useCache && config.method === 'GET' && response.data) {
        const cacheKey = this.generateCacheKey(config);
        this.setCache(cacheKey, response.data, this.config.cacheTimeout);
      }

      return {
        data: response.data,
        correlation_id: response.headers['x-correlation-id'],
        timestamp: new Date().toISOString(),
      };

    } catch (error: any) {
      // Re-throw with additional context
      throw {
        ...error,
        url: config.url,
        method: config.method,
      };
    }
  }

  /**
   * GET request with caching
   */
  async get<T = any>(url: string, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.request<T>({
      method: 'GET',
      url,
      ...config,
    }, true);
  }

  /**
   * POST request
   */
  async post<T = any>(url: string, data?: any, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.request<T>({
      method: 'POST',
      url,
      data,
      ...config,
    });
  }

  /**
   * PUT request
   */
  async put<T = any>(url: string, data?: any, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.request<T>({
      method: 'PUT',
      url,
      data,
      ...config,
    });
  }

  /**
   * DELETE request
   */
  async delete<T = any>(url: string, config?: Partial<RequestConfig>): Promise<ApiResponse<T>> {
    return this.request<T>({
      method: 'DELETE',
      url,
      ...config,
    });
  }

  /**
   * File upload with progress tracking
   */
  async uploadFile<T = any>(
    url: string,
    file: File,
    additionalData?: Record<string, any>,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<T>> {
    const formData = new FormData();
    formData.append('file', file);

    // Add additional form data
    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, value);
      });
    }

    return this.request<T>({
      method: 'POST',
      url,
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
  }

  // ========================================
  // HEALTH CHECK ENDPOINTS
  // ========================================

  /**
   * Basic health check
   */
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    return this.get('/api/v1/health');
  }

  /**
   * Detailed health check
   */
  async detailedHealthCheck(): Promise<ApiResponse<HealthStatus>> {
    return this.get('/api/v1/health/detailed');
  }

  // ========================================
  // PREDICTION ENDPOINTS
  // ========================================

  /**
   * Make single prediction
   */
  async makePrediction(request: PredictionRequest): Promise<ApiResponse<PredictionResponse>> {
    return this.post('/api/v1/predict', request);
  }

  /**
   * Make batch predictions
   */
  async makeBatchPrediction(request: BatchPredictionRequest): Promise<ApiResponse<BatchPredictionResponse>> {
    return this.post('/api/v1/predict/batch', request);
  }

  /**
   * Upload CSV for predictions
   */
  async uploadCsvForPrediction(
    file: File,
    options: {
      modelVersion?: string;
      includeConfidence?: boolean;
    } = {},
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<Blob>> {
    const params = new URLSearchParams();
    if (options.modelVersion) params.append('model_version', options.modelVersion);
    if (options.includeConfidence) params.append('include_confidence', 'true');

    const response = await this.axiosInstance.post(
      `/api/v1/predict/upload?${params.toString()}`,
      (() => {
        const formData = new FormData();
        formData.append('file', file);
        return formData;
      })(),
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      }
    );

    return { data: response.data };
  }

  /**
   * Get prediction history
   */
  async getPredictionHistory(params: {
    limit?: number;
    offset?: number;
    startDate?: string;
    endDate?: string;
    riskCategory?: string;
  } = {}): Promise<ApiResponse<any>> {
    return this.get('/api/v1/history', { params });
  }

  /**
   * Get prediction statistics
   */
  async getPredictionStatistics(timeframe: string = '30d'): Promise<ApiResponse<PredictionStatistics>> {
    return this.get('/api/v1/stats', { params: { timeframe } });
  }

  /**
   * Delete prediction
   */
  async deletePrediction(predictionId: string): Promise<ApiResponse<{ message: string }>> {
    return this.delete(`/api/v1/predictions/${predictionId}`);
  }

  // ========================================
  // MODEL MANAGEMENT ENDPOINTS
  // ========================================

  /**
   * Get available models
   */
  async getAvailableModels(): Promise<ApiResponse<{
    models: ModelInfo[];
    default_model: string;
    model_types: string[];
    total_models: number;
  }>> {
    return this.get('/api/v1/models', undefined, true); // Use cache
  }

  /**
   * Get model performance
   */
  async getModelPerformance(modelVersion: string): Promise<ApiResponse<ModelPerformance>> {
    return this.get(`/api/v1/models/${modelVersion}/performance`, undefined, true);
  }

  /**
   * Trigger model retraining
   */
  async triggerModelRetraining(params: Record<string, any> = {}): Promise<ApiResponse<{
    job_id: string;
    status: string;
    estimated_duration: string;
  }>> {
    return this.post('/api/v1/models/retrain', params);
  }

  /**
   * Get retraining job status
   */
  async getRetrainingJobStatus(jobId: string): Promise<ApiResponse<{
    job_id: string;
    status: string;
    progress: number;
    message: string;
  }>> {
    return this.get(`/api/v1/models/retrain/${jobId}/status`);
  }

  // ========================================
  // DATA MANAGEMENT ENDPOINTS
  // ========================================

  /**
   * Upload dataset
   */
  async uploadDataset(
    file: File,
    metadata: {
      datasetName: string;
      datasetType: string;
      description?: string;
    },
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<{
    dataset_id: string;
    validation_results: any;
  }>> {
    return this.uploadFile('/api/v1/data/upload', file, metadata, onProgress);
  }

  /**
   * Get datasets list
   */
  async getDatasets(params: {
    datasetType?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<ApiResponse<Dataset[]>> {
    return this.get('/api/v1/data/datasets', { params }, true);
  }

  /**
   * Get dataset details
   */
  async getDatasetDetails(
    datasetId: string, 
    includeSample: boolean = false
  ): Promise<ApiResponse<Dataset>> {
    return this.get(`/api/v1/data/datasets/${datasetId}`, {
      params: { include_sample: includeSample }
    });
  }

  /**
   * Validate dataset
   */
  async validateDataset(
    file: File,
    validationType: string = 'full'
  ): Promise<ApiResponse<any>> {
    return this.uploadFile('/api/v1/data/validate', file, { validation_type: validationType });
  }

  /**
   * Sync CDC data
   */
  async syncCdcData(params: {
    forceUpdate?: boolean;
    dataRange?: { start: string; end: string };
  } = {}): Promise<ApiResponse<{
    sync_job_id: string;
    status: string;
    estimated_duration: string;
  }>> {
    return this.post('/api/v1/data/sync/cdc', params);
  }

  /**
   * Get sync job status
   */
  async getSyncJobStatus(jobId: string): Promise<ApiResponse<any>> {
    return this.get(`/api/v1/data/sync/${jobId}/status`);
  }

  /**
   * Export dataset
   */
  async exportDataset(
    datasetId: string,
    format: string = 'csv',
    includeMetadata: boolean = true
  ): Promise<Blob> {
    const response = await this.axiosInstance.get(
      `/api/v1/data/export/${datasetId}`,
      {
        params: { format, include_metadata: includeMetadata },
        responseType: 'blob',
      }
    );

    return response.data;
  }

  /**
   * Delete dataset
   */
  async deleteDataset(datasetId: string): Promise<ApiResponse<{ message: string }>> {
    return this.delete(`/api/v1/data/datasets/${datasetId}`);
  }

  /**
   * Get data statistics
   */
  async getDataStatistics(): Promise<ApiResponse<any>> {
    return this.get('/api/v1/data/stats', undefined, true);
  }

  // ========================================
  // CACHE MANAGEMENT
  // ========================================

  /**
   * Generate cache key for request
   */
  private generateCacheKey(config: RequestConfig): string {
    const key = `${config.method}_${config.url}_${JSON.stringify(config.params || {})}`;
    return btoa(key).replace(/[^a-zA-Z0-9]/g, '');
  }

  /**
   * Get data from cache
   */
  private getFromCache<T>(key: string): T | null {
    const entry = this.requestCache.get(key);
    
    if (entry && Date.now() - entry.timestamp < entry.ttl) {
      return entry.data;
    }

    // Remove expired entry
    if (entry) {
      this.requestCache.delete(key);
    }

    return null;
  }

  /**
   * Set data in cache
   */
  private setCache<T>(key: string, data: T, ttl: number): void {
    this.requestCache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });

    // Clean up old cache entries periodically
    if (this.requestCache.size > 100) {
      this.cleanupCache();
    }
  }

  /**
   * Clean up expired cache entries
   */
  private cleanupCache(): void {
    const now = Date.now();
    
    for (const [key, entry] of this.requestCache.entries()) {
      if (now - entry.timestamp >= entry.ttl) {
        this.requestCache.delete(key);
      }
    }
  }

  /**
   * Clear all cache
   */
  clearCache(): void {
    this.requestCache.clear();
    console.log('🧹 API cache cleared');
  }

  // ========================================
  // UTILITY METHODS
  // ========================================

  /**
   * Check if API is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get API version info
   */
  async getVersionInfo(): Promise<ApiResponse<{
    version: string;
    environment: string;
    build_date: string;
  }>> {
    return this.get('/', undefined, true);
  }

  /**
   * Download file from URL
   */
  async downloadFile(url: string, filename?: string): Promise<void> {
    try {
      const response = await this.axiosInstance.get(url, {
        responseType: 'blob',
      });

      // Create download link
      const blob = new Blob([response.data]);
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      
      link.href = downloadUrl;
      link.download = filename || 'download';
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);

    } catch (error) {
      console.error('File download failed:', error);
      throw error;
    }
  }

  /**
   * Get network status
   */
  getNetworkStatus(): {
    online: boolean;
    effectiveType?: string;
    downlink?: number;
    rtt?: number;
  } {
    return {
      online: navigator.onLine,
      effectiveType: (navigator as any).connection?.effectiveType,
      downlink: (navigator as any).connection?.downlink,
      rtt: (navigator as any).connection?.rtt,
    };
  }

  /**
   * Get request statistics
   */
  getRequestStats(): {
    cacheSize: number;
    retryQueueSize: number;
    lastRequest?: string;
  } {
    return {
      cacheSize: this.requestCache.size,
      retryQueueSize: this.retryQueue.size,
      lastRequest: new Date().toISOString(),
    };
  }
}

// Create singleton instance
const apiClient = new ApiClient({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
});

// Helper functions for common API patterns
export const withRetry = async <T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      if (attempt === maxRetries) {
        throw error;
      }
      
      console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay * attempt));
    }
  }
  
  throw new Error('Max retries exceeded');
};

export const withTimeout = async <T>(
  operation: () => Promise<T>,
  timeoutMs: number = 30000
): Promise<T> => {
  return Promise.race([
    operation(),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('Operation timed out')), timeoutMs)
    ),
  ]);
};

// Export API client and utilities
export { apiClient, ApiClient };
export default apiClient;