import { Theme } from '@mui/material/styles';

// ========================================
// THEME AND UI TYPES
// ========================================

export type AppTheme = Theme;

export type ThemeMode = 'light' | 'dark';

export interface AppConfig {
  version: string;
  buildDate: string;
  environment: 'development' | 'staging' | 'production';
  apiUrl: string;
  features: {
    darkMode: boolean;
    notifications: boolean;
    analytics: boolean;
    devTools: boolean;
  };
}

export interface LoadingState {
  isLoading: boolean;
  message?: string;
  progress?: number;
}

export interface ErrorState {
  hasError: boolean;
  error?: Error | null;
  errorInfo?: string;
  correlationId?: string;
}

// ========================================
// API TYPES
// ========================================

export interface ApiResponse<T = any> {
  data?: T;
  message?: string;
  error?: string;
  correlation_id?: string;
  timestamp?: string;
}

export interface ApiError {
  error: string;
  message: string;
  correlation_id?: string;
  details?: Record<string, any>;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total_count: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

// ========================================
// PATIENT TYPES
// ========================================

export type GenderType = 'male' | 'female' | 'other' | 'unknown';

export type RaceEthnicityType = 
  | 'white_non_hispanic'
  | 'black_non_hispanic' 
  | 'hispanic'
  | 'asian_non_hispanic'
  | 'american_indian_alaska_native'
  | 'native_hawaiian_pacific_islander'
  | 'multiracial'
  | 'other'
  | 'unknown';

export type SmokingStatusType = 'never' | 'former' | 'current' | 'unknown';

export interface PatientData {
  // Demographics
  age: number;
  gender: GenderType;
  race_ethnicity?: RaceEthnicityType;
  
  // Geographic
  state_code?: string;
  zip_code?: string;
  urban_rural_code?: string;
  
  // Medical History
  has_hypertension: boolean;
  has_diabetes: boolean;
  has_heart_disease: boolean;
  has_stroke_history: boolean;
  has_heart_attack_history: boolean;
  has_atrial_fibrillation: boolean;
  has_heart_failure: boolean;
  has_kidney_disease: boolean;
  has_copd: boolean;
  has_depression: boolean;
  
  // Risk Factors
  smoking_status?: SmokingStatusType;
  bmi?: number;
  
  // Vital Signs
  systolic_bp?: number;
  diastolic_bp?: number;
  heart_rate?: number;
  
  // Lab Values
  total_cholesterol?: number;
  hdl_cholesterol?: number;
  ldl_cholesterol?: number;
  triglycerides?: number;
  blood_glucose?: number;
  hba1c?: number;
  creatinine?: number;
  
  // Utilization
  cardiovascular_hospitalizations_last_year: number;
  total_hospitalizations_last_year: number;
  
  // Insurance
  medicare_part_a: boolean;
  medicare_part_b: boolean;
  medicare_part_c: boolean;
  medicare_part_d: boolean;
  dual_eligible: boolean;
  
  // Additional
  additional_features?: Record<string, string | number | boolean>;
}

export interface PatientFormData extends Partial<PatientData> {
  // Form-specific fields
  includeConfidence?: boolean;
  includeFeatures?: boolean;
  modelVersion?: string;
}

// ========================================
// PREDICTION TYPES
// ========================================

export type RiskCategoryType = 'low' | 'medium' | 'high';

export interface FeatureImportance {
  feature_name: string;
  importance_score: number;
  description?: string;
  category?: string;
}

export interface PredictionRequest {
  patient_data: PatientData;
  model_version?: string;
  include_confidence: boolean;
  include_features: boolean;
}

export interface PredictionResponse {
  // Core prediction results
  risk_score: number;
  risk_category: RiskCategoryType;
  confidence_score?: number;
  
  // Confidence intervals
  confidence_lower?: number;
  confidence_upper?: number;
  confidence_level?: number;
  
  // Feature importance
  feature_importance?: FeatureImportance[];
  
  // Model information
  model_version: string;
  model_type: string;
  
  // Metadata
  prediction_id: string;
  prediction_time_ms: number;
  timestamp: string;
  correlation_id?: string;
  
  // Clinical insights
  recommendations?: string[];
  risk_factors?: string[];
}

export interface BatchPredictionRequest {
  patients_data: PatientData[];
  model_version?: string;
  include_confidence: boolean;
  parallel_processing: boolean;
  batch_name?: string;
}

export interface BatchPredictionItem {
  patient_index: number;
  prediction?: PredictionResponse;
  error?: string;
}

export interface BatchPredictionResponse {
  predictions: BatchPredictionItem[];
  failed_predictions: BatchPredictionItem[];
  summary: Record<string, any>;
  batch_id: string;
  total_requested: number;
  successful_count: number;
  failed_count: number;
  processing_time_ms: number;
  model_version: string;
  timestamp: string;
  correlation_id?: string;
}

// ========================================
// MODEL TYPES
// ========================================

export type ModelType = 'linear' | 'ridge' | 'lasso' | 'random_forest' | 'xgboost' | 'lightgbm';
export type ModelStatus = 'training' | 'active' | 'deprecated' | 'failed';

export interface ModelInfo {
  id: string;
  version: string;
  name: string;
  model_type: ModelType;
  description?: string;
  is_active: boolean;
  is_default: boolean;
  status: ModelStatus;
  r2_score?: number;
  mse_score?: number;
  cross_val_score?: number;
  training_data_size?: number;
  created_at: string;
  deployed_at?: string;
}

export interface ModelPerformance {
  model_id: string;
  model_version: string;
  model_type: string;
  r2_score: number;
  mse_score: number;
  mae_score: number;
  rmse_score: number;
  accuracy_score?: number;
  precision_score?: number;
  recall_score?: number;
  f1_score?: number;
  auc_score?: number;
  cross_validation: Record<string, any>;
  feature_importance: FeatureImportance[];
  training_info: Record<string, any>;
  evaluation_date: string;
  evaluation_method: string;
  test_data_size: number;
}

export interface ModelTrainingJob {
  job_id: string;
  job_status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  message: string;
  estimated_duration_minutes?: number;
  estimated_completion?: string;
  progress_percentage: number;
  current_step?: string;
  created_at: string;
  correlation_id?: string;
}

// ========================================
// DATA TYPES
// ========================================

export type DatasetType = 'training' | 'validation' | 'test';
export type ProcessingStatus = 'uploaded' | 'processing' | 'processed' | 'failed';
export type ValidationStatus = 'pending' | 'passed' | 'failed';

export interface Dataset {
  id: string;
  name: string;
  description?: string;
  dataset_type: DatasetType;
  total_records: number;
  total_columns: number;
  file_size_mb: number;
  completeness_score?: number;
  validation_status: ValidationStatus;
  processing_status: ProcessingStatus;
  created_at: string;
  last_used_date?: string;
  usage_count: number;
  owner_id: string;
  sample_data?: {
    columns: string[];
    sample_rows: Record<string, any>[];
    data_types: Record<string, string>;
    sample_size: number;
  };
}

export interface DataValidationResult {
  is_valid: boolean;
  quality_score: number;
  errors: Array<{ field: string; message: string }>;
  warnings: Array<{ field: string; message: string }>;
  field_validation: Record<string, any>;
  required_fields: string[];
  missing_fields: string[];
  total_records: number;
  valid_records: number;
  invalid_records: number;
  recommendations: string[];
}

export interface DataSyncJob {
  sync_job_id: string;
  status: 'started' | 'running' | 'completed' | 'failed';
  message: string;
  estimated_duration?: string;
  data_sources?: string[];
  last_sync_time?: string;
  correlation_id?: string;
}

// ========================================
// ANALYTICS TYPES
// ========================================

export interface PredictionStatistics {
  total_predictions: number;
  timeframe: string;
  period: {
    start_date: string;
    end_date: string;
  };
  risk_distribution: {
    low: number;
    medium: number;
    high: number;
  };
  risk_distribution_percentages: {
    low: number;
    medium: number;
    high: number;
  };
  risk_statistics: {
    average_risk_score: number;
    median_risk_score: number;
    std_risk_score: number;
    min_risk_score: number;
    max_risk_score: number;
  };
  model_usage: Record<string, number>;
  trends: Record<string, any>;
  performance_metrics: Record<string, any>;
  generated_at: string;
}

export interface HealthStatus {
  overall_status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  correlation_id: string;
  components: Record<string, ComponentHealth>;
  system_info: Record<string, any>;
  performance_metrics: Record<string, any>;
  warnings: string[];
  errors: string[];
}

export interface ComponentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  message: string;
  response_time?: string;
  details?: Record<string, any>;
  error?: string;
}

// ========================================
// FORM TYPES
// ========================================

export interface FormFieldError {
  field: string;
  message: string;
}

export interface FormState<T = Record<string, any>> {
  values: T;
  errors: Record<string, string>;
  touched: Record<string, boolean>;
  isSubmitting: boolean;
  isValid: boolean;
}

export interface SelectOption {
  value: string | number;
  label: string;
  description?: string;
  disabled?: boolean;
}

export interface FileUploadState {
  file: File | null;
  progress: number;
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  result?: any;
}

// ========================================
// NAVIGATION TYPES
// ========================================

export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: React.ComponentType;
  description?: string;
  badge?: string | number;
  children?: NavigationItem[];
  permissions?: string[];
}

export interface BreadcrumbItem {
  label: string;
  path?: string;
  icon?: React.ComponentType;
}

// ========================================
// CHART TYPES
// ========================================

export interface ChartDataPoint {
  x: string | number | Date;
  y: number;
  label?: string;
  color?: string;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
  color?: string;
  type?: 'line' | 'bar' | 'area' | 'scatter';
}

export interface ChartConfig {
  type: 'line' | 'bar' | 'pie' | 'doughnut' | 'scatter' | 'area';
  title?: string;
  subtitle?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  showLegend?: boolean;
  showTooltip?: boolean;
  responsive?: boolean;
  height?: number;
  width?: number;
}

// ========================================
// TABLE TYPES
// ========================================

export interface TableColumn<T = any> {
  id: keyof T;
  label: string;
  minWidth?: number;
  align?: 'left' | 'center' | 'right';
  format?: (value: any) => string;
  sortable?: boolean;
  filterable?: boolean;
  renderCell?: (value: any, row: T) => React.ReactNode;
}

export interface TableSort {
  field: string;
  direction: 'asc' | 'desc';
}

export interface TableFilter {
  field: string;
  operator: 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'contains' | 'in';
  value: any;
}

export interface TableState<T = any> {
  data: T[];
  loading: boolean;
  error?: string;
  sort?: TableSort;
  filters: TableFilter[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
  };
  selection: string[];
}

// ========================================
// NOTIFICATION TYPES
// ========================================

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
  timestamp: Date;
}

// ========================================
// USER TYPES
// ========================================

export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  roles: string[];
  permissions: string[];
  organization?: string;
  department?: string;
  is_active: boolean;
  last_login?: string;
  created_at: string;
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  loading: boolean;
  error?: string;
}

// ========================================
// PREDICTION HISTORY TYPES
// ========================================

export interface PredictionHistoryItem {
  prediction_id: string;
  risk_score: number;
  risk_category: RiskCategoryType;
  model_version: string;
  created_at: string;
  patient_age: number;
  patient_gender: string;
  has_feedback: boolean;
  actual_outcome?: boolean;
}

export interface PredictionHistory {
  predictions: PredictionHistoryItem[];
  total_count: number;
  summary: Record<string, any>;
  limit: number;
  offset: number;
  has_more: boolean;
}

// ========================================
// COMPONENT PROP TYPES
// ========================================

export interface BaseComponentProps {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}

export interface CardProps extends BaseComponentProps {
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
  loading?: boolean;
  error?: string;
  elevation?: number;
}

export interface FormProps extends BaseComponentProps {
  onSubmit: (data: any) => void | Promise<void>;
  onCancel?: () => void;
  loading?: boolean;
  disabled?: boolean;
  initialValues?: Record<string, any>;
  validationSchema?: any;
}

export interface ModalProps extends BaseComponentProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  maxWidth?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  fullWidth?: boolean;
  disableBackdropClick?: boolean;
}

// ========================================
// DASHBOARD TYPES
// ========================================

export interface DashboardWidget {
  id: string;
  title: string;
  type: 'metric' | 'chart' | 'table' | 'custom';
  size: 'small' | 'medium' | 'large' | 'full';
  data?: any;
  config?: Record<string, any>;
  loading?: boolean;
  error?: string;
  refreshInterval?: number;
}

export interface DashboardLayout {
  widgets: DashboardWidget[];
  layout: Array<{
    i: string;
    x: number;
    y: number;
    w: number;
    h: number;
  }>;
}

export interface MetricCard {
  title: string;
  value: string | number;
  change?: {
    value: number;
    direction: 'up' | 'down';
    period: string;
  };
  icon?: React.ComponentType;
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info';
  loading?: boolean;
}

// ========================================
// SEARCH AND FILTER TYPES
// ========================================

export interface SearchState {
  query: string;
  filters: Record<string, any>;
  sort: {
    field: string;
    direction: 'asc' | 'desc';
  };
  pagination: {
    page: number;
    pageSize: number;
  };
}

export interface FilterOption {
  id: string;
  label: string;
  type: 'text' | 'select' | 'multiselect' | 'date' | 'daterange' | 'number' | 'boolean';
  options?: SelectOption[];
  placeholder?: string;
  defaultValue?: any;
}

// ========================================
// UPLOAD TYPES
// ========================================

export interface FileUploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  result?: any;
  correlationId?: string;
}

export interface UploadConfig {
  accept: string[];
  maxSize: number;
  maxFiles: number;
  multiple: boolean;
}

// ========================================
// SETTINGS TYPES
// ========================================

export interface AppSettings {
  theme: {
    mode: ThemeMode;
    primaryColor: string;
    fontSize: 'small' | 'medium' | 'large';
  };
  notifications: {
    enabled: boolean;
    position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
    duration: number;
  };
  predictions: {
    defaultModelVersion?: string;
    includeConfidenceByDefault: boolean;
    includeFeaturesByDefault: boolean;
    batchSizeLimit: number;
  };
  data: {
    autoRefreshInterval: number;
    defaultPageSize: number;
    exportFormat: 'csv' | 'json' | 'excel';
  };
  accessibility: {
    highContrast: boolean;
    reducedMotion: boolean;
    screenReaderSupport: boolean;
  };
}

// ========================================
// ANALYTICS TYPES
// ========================================

export interface AnalyticsTimeframe {
  value: '1h' | '24h' | '7d' | '30d' | '90d' | '1y' | 'all';
  label: string;
}

export interface AnalyticsMetric {
  name: string;
  current_value: number;
  previous_value?: number;
  change_percentage?: number;
  trend: 'up' | 'down' | 'stable';
  format: 'number' | 'percentage' | 'currency' | 'duration';
}

export interface AnalyticsDashboard {
  summary_metrics: AnalyticsMetric[];
  charts: ChartSeries[];
  tables: TableState[];
  last_updated: string;
  timeframe: AnalyticsTimeframe;
}

// ========================================
// UTILITY TYPES
// ========================================

export type Optional<T, K extends keyof T> = Pick<Partial<T>, K> & Omit<T, K>;

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

export interface AsyncState<T = any> {
  data?: T;
  loading: boolean;
  error?: string;
  lastFetch?: Date;
}

export interface CacheEntry<T = any> {
  data: T;
  timestamp: Date;
  ttl: number;
}

// ========================================
// EVENT TYPES
// ========================================

export interface AppEvent {
  type: string;
  payload?: any;
  timestamp: Date;
  source: string;
}

export interface PredictionEvent extends AppEvent {
  type: 'prediction_started' | 'prediction_completed' | 'prediction_failed';
  payload: {
    predictionId?: string;
    riskScore?: number;
    processingTime?: number;
    error?: string;
  };
}

// ========================================
// HOOK TYPES
// ========================================

export interface UseApiOptions {
  enabled?: boolean;
  retry?: number;
  staleTime?: number;
  cacheTime?: number;
  refetchOnWindowFocus?: boolean;
  onSuccess?: (data: any) => void;
  onError?: (error: any) => void;
}

export interface UsePaginationOptions {
  initialPage?: number;
  initialPageSize?: number;
  maxPageSize?: number;
}

export interface UsePaginationResult {
  page: number;
  pageSize: number;
  offset: number;
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
  nextPage: () => void;
  prevPage: () => void;
  hasNextPage: boolean;
  hasPrevPage: boolean;
}

// ========================================
// STORE TYPES
// ========================================

export interface AppState {
  // App configuration
  config: AppConfig;
  theme: ThemeMode;
  ready: boolean;
  
  // User state
  auth: AuthState;
  settings: AppSettings;
  
  // Feature states
  predictions: {
    history: PredictionHistory | null;
    statistics: PredictionStatistics | null;
    activeModel: ModelInfo | null;
  };
  
  // System state
  health: HealthStatus | null;
  notifications: Notification[];
  
  // UI state
  loading: Record<string, boolean>;
  errors: Record<string, string>;
  modals: Record<string, boolean>;
  
  // Cache
  cache: Record<string, CacheEntry>;
}

// ========================================
// API CLIENT TYPES
// ========================================

export interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  retryDelay: number;
  headers: Record<string, string>;
}

export interface RequestConfig {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  url: string;
  data?: any;
  params?: Record<string, any>;
  headers?: Record<string, string>;
  timeout?: number;
  onUploadProgress?: (progress: number) => void;
}

// ========================================
// EXPORT ALL TYPES
// ========================================

export type {
  // Re-export for convenience
  React,
};

// Default export for main types
export default {
  // Core types
  PatientData,
  PredictionResponse,
  ModelInfo,
  Dataset,
  
  // UI types
  AppTheme,
  LoadingState,
  ErrorState,
  
  // Analytics types
  PredictionStatistics,
  HealthStatus,
  
  // Utility types
  ApiResponse,
  AsyncState,
  FormState,
};