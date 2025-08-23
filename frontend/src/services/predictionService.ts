import { 
  PredictionRequest,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  PatientData,
  PredictionStatistics,
  ModelInfo
} from '../types';
import apiClient from './api';

/**
 * Prediction result cache entry
 */
interface PredictionCacheEntry {
  request: PredictionRequest;
  response: PredictionResponse;
  timestamp: number;
  hash: string;
}

/**
 * Prediction validation result
 */
interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  completenessScore: number;
  recommendations: string[];
}

/**
 * Prediction Service Class
 */
class PredictionService {
  private predictionCache = new Map<string, PredictionCacheEntry>();
  private readonly CACHE_TTL = 30 * 60 * 1000; // 30 minutes
  private readonly MAX_CACHE_SIZE = 100;

  // ========================================
  // CORE PREDICTION METHODS
  // ========================================

  /**
   * Make a single cardiovascular disease risk prediction
   */
  async makePrediction(request: PredictionRequest): Promise<PredictionResponse> {
    try {
      console.log('🔮 Making prediction request:', request);

      // Validate input data
      const validation = this.validatePatientData(request.patient_data);
      if (!validation.isValid) {
        throw new Error(`Validation failed: ${validation.errors.join(', ')}`);
      }

      // Check cache first
      const cacheKey = this.generatePredictionHash(request);
      const cached = this.getCachedPrediction(cacheKey);
      
      if (cached) {
        console.log('💾 Using cached prediction');
        return cached.response;
      }

      // Make API request
      const response = await apiClient.makePrediction(request);
      
      if (!response.data) {
        throw new Error('No prediction data received from server');
      }

      // Process and enhance the response
      const enhancedResponse = await this.enhancePredictionResponse(response.data, request);

      // Cache the result
      this.cachePrediction(cacheKey, request, enhancedResponse);

      console.log('✅ Prediction completed:', {
        riskScore: enhancedResponse.risk_score,
        riskCategory: enhancedResponse.risk_category,
        processingTime: enhancedResponse.prediction_time_ms,
      });

      return enhancedResponse;

    } catch (error: any) {
      console.error('❌ Prediction failed:', error);
      
      // Transform error for user-friendly display
      const userError = this.transformPredictionError(error);
      throw userError;
    }
  }

  /**
   * Make batch predictions
   */
  async makeBatchPrediction(request: BatchPredictionRequest): Promise<BatchPredictionResponse> {
    try {
      console.log('📊 Making batch prediction request:', {
        batchSize: request.patients_data.length,
        modelVersion: request.model_version,
      });

      // Validate batch size
      if (request.patients_data.length === 0) {
        throw new Error('Batch cannot be empty');
      }

      if (request.patients_data.length > 1000) {
        throw new Error('Batch size cannot exceed 1000 patients');
      }

      // Validate each patient data entry
      const validationResults = request.patients_data.map((patient, index) => ({
        index,
        validation: this.validatePatientData(patient),
      }));

      const invalidEntries = validationResults.filter(result => !result.validation.isValid);
      
      if (invalidEntries.length > 0) {
        const errorMessage = `Invalid data for patients at indices: ${invalidEntries.map(e => e.index).join(', ')}`;
        throw new Error(errorMessage);
      }

      // Make API request
      const response = await apiClient.makeBatchPrediction(request);
      
      if (!response.data) {
        throw new Error('No batch prediction data received from server');
      }

      // Process and enhance batch results
      const enhancedResponse = await this.enhanceBatchResponse(response.data);

      console.log('✅ Batch prediction completed:', {
        totalRequested: enhancedResponse.total_requested,
        successfulCount: enhancedResponse.successful_count,
        failedCount: enhancedResponse.failed_count,
        processingTime: enhancedResponse.processing_time_ms,
      });

      return enhancedResponse;

    } catch (error: any) {
      console.error('❌ Batch prediction failed:', error);
      throw this.transformPredictionError(error);
    }
  }

  /**
   * Upload CSV file for batch predictions
   */
  async uploadCsvForPrediction(
    file: File,
    options: {
      modelVersion?: string;
      includeConfidence?: boolean;
    } = {},
    onProgress?: (progress: number) => void
  ): Promise<Blob> {
    try {
      console.log('📁 Uploading CSV for prediction:', {
        fileName: file.name,
        fileSize: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
        options,
      });

      // Validate file
      if (!file.name.toLowerCase().endsWith('.csv')) {
        throw new Error('Only CSV files are supported for batch predictions');
      }

      if (file.size > 50 * 1024 * 1024) { // 50MB limit
        throw new Error('File size cannot exceed 50MB');
      }

      // Upload and process
      const response = await apiClient.uploadCsvForPrediction(file, options, onProgress);

      console.log('✅ CSV prediction completed');
      return response.data;

    } catch (error: any) {
      console.error('❌ CSV prediction failed:', error);
      throw this.transformPredictionError(error);
    }
  }

  // ========================================
  // PREDICTION HISTORY AND ANALYTICS
  // ========================================

  /**
   * Get prediction history with filtering
   */
  async getPredictionHistory(filters: {
    limit?: number;
    offset?: number;
    startDate?: Date;
    endDate?: Date;
    riskCategory?: string;
  } = {}): Promise<any> {
    try {
      const params: any = {
        limit: filters.limit || 50,
        offset: filters.offset || 0,
      };

      if (filters.startDate) {
        params.start_date = filters.startDate.toISOString();
      }
      
      if (filters.endDate) {
        params.end_date = filters.endDate.toISOString();
      }
      
      if (filters.riskCategory) {
        params.risk_category = filters.riskCategory;
      }

      const response = await apiClient.getPredictionHistory(params);
      return response.data;

    } catch (error: any) {
      console.error('❌ Failed to get prediction history:', error);
      throw error;
    }
  }

  /**
   * Get prediction statistics and analytics
   */
  async getPredictionStatistics(timeframe: string = '30d'): Promise<PredictionStatistics> {
    try {
      const response = await apiClient.getPredictionStatistics(timeframe);
      
      if (!response.data) {
        throw new Error('No statistics data received');
      }

      return response.data;

    } catch (error: any) {
      console.error('❌ Failed to get prediction statistics:', error);
      throw error;
    }
  }

  // ========================================
  // DATA VALIDATION
  // ========================================

  /**
   * Validate patient data before prediction
   */
  validatePatientData(patientData: PatientData): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    const recommendations: string[] = [];

    // Required field validation
    if (!patientData.age || patientData.age < 18 || patientData.age > 120) {
      errors.push('Age must be between 18 and 120 years');
    }

    if (!patientData.gender || !['male', 'female', 'other', 'unknown'].includes(patientData.gender)) {
      errors.push('Valid gender selection is required');
    }

    // Range validations
    if (patientData.systolic_bp && (patientData.systolic_bp < 50 || patientData.systolic_bp > 300)) {
      errors.push('Systolic blood pressure must be between 50-300 mmHg');
    }

    if (patientData.diastolic_bp && (patientData.diastolic_bp < 30 || patientData.diastolic_bp > 200)) {
      errors.push('Diastolic blood pressure must be between 30-200 mmHg');
    }

    if (patientData.bmi && (patientData.bmi < 10 || patientData.bmi > 100)) {
      errors.push('BMI must be between 10-100');
    }

    if (patientData.total_cholesterol && (patientData.total_cholesterol < 50 || patientData.total_cholesterol > 500)) {
      errors.push('Total cholesterol must be between 50-500 mg/dL');
    }

    // Logical validations
    if (patientData.systolic_bp && patientData.diastolic_bp && 
        patientData.systolic_bp <= patientData.diastolic_bp) {
      warnings.push('Systolic blood pressure should typically be higher than diastolic');
    }

    if (patientData.cardiovascular_hospitalizations_last_year > patientData.total_hospitalizations_last_year) {
      errors.push('Cardiovascular hospitalizations cannot exceed total hospitalizations');
    }

    // Data completeness recommendations
    const importantFields = [
      'systolic_bp', 'diastolic_bp', 'total_cholesterol', 'hdl_cholesterol', 'bmi'
    ];
    
    const missingImportantFields = importantFields.filter(field => 
      !patientData[field as keyof PatientData]
    );

    if (missingImportantFields.length > 0) {
      recommendations.push(
        `Consider providing ${missingImportantFields.join(', ')} for more accurate predictions`
      );
    }

    // Calculate completeness score
    const totalFields = Object.keys(patientData).length;
    const completedFields = Object.values(patientData).filter(value => 
      value !== null && value !== undefined && value !== ''
    ).length;
    
    const completenessScore = completedFields / totalFields;

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      completenessScore,
      recommendations,
    };
  }

  /**
   * Validate CSV data before upload
   */
  async validateCsvData(file: File): Promise<{
    isValid: boolean;
    errors: string[];
    warnings: string[];
    preview?: any[];
    estimatedRows?: number;
  }> {
    try {
      // File validation
      const errors: string[] = [];
      const warnings: string[] = [];

      // Check file type
      if (!file.name.toLowerCase().endsWith('.csv')) {
        errors.push('File must be in CSV format');
      }

      // Check file size
      if (file.size > 50 * 1024 * 1024) {
        errors.push('File size cannot exceed 50MB');
      }

      if (file.size === 0) {
        errors.push('File cannot be empty');
      }

      // Estimate number of rows (rough calculation)
      const estimatedRows = Math.floor(file.size / 100); // Rough estimate

      if (estimatedRows > 10000) {
        warnings.push(`Large file detected (~${estimatedRows.toLocaleString()} rows). Processing may take several minutes.`);
      }

      // Read file preview for structure validation
      let preview: any[] = [];
      
      try {
        const text = await this.readFilePreview(file, 1000); // First 1KB
        const lines = text.split('\n').filter(line => line.trim());
        
        if (lines.length < 2) {
          errors.push('File must contain header row and at least one data row');
        } else {
          const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
          
          // Check for required columns
          const requiredColumns = ['age', 'gender'];
          const missingColumns = requiredColumns.filter(col => 
            !headers.some(h => h.toLowerCase().includes(col.toLowerCase()))
          );
          
          if (missingColumns.length > 0) {
            errors.push(`Missing required columns: ${missingColumns.join(', ')}`);
          }

          // Create preview data
          preview = lines.slice(1, 4).map(line => {
            const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
            const row: any = {};
            headers.forEach((header, index) => {
              row[header] = values[index] || '';
            });
            return row;
          });
        }
      } catch (fileError) {
        errors.push('Could not read file content. Please ensure the file is a valid CSV.');
      }

      return {
        isValid: errors.length === 0,
        errors,
        warnings,
        preview: preview.length > 0 ? preview : undefined,
        estimatedRows,
      };

    } catch (error: any) {
      console.error('CSV validation failed:', error);
      return {
        isValid: false,
        errors: ['File validation failed: ' + error.message],
        warnings: [],
      };
    }
  }

  // ========================================
  // HELPER METHODS
  // ========================================

  /**
   * Generate hash for prediction request (for caching)
   */
  private generatePredictionHash(request: PredictionRequest): string {
    const hashInput = {
      patient_data: request.patient_data,
      model_version: request.model_version,
      include_confidence: request.include_confidence,
      include_features: request.include_features,
    };

    return btoa(JSON.stringify(hashInput)).replace(/[^a-zA-Z0-9]/g, '');
  }

  /**
   * Get cached prediction if available and valid
   */
  private getCachedPrediction(hash: string): PredictionCacheEntry | null {
    const cached = this.predictionCache.get(hash);
    
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached;
    }

    // Remove expired cache entry
    if (cached) {
      this.predictionCache.delete(hash);
    }

    return null;
  }

  /**
   * Cache prediction result
   */
  private cachePrediction(
    hash: string, 
    request: PredictionRequest, 
    response: PredictionResponse
  ): void {
    // Clean up old cache entries if at max size
    if (this.predictionCache.size >= this.MAX_CACHE_SIZE) {
      const oldestKey = Array.from(this.predictionCache.keys())[0];
      this.predictionCache.delete(oldestKey);
    }

    this.predictionCache.set(hash, {
      request,
      response,
      timestamp: Date.now(),
      hash,
    });
  }

  /**
   * Enhance prediction response with additional insights
   */
  private async enhancePredictionResponse(
    response: PredictionResponse,
    request: PredictionRequest
  ): Promise<PredictionResponse> {
    try {
      // Add clinical context
      const clinicalContext = this.generateClinicalContext(response, request.patient_data);
      
      // Add peer comparison
      const peerComparison = await this.generatePeerComparison(response, request.patient_data);

      // Enhanced response
      return {
        ...response,
        clinical_context: clinicalContext,
        peer_comparison: peerComparison,
        data_completeness_score: this.calculateDataCompleteness(request.patient_data),
      };

    } catch (error) {
      console.warn('Failed to enhance prediction response:', error);
      return response; // Return original response if enhancement fails
    }
  }

  /**
   * Enhance batch prediction response
   */
  private async enhanceBatchResponse(response: BatchPredictionResponse): Promise<BatchPredictionResponse> {
    try {
      // Add batch-level analytics
      const batchAnalytics = this.calculateBatchAnalytics(response);

      return {
        ...response,
        batch_analytics: batchAnalytics,
      };

    } catch (error) {
      console.warn('Failed to enhance batch response:', error);
      return response;
    }
  }

  /**
   * Generate clinical context for prediction
   */
  private generateClinicalContext(
    prediction: PredictionResponse,
    patientData: PatientData
  ): any {
    const context = {
      risk_interpretation: this.interpretRiskScore(prediction.risk_score),
      clinical_significance: this.assessClinicalSignificance(prediction, patientData),
      urgency_level: this.determineUrgencyLevel(prediction.risk_score, patientData),
      monitoring_frequency: this.recommendMonitoringFrequency(prediction.risk_category),
    };

    return context;
  }

  /**
   * Generate peer comparison data
   */
  private async generatePeerComparison(
    prediction: PredictionResponse,
    patientData: PatientData
  ): Promise<any> {
    try {
      // This would typically call an API endpoint for peer data
      // For now, we'll generate mock peer comparison data
      
      const ageGroup = this.getAgeGroup(patientData.age);
      const genderGroup = patientData.gender;

      // Mock peer data (in production, this would come from API)
      const mockPeerData = {
        age_group: ageGroup,
        gender: genderGroup,
        average_risk_score: this.getMockAverageRiskScore(ageGroup, genderGroup),
        percentile: this.calculatePercentile(prediction.risk_score, ageGroup, genderGroup),
        risk_factors_comparison: this.comparePeerRiskFactors(patientData),
      };

      return mockPeerData;

    } catch (error) {
      console.warn('Peer comparison generation failed:', error);
      return null;
    }
  }

  /**
   * Calculate data completeness score
   */
  private calculateDataCompleteness(patientData: PatientData): number {
    const allFields = [
      'age', 'gender', 'race_ethnicity', 'state_code', 'zip_code',
      'has_hypertension', 'has_diabetes', 'has_heart_disease',
      'systolic_bp', 'diastolic_bp', 'total_cholesterol', 'hdl_cholesterol',
      'bmi', 'smoking_status', 'blood_glucose', 'hba1c'
    ];

    const completedFields = allFields.filter(field => {
      const value = patientData[field as keyof PatientData];
      return value !== null && value !== undefined && value !== '';
    });

    return completedFields.length / allFields.length;
  }

  /**
   * Calculate batch analytics
   */
  private calculateBatchAnalytics(response: BatchPredictionResponse): any {
    const successful = response.predictions.filter(p => p.prediction !== null);
    const riskScores = successful.map(p => p.prediction!.risk_score);

    if (riskScores.length === 0) {
      return { message: 'No successful predictions to analyze' };
    }

    return {
      risk_distribution_detailed: {
        very_low: riskScores.filter(score => score < 0.2).length,
        low: riskScores.filter(score => score >= 0.2 && score < 0.4).length,
        moderate: riskScores.filter(score => score >= 0.4 && score < 0.6).length,
        high: riskScores.filter(score => score >= 0.6 && score < 0.8).length,
        very_high: riskScores.filter(score => score >= 0.8).length,
      },
      statistics: {
        mean: this.calculateMean(riskScores),
        median: this.calculateMedian(riskScores),
        standard_deviation: this.calculateStandardDeviation(riskScores),
        min: Math.min(...riskScores),
        max: Math.max(...riskScores),
      },
      clinical_insights: this.generateBatchClinicalInsights(riskScores),
    };
  }

  /**
   * Transform API errors for user-friendly display
   */
  private transformPredictionError(error: any): Error {
    if (error.name === 'PermissionError') {
      return new Error('You do not have permission to make predictions. Please contact your administrator.');
    }

    if (error.message?.includes('validation')) {
      return new Error('Patient data validation failed. Please check your input and try again.');
    }

    if (error.message?.includes('timeout')) {
      return new Error('Prediction request timed out. The server may be busy. Please try again.');
    }

    if (error.message?.includes('network') || error.code === 'NETWORK_ERROR') {
      return new Error('Network connection failed. Please check your internet connection and try again.');
    }

    if (error.response?.status === 429) {
      return new Error('Too many requests. Please wait a moment before trying again.');
    }

    if (error.response?.status >= 500) {
      return new Error('Server error occurred. Please try again later or contact support.');
    }

    // Return original error message or generic fallback
    return new Error(error.message || 'An unexpected error occurred while making the prediction.');
  }

  /**
   * Read file preview for validation
   */
  private readFilePreview(file: File, maxBytes: number = 1000): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        resolve(event.target?.result as string);
      };
      
      reader.onerror = () => {
        reject(new Error('Failed to read file'));
      };

      // Read only the first part of the file
      const blob = file.slice(0, maxBytes);
      reader.readAsText(blob);
    });
  }

  // ========================================
  // CLINICAL INTERPRETATION HELPERS
  // ========================================

  private interpretRiskScore(riskScore: number): string {
    if (riskScore < 0.2) {
      return 'Very low risk - routine preventive care recommended';
    } else if (riskScore < 0.4) {
      return 'Low risk - continue current preventive measures';
    } else if (riskScore < 0.6) {
      return 'Moderate risk - enhanced monitoring and lifestyle modifications recommended';
    } else if (riskScore < 0.8) {
      return 'High risk - active medical management and close monitoring required';
    } else {
      return 'Very high risk - immediate medical attention and intensive management required';
    }
  }

  private assessClinicalSignificance(prediction: PredictionResponse, patientData: PatientData): string {
    const age = patientData.age;
    const riskScore = prediction.risk_score;

    if (age >= 80 && riskScore > 0.5) {
      return 'High clinical significance due to advanced age and elevated risk';
    } else if (age < 65 && riskScore > 0.7) {
      return 'High clinical significance due to unusually high risk for age group';
    } else if (riskScore > 0.8) {
      return 'Very high clinical significance - immediate intervention warranted';
    } else {
      return 'Standard clinical significance for age and risk profile';
    }
  }

  private determineUrgencyLevel(riskScore: number, patientData: PatientData): 'routine' | 'priority' | 'urgent' | 'emergent' {
    if (riskScore >= 0.8) return 'emergent';
    if (riskScore >= 0.6) return 'urgent';
    if (riskScore >= 0.4) return 'priority';
    return 'routine';
  }

  private recommendMonitoringFrequency(riskCategory: string): string {
    switch (riskCategory) {
      case 'low': return 'Annual assessment recommended';
      case 'medium': return 'Semi-annual assessment recommended';
      case 'high': return 'Quarterly assessment recommended';
      default: return 'Follow clinical guidelines for monitoring frequency';
    }
  }

  // ========================================
  // STATISTICAL HELPERS
  // ========================================

  private calculateMean(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculateMedian(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[mid - 1] + sorted[mid]) / 2 
      : sorted[mid];
  }

  private calculateStandardDeviation(values: number[]): number {
    const mean = this.calculateMean(values);
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const avgSquaredDiff = this.calculateMean(squaredDiffs);
    return Math.sqrt(avgSquaredDiff);
  }

  private getAgeGroup(age: number): string {
    if (age < 65) return 'under_65';
    if (age < 75) return '65_74';
    if (age < 85) return '75_84';
    return '85_plus';
  }

  private getMockAverageRiskScore(ageGroup: string, gender: string): number {
    // Mock data - in production, this would come from real statistics
    const baselines: Record<string, Record<string, number>> = {
      under_65: { male: 0.15, female: 0.12, other: 0.13, unknown: 0.13 },
      '65_74': { male: 0.35, female: 0.28, other: 0.31, unknown: 0.31 },
      '75_84': { male: 0.55, female: 0.48, other: 0.51, unknown: 0.51 },
      '85_plus': { male: 0.75, female: 0.68, other: 0.71, unknown: 0.71 },
    };

    return baselines[ageGroup]?.[gender] || 0.4;
  }

  private calculatePercentile(riskScore: number, ageGroup: string, gender: string): number {
    // Mock percentile calculation
    const averageScore = this.getMockAverageRiskScore(ageGroup, gender);
    
    if (riskScore <= averageScore * 0.5) return 10;
    if (riskScore <= averageScore * 0.8) return 25;
    if (riskScore <= averageScore) return 50;
    if (riskScore <= averageScore * 1.2) return 75;
    if (riskScore <= averageScore * 1.5) return 90;
    return 95;
  }

  private comparePeerRiskFactors(patientData: PatientData): any {
    const patientFactors = [
      patientData.has_hypertension,
      patientData.has_diabetes,
      patientData.has_heart_disease,
      patientData.smoking_status === 'current',
    ].filter(Boolean).length;

    // Mock peer comparison
    const avgFactorsForAge = patientData.age >= 75 ? 2.1 : patientData.age >= 65 ? 1.5 : 0.8;

    return {
      patient_risk_factors: patientFactors,
      peer_average: avgFactorsForAge,
      comparison: patientFactors > avgFactorsForAge ? 'above_average' : 
                 patientFactors < avgFactorsForAge ? 'below_average' : 'average',
    };
  }

  private generateBatchClinicalInsights(riskScores: number[]): string[] {
    const insights: string[] = [];
    
    const highRiskCount = riskScores.filter(score => score >= 0.7).length;
    const highRiskPercentage = (highRiskCount / riskScores.length) * 100;

    if (highRiskPercentage > 30) {
      insights.push(`High percentage (${highRiskPercentage.toFixed(1)}%) of patients are at high cardiovascular risk`);
    }

    const averageRisk = this.calculateMean(riskScores);
    if (averageRisk > 0.6) {
      insights.push('Overall patient population shows elevated cardiovascular risk');
    }

    const riskVariability = this.calculateStandardDeviation(riskScores);
    if (riskVariability > 0.3) {
      insights.push('High variability in risk scores suggests diverse patient population');
    }

    return insights;
  }

  // ========================================
  // CACHE MANAGEMENT
  // ========================================

  /**
   * Clear prediction cache
   */
  clearCache(): void {
    this.predictionCache.clear();
    console.log('🧹 Prediction cache cleared');
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    size: number;
    hitRate: number;
    oldestEntry?: string;
  } {
    const entries = Array.from(this.predictionCache.values());
    
    return {
      size: this.predictionCache.size,
      hitRate: 0, // Would track this in production
      oldestEntry: entries.length > 0 
        ? new Date(Math.min(...entries.map(e => e.timestamp))).toISOString()
        : undefined,
    };
  }

  /**
   * Clean expired cache entries
   */
  cleanExpiredCache(): void {
    const now = Date.now();
    
    for (const [key, entry] of this.predictionCache.entries()) {
      if (now - entry.timestamp >= this.CACHE_TTL) {
        this.predictionCache.delete(key);
      }
    }
  }

  // ========================================
  // UTILITY METHODS
  // ========================================

  /**
   * Check if service is available
   */
  async isServiceAvailable(): Promise<boolean> {
    try {
      await apiClient.healthCheck();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get service status
   */
  async getServiceStatus(): Promise<{
    available: boolean;
    responseTime?: number;
    version?: string;
    lastCheck: string;
  }> {
    const startTime = Date.now();
    
    try {
      const healthResponse = await apiClient.healthCheck();
      const responseTime = Date.now() - startTime;

      return {
        available: true,
        responseTime,
        version: healthResponse.data?.version,
        lastCheck: new Date().toISOString(),
      };

    } catch (error) {
      return {
        available: false,
        lastCheck: new Date().toISOString(),
      };
    }
  }

  /**
   * Preload models for faster predictions
   */
  async preloadModels(): Promise<void> {
    try {
      console.log('🚀 Preloading models...');
      await apiClient.getAvailableModels();
      console.log('✅ Models preloaded');
    } catch (error) {
      console.warn('⚠️ Model preloading failed:', error);
    }
  }
}

// Create singleton instance
const predictionService = new PredictionService();

// Export prediction service and utilities
export { predictionService, PredictionService };
export default predictionService;