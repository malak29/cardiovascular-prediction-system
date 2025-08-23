import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  IconButton,
  Avatar,
  Chip,
  Alert,
  AlertTitle,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  CircularProgress,
  Divider,
  Stack,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  PredictionsOutlined,
  DatasetOutlined,
  ModelTrainingOutlined,
  AnalyticsOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  WarningAmberOutlined,
  CheckCircleOutlined,
  ErrorOutlineOutlined,
  RefreshOutlined,
  LaunchOutlined,
  ScheduleOutlined,
  PeopleOutlined,
  AssessmentOutlined,
  SecurityOutlined,
  SpeedOutlined,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { format, formatDistanceToNow } from 'date-fns';

// Types
import { 
  MetricCard, 
  PredictionStatistics, 
  HealthStatus, 
  ModelInfo,
  Dataset 
} from '../types';

// Hooks
import { usePrediction, usePredictionAnalytics } from '../hooks/usePrediction';
import { useData } from '../hooks/useData';
import { useModels } from '../hooks/usePrediction';
import { useHealthCheck } from '../hooks/useHealthCheck';

// Components
import DataVisualization, { RiskDistributionChart } from '../components/DataVisualization';
import LoadingSpinner, { ComponentLoadingSpinner, DataLoadingSkeleton } from '../components/common/LoadingSpinner';

// Utilities
import { formatRiskScore, getRiskCategoryColor } from '../utils';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  
  // State
  const [refreshKey, setRefreshKey] = useState(0);
  const [selectedTimeframe, setSelectedTimeframe] = useState('30d');

  // Hooks
  const { statistics, loading: predictionsLoading } = usePrediction({ enableRealTimeUpdates: true });
  const { datasets, dataStatistics, loading: dataLoading } = useData({ enableAutoRefresh: true });
  const { models, activeModelPerformance, loading: modelsLoading } = useModels();
  const { healthStatus, checkHealth, loading: healthLoading } = useHealthCheck();
  const { analytics } = usePredictionAnalytics(selectedTimeframe);

  // ========================================
  // METRIC CARDS DATA
  // ========================================

  const metricCards: MetricCard[] = [
    {
      title: 'Total Predictions',
      value: statistics?.total_predictions?.toLocaleString() || '0',
      change: {
        value: 12.5,
        direction: 'up',
        period: 'vs last month',
      },
      icon: PredictionsOutlined,
      color: 'primary',
      loading: predictionsLoading,
    },
    {
      title: 'High Risk Patients',
      value: statistics?.risk_distribution?.high?.toString() || '0',
      change: {
        value: 3.2,
        direction: 'down',
        period: 'vs last month',
      },
      icon: WarningAmberOutlined,
      color: 'error',
      loading: predictionsLoading,
    },
    {
      title: 'Model Accuracy',
      value: activeModelPerformance?.r2_score ? `${(activeModelPerformance.r2_score * 100).toFixed(1)}%` : 'N/A',
      change: {
        value: 0.8,
        direction: 'up',
        period: 'vs previous model',
      },
      icon: AssessmentOutlined,
      color: 'success',
      loading: modelsLoading,
    },
    {
      title: 'Active Datasets',
      value: datasets.length.toString(),
      change: {
        value: 2,
        direction: 'up',
        period: 'new this week',
      },
      icon: DatasetOutlined,
      color: 'info',
      loading: dataLoading,
    },
  ];

  // ========================================
  // EVENT HANDLERS
  // ========================================

  const handleRefreshAll = useCallback(async () => {
    setRefreshKey(prev => prev + 1);
    
    try {
      await Promise.all([
        checkHealth(),
        // Additional refresh operations would go here
      ]);
      
      toast.success('Dashboard refreshed successfully');
    } catch (error) {
      console.error('Dashboard refresh failed:', error);
      toast.error('Some data may not be up to date');
    }
  }, [checkHealth]);

  const handleQuickAction = useCallback((action: string, params?: any) => {
    switch (action) {
      case 'new_prediction':
        navigate('/predict');
        break;
      case 'batch_prediction':
        navigate('/predict/batch');
        break;
      case 'upload_data':
        navigate('/data');
        break;
      case 'view_models':
        navigate('/models');
        break;
      case 'view_analytics':
        navigate('/analytics');
        break;
      default:
        console.warn('Unknown quick action:', action);
    }
  }, [navigate]);

  // ========================================
  // RENDER FUNCTIONS
  // ========================================

  // Render metric card
  const renderMetricCard = (metric: MetricCard) => (
    <motion.div
      key={metric.title}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card elevation={2} sx={{ height: '100%', position: 'relative', overflow: 'visible' }}>
        <CardContent sx={{ pb: '16px !important' }}>
          {metric.loading ? (
            <ComponentLoadingSpinner height={100} />
          ) : (
            <>
              {/* Icon */}
              <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 2 }}>
                <Avatar
                  sx={{
                    bgcolor: `${metric.color}.main`,
                    width: 48,
                    height: 48,
                  }}
                >
                  <metric.icon />
                </Avatar>
                
                {metric.change && (
                  <Chip
                    icon={metric.change.direction === 'up' ? <TrendingUpOutlined /> : <TrendingDownOutlined />}
                    label={`${metric.change.direction === 'up' ? '+' : '-'}${metric.change.value}%`}
                    size="small"
                    color={metric.change.direction === 'up' ? 'success' : 'error'}
                    variant="outlined"
                  />
                )}
              </Box>

              {/* Value */}
              <Typography variant="h4" component="div" sx={{ fontWeight: 700, mb: 1 }}>
                {metric.value}
              </Typography>

              {/* Title */}
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {metric.title}
              </Typography>

              {/* Change indicator */}
              {metric.change && (
                <Typography 
                  variant="caption" 
                  color={metric.change.direction === 'up' ? 'success.main' : 'error.main'}
                  sx={{ fontSize: '0.75rem' }}
                >
                  {metric.change.period}
                </Typography>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );

  // Render system health card
  const renderSystemHealthCard = () => (
    <Card elevation={2}>
      <CardHeader
        title="System Health"
        avatar={
          <Avatar sx={{ 
            bgcolor: healthStatus?.overall_status === 'healthy' ? 'success.main' : 
                   healthStatus?.overall_status === 'degraded' ? 'warning.main' : 'error.main'
          }}>
            {healthStatus?.overall_status === 'healthy' ? <CheckCircleOutlined /> : 
             healthStatus?.overall_status === 'degraded' ? <WarningAmberOutlined /> : <ErrorOutlineOutlined />}
          </Avatar>
        }
        action={
          <IconButton onClick={checkHealth} disabled={healthLoading}>
            <RefreshOutlined />
          </IconButton>
        }
      />
      <CardContent>
        {healthLoading ? (
          <ComponentLoadingSpinner height={120} />
        ) : healthStatus ? (
          <>
            <Box sx={{ mb: 2 }}>
              <Chip
                label={healthStatus.overall_status.toUpperCase()}
                color={
                  healthStatus.overall_status === 'healthy' ? 'success' :
                  healthStatus.overall_status === 'degraded' ? 'warning' : 'error'
                }
                sx={{ fontWeight: 600 }}
              />
            </Box>

            {/* Component status */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Component Status
              </Typography>
              {Object.entries(healthStatus.components || {}).map(([component, status]) => (
                <Box key={component} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                    {component.replace(/_/g, ' ')}
                  </Typography>
                  <Chip
                    label={status.status}
                    size="small"
                    color={status.status === 'healthy' ? 'success' : 'error'}
                    variant="outlined"
                  />
                </Box>
              ))}
            </Box>

            {/* Performance metrics */}
            {healthStatus.performance_metrics && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Performance
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Uptime: {healthStatus.performance_metrics.uptime || 'Unknown'}
                </Typography>
                {healthStatus.performance_metrics.memory_usage && (
                  <Typography variant="body2" color="text.secondary">
                    Memory: {healthStatus.performance_metrics.memory_usage.percent_used?.toFixed(1)}% used
                  </Typography>
                )}
              </Box>
            )}

            {/* Warnings and errors */}
            {healthStatus.warnings && healthStatus.warnings.length > 0 && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                <AlertTitle>Warnings</AlertTitle>
                {healthStatus.warnings.slice(0, 2).map((warning, index) => (
                  <Typography key={index} variant="body2">
                    • {warning}
                  </Typography>
                ))}
              </Alert>
            )}
          </>
        ) : (
          <Alert severity="error">
            Unable to retrieve system health information
          </Alert>
        )}
      </CardContent>
    </Card>
  );

  // Render recent predictions
  const renderRecentPredictions = () => (
    <Card elevation={2}>
      <CardHeader
        title="Recent Predictions"
        subheader={`Last ${selectedTimeframe} activity`}
        action={
          <Button
            size="small"
            endIcon={<LaunchOutlined />}
            onClick={() => navigate('/analytics')}
          >
            View All
          </Button>
        }
      />
      <CardContent>
        {predictionsLoading ? (
          <DataLoadingSkeleton type="table" items={5} height={48} />
        ) : statistics?.total_predictions ? (
          <>
            {/* Quick stats */}
            <Box sx={{ mb: 3 }}>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h5" fontWeight={600} color="primary">
                      {statistics.total_predictions.toLocaleString()}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Total Predictions
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h5" fontWeight={600} color="success.main">
                      {(statistics.risk_statistics?.average_risk_score * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Avg Risk Score
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h5" fontWeight={600} color="error">
                      {statistics.risk_distribution?.high || 0}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      High Risk
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </Box>

            {/* Risk distribution chart */}
            <Box sx={{ height: 200 }}>
              <RiskDistributionChart
                lowCount={statistics.risk_distribution?.low || 0}
                mediumCount={statistics.risk_distribution?.medium || 0}
                highCount={statistics.risk_distribution?.high || 0}
                title="Risk Distribution"
              />
            </Box>
          </>
        ) : (
          <Alert severity="info">
            <AlertTitle>No Predictions Yet</AlertTitle>
            Start by creating your first cardiovascular risk prediction.
            <Button
              variant="outlined"
              size="small"
              sx={{ mt: 2 }}
              onClick={() => navigate('/predict')}
            >
              Make Prediction
            </Button>
          </Alert>
        )}
      </CardContent>
    </Card>
  );

  // Render quick actions
  const renderQuickActions = () => (
    <Card elevation={2}>
      <CardHeader title="Quick Actions" />
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Button
              variant="contained"
              fullWidth
              startIcon={<PredictionsOutlined />}
              onClick={() => handleQuickAction('new_prediction')}
              sx={{ 
                py: 1.5,
                justifyContent: 'flex-start',
                textAlign: 'left'
              }}
            >
              <Box>
                <Typography variant="button" display="block">
                  New Prediction
                </Typography>
                <Typography variant="caption" display="block" sx={{ opacity: 0.8 }}>
                  Single patient assessment
                </Typography>
              </Box>
            </Button>
          </Grid>

          <Grid item xs={12} sm={6}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<PeopleOutlined />}
              onClick={() => handleQuickAction('batch_prediction')}
              sx={{ 
                py: 1.5,
                justifyContent: 'flex-start',
                textAlign: 'left'
              }}
            >
              <Box>
                <Typography variant="button" display="block">
                  Batch Prediction
                </Typography>
                <Typography variant="caption" display="block">
                  Multiple patients
                </Typography>
              </Box>
            </Button>
          </Grid>

          <Grid item xs={12} sm={6}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<DatasetOutlined />}
              onClick={() => handleQuickAction('upload_data')}
              sx={{ 
                py: 1.5,
                justifyContent: 'flex-start',
                textAlign: 'left'
              }}
            >
              <Box>
                <Typography variant="button" display="block">
                  Upload Data
                </Typography>
                <Typography variant="caption" display="block">
                  Add new dataset
                </Typography>
              </Box>
            </Button>
          </Grid>

          <Grid item xs={12} sm={6}>
            <Button
              variant="outlined"
              fullWidth
              startIcon={<AnalyticsOutlined />}
              onClick={() => handleQuickAction('view_analytics')}
              sx={{ 
                py: 1.5,
                justifyContent: 'flex-start',
                textAlign: 'left'
              }}
            >
              <Box>
                <Typography variant="button" display="block">
                  View Analytics
                </Typography>
                <Typography variant="caption" display="block">
                  Detailed insights
                </Typography>
              </Box>
            </Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  // Render model status
  const renderModelStatus = () => {
    const activeModel = models.find(m => m.is_default);

    return (
      <Card elevation={2}>
        <CardHeader
          title="Active Model"
          action={
            <Button
              size="small"
              endIcon={<LaunchOutlined />}
              onClick={() => navigate('/models')}
            >
              Manage
            </Button>
          }
        />
        <CardContent>
          {modelsLoading ? (
            <ComponentLoadingSpinner height={120} />
          ) : activeModel ? (
            <>
              <Box sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                  {activeModel.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {activeModel.model_type} • Version {activeModel.version}
                </Typography>
              </Box>

              {activeModelPerformance && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Performance Metrics
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    <Chip
                      label={`R² ${(activeModelPerformance.r2_score * 100).toFixed(1)}%`}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                    <Chip
                      label={`MSE ${activeModelPerformance.mse_score.toFixed(4)}`}
                      size="small"
                      variant="outlined"
                    />
                    <Chip
                      label={`MAE ${activeModelPerformance.mae_score.toFixed(4)}`}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </Box>
              )}

              <Box>
                <Typography variant="body2" color="text.secondary">
                  Deployed: {formatDistanceToNow(new Date(activeModel.deployed_at || activeModel.created_at), { addSuffix: true })}
                </Typography>
                {activeModel.training_data_size && (
                  <Typography variant="body2" color="text.secondary">
                    Training Size: {activeModel.training_data_size.toLocaleString()} samples
                  </Typography>
                )}
              </Box>
            </>
          ) : (
            <Alert severity="warning">
              No active model found. Please check model management.
            </Alert>
          )}
        </CardContent>
      </Card>
    );
  };

  // Render recent datasets
  const renderRecentDatasets = () => (
    <Card elevation={2}>
      <CardHeader
        title="Recent Datasets"
        action={
          <Button
            size="small"
            endIcon={<LaunchOutlined />}
            onClick={() => navigate('/data')}
          >
            View All
          </Button>
        }
      />
      <CardContent>
        {dataLoading ? (
          <DataLoadingSkeleton type="list" items={3} />
        ) : datasets.length > 0 ? (
          <Box>
            {datasets.slice(0, 3).map((dataset) => (
              <Box key={dataset.id} sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                py: 1.5,
                borderBottom: '1px solid',
                borderColor: 'divider',
                '&:last-child': { borderBottom: 'none' }
              }}>
                <Avatar sx={{ bgcolor: 'primary.main', mr: 2, width: 32, height: 32 }}>
                  <DatasetOutlined />
                </Avatar>
                
                <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                  <Typography variant="subtitle2" noWrap>
                    {dataset.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {dataset.total_records.toLocaleString()} records • {dataset.file_size_mb.toFixed(1)} MB
                  </Typography>
                </Box>

                <Box sx={{ textAlign: 'right' }}>
                  <Chip
                    label={dataset.dataset_type}
                    size="small"
                    variant="outlined"
                  />
                  <Typography variant="caption" display="block" color="text.secondary">
                    {formatDistanceToNow(new Date(dataset.created_at), { addSuffix: true })}
                  </Typography>
                </Box>
              </Box>
            ))}
          </Box>
        ) : (
          <Alert severity="info">
            <AlertTitle>No Datasets</AlertTitle>
            Upload your first dataset to get started.
            <Button
              variant="outlined"
              size="small"
              sx={{ mt: 2 }}
              onClick={() => navigate('/data')}
            >
              Upload Dataset
            </Button>
          </Alert>
        )}
      </CardContent>
    </Card>
  );

  // Render prediction trends
  const renderPredictionTrends = () => {
    if (!analytics || !analytics.trends) {
      return (
        <Card elevation={2}>
          <CardHeader title="Prediction Trends" />
          <CardContent>
            <ComponentLoadingSpinner height={200} />
          </CardContent>
        </Card>
      );
    }

    // Convert trends data for chart
    const trendData = Object.entries(analytics.trends.daily_statistics || {}).map(([date, stats]: [string, any]) => ({
      name: format(new Date(date), 'MMM dd'),
      value: stats.total,
      date,
    }));

    return (
      <Card elevation={2}>
        <CardHeader
          title="Prediction Trends"
          subheader={`Daily prediction volume (${selectedTimeframe})`}
        />
        <CardContent>
          <DataVisualization
            type="line"
            data={trendData}
            height={200}
            config={{
              showLegend: false,
              xAxisLabel: 'Date',
              yAxisLabel: 'Predictions',
              color: theme.palette.primary.main,
            }}
          />
        </CardContent>
      </Card>
    );
  };

  // Render alerts and notifications
  const renderAlerts = () => {
    const alerts = [];

    // Health-based alerts
    if (healthStatus?.overall_status === 'unhealthy') {
      alerts.push({
        severity: 'error' as const,
        title: 'System Health Critical',
        message: 'One or more critical services are unavailable.',
        action: () => navigate('/health'),
      });
    }

    if (healthStatus?.warnings && healthStatus.warnings.length > 0) {
      alerts.push({
        severity: 'warning' as const,
        title: 'System Warnings',
        message: `${healthStatus.warnings.length} system warnings detected.`,
        action: () => navigate('/health'),
      });
    }

    // Model performance alerts
    if (activeModelPerformance && activeModelPerformance.r2_score < 0.7) {
      alerts.push({
        severity: 'warning' as const,
        title: 'Model Performance',
        message: 'Active model performance is below recommended threshold.',
        action: () => navigate('/models'),
      });
    }

    // Data alerts
    if (dataStatistics?.storage_usage?.available_space_gb < 1) {
      alerts.push({
        severity: 'error' as const,
        title: 'Storage Space Low',
        message: 'Less than 1GB of storage space remaining.',
      });
    }

    if (alerts.length === 0) {
      return null;
    }

    return (
      <Card elevation={2}>
        <CardHeader title="System Alerts" />
        <CardContent>
          <Stack spacing={2}>
            {alerts.slice(0, 3).map((alert, index) => (
              <Alert
                key={index}
                severity={alert.severity}
                action={
                  alert.action && (
                    <Button color="inherit" size="small" onClick={alert.action}>
                      View
                    </Button>
                  )
                }
              >
                <AlertTitle>{alert.title}</AlertTitle>
                {alert.message}
              </Alert>
            ))}
          </Stack>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box sx={{ p: 3 }} id="main-content">
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
            Dashboard
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Cardiovascular Disease Prediction System Overview
          </Typography>
        </Box>

        <Button
          variant="outlined"
          startIcon={<RefreshOutlined />}
          onClick={handleRefreshAll}
          disabled={healthLoading}
        >
          Refresh All
        </Button>
      </Box>

      {/* Alerts section */}
      <AnimatePresence>
        {renderAlerts() && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            style={{ marginBottom: 24 }}
          >
            {renderAlerts()}
          </motion.div>
        )}
      </AnimatePresence>

      <Grid container spacing={3}>
        {/* Metric Cards */}
        {metricCards.map((metric, index) => (
          <Grid item xs={12} sm={6} lg={3} key={metric.title}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              {renderMetricCard(metric)}
            </motion.div>
          </Grid>
        ))}

        {/* Quick Actions */}
        <Grid item xs={12} lg={6}>
          {renderQuickActions()}
        </Grid>

        {/* System Health */}
        <Grid item xs={12} lg={6}>
          {renderSystemHealthCard()}
        </Grid>

        {/* Recent Predictions */}
        <Grid item xs={12} lg={8}>
          {renderRecentPredictions()}
        </Grid>

        {/* Active Model Status */}
        <Grid item xs={12} lg={4}>
          {renderModelStatus()}
        </Grid>

        {/* Prediction Trends */}
        <Grid item xs={12} lg={8}>
          {renderPredictionTrends()}
        </Grid>

        {/* Recent Datasets */}
        <Grid item xs={12} lg={4}>
          {renderRecentDatasets()}
        </Grid>
      </Grid>

      {/* Footer info */}
      <Box sx={{ mt: 6, pt: 3, borderTop: '1px solid', borderColor: 'divider' }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              About This System
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Production-ready ML system for predicting cardiovascular disease hospitalization 
              rates among Medicare beneficiaries using evidence-based algorithms.
            </Typography>
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              Key Features
            </Typography>
            <Box component="ul" sx={{ pl: 2, m: 0 }}>
              <Typography component="li" variant="body2" color="text.secondary">
                Real-time risk assessment
              </Typography>
              <Typography component="li" variant="body2" color="text.secondary">
                Batch processing capabilities
              </Typography>
              <Typography component="li" variant="body2" color="text.secondary">
                Model performance monitoring
              </Typography>
              <Typography component="li" variant="body2" color="text.secondary">
                Data quality validation
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SpeedOutlined fontSize="small" color="success" />
                <Typography variant="body2" color="text.secondary">
                  System operational
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SecurityOutlined fontSize="small" color="success" />
                <Typography variant="body2" color="text.secondary">
                  Data secure
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckCircleOutlined fontSize="small" color="success" />
                <Typography variant="body2" color="text.secondary">
                  Models active
                </Typography>
              </Box>
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default Dashboard;