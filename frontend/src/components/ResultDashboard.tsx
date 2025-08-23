import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Chip,
  Button,
  IconButton,
  Alert,
  AlertTitle,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Paper,
  Stack,
  CircularProgress,
  LinearProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  FavoriteOutlined,
  WarningAmberOutlined,
  CheckCircleOutlined,
  ErrorOutlineOutlined,
  ExpandMoreOutlined,
  ShareOutlined,
  PrintOutlined,
  DownloadOutlined,
  RefreshOutlined,
  InfoOutlined,
  LocalHospitalOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  PersonOutlined,
  ScheduleOutlined,
  AssessmentOutlined,
} from '@mui/icons-material';
import { format } from 'date-fns';

// Types and utilities
import { PredictionResponse, FeatureImportance, RiskCategoryType } from '../types';
import { formatRiskScore, getRiskCategoryColor, exportToPDF, copyToClipboard } from '../utils';

// Components
import DataVisualization from './DataVisualization';

interface ResultsDashboardProps {
  prediction: PredictionResponse;
  onNewPrediction?: () => void;
  onFeedback?: (feedback: any) => void;
  showActions?: boolean;
  compact?: boolean;
  className?: string;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({
  prediction,
  onNewPrediction,
  onFeedback,
  showActions = true,
  compact = false,
  className,
}) => {
  // State
  const [expandedSections, setExpandedSections] = useState<string[]>(['overview', 'recommendations']);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);

  // Risk category styling
  const riskCategoryConfig = useMemo(() => {
    const config = {
      low: {
        color: 'success' as const,
        icon: CheckCircleOutlined,
        description: 'Low risk of cardiovascular hospitalization',
        actionLevel: 'Routine monitoring',
        bgColor: '#e8f5e8',
        textColor: '#2e7d32',
      },
      medium: {
        color: 'warning' as const,
        icon: WarningAmberOutlined,
        description: 'Moderate risk of cardiovascular hospitalization',
        actionLevel: 'Enhanced monitoring recommended',
        bgColor: '#fff3e0',
        textColor: '#f57c00',
      },
      high: {
        color: 'error' as const,
        icon: ErrorOutlineOutlined,
        description: 'High risk of cardiovascular hospitalization',
        actionLevel: 'Immediate clinical attention recommended',
        bgColor: '#ffebee',
        textColor: '#d32f2f',
      },
    };

    return config[prediction.risk_category] || config.medium;
  }, [prediction.risk_category]);

  // Handle section expansion
  const handleSectionToggle = useCallback((section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  }, []);

  // Export functionality
  const handleExport = useCallback(async (format: 'pdf' | 'json' | 'print') => {
    try {
      switch (format) {
        case 'pdf':
          await exportToPDF(prediction, 'cardiovascular_prediction_report.pdf');
          break;
        case 'json':
          const dataStr = JSON.stringify(prediction, null, 2);
          const dataBlob = new Blob([dataStr], { type: 'application/json' });
          const url = URL.createObjectURL(dataBlob);
          const link = document.createElement('a');
          link.href = url;
          link.download = `prediction_${prediction.prediction_id}.json`;
          link.click();
          URL.revokeObjectURL(url);
          break;
        case 'print':
          window.print();
          break;
      }
      setShowExportDialog(false);
    } catch (error) {
      console.error('Export failed:', error);
    }
  }, [prediction]);

  // Share functionality
  const handleShare = useCallback(async () => {
    const shareData = {
      title: 'Cardiovascular Risk Assessment Results',
      text: `Risk Score: ${formatRiskScore(prediction.risk_score)} (${prediction.risk_category.toUpperCase()})`,
      url: window.location.href,
    };

    if (navigator.share) {
      try {
        await navigator.share(shareData);
      } catch (error) {
        // Fallback to clipboard
        await copyToClipboard(shareData.text);
      }
    } else {
      await copyToClipboard(shareData.text);
    }
    
    setShowShareDialog(false);
  }, [prediction]);

  // Render main risk score card
  const renderRiskScoreCard = () => {
    const RiskIcon = riskCategoryConfig.icon;
    
    return (
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      >
        <Card 
          elevation={6} 
          sx={{ 
            background: `linear-gradient(135deg, ${riskCategoryConfig.bgColor} 0%, ${riskCategoryConfig.bgColor}ee 100%)`,
            border: `2px solid ${riskCategoryConfig.textColor}`,
            position: 'relative',
            overflow: 'visible'
          }}
        >
          <CardContent sx={{ textAlign: 'center', py: 4 }}>
            {/* Risk icon */}
            <Box sx={{ mb: 2 }}>
              <RiskIcon 
                sx={{ 
                  fontSize: 64, 
                  color: riskCategoryConfig.textColor,
                  filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))'
                }} 
              />
            </Box>

            {/* Risk score */}
            <Typography variant="h2" component="div" sx={{ 
              fontWeight: 700, 
              color: riskCategoryConfig.textColor,
              mb: 1
            }}>
              {formatRiskScore(prediction.risk_score)}
            </Typography>

            {/* Risk category */}
            <Chip
              label={prediction.risk_category.toUpperCase()}
              color={riskCategoryConfig.color}
              size="large"
              sx={{ 
                fontSize: '1.1rem', 
                fontWeight: 600,
                px: 3,
                py: 1,
                mb: 2
              }}
            />

            {/* Description */}
            <Typography variant="h6" sx={{ color: riskCategoryConfig.textColor, mb: 2 }}>
              {riskCategoryConfig.description}
            </Typography>

            {/* Action level */}
            <Typography variant="body1" sx={{ fontWeight: 500, color: 'text.secondary' }}>
              {riskCategoryConfig.actionLevel}
            </Typography>

            {/* Confidence interval */}
            {prediction.confidence_lower !== undefined && prediction.confidence_upper !== undefined && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  95% Confidence Interval
                </Typography>
                <Typography variant="h6" sx={{ color: riskCategoryConfig.textColor }}>
                  {formatRiskScore(prediction.confidence_lower)} - {formatRiskScore(prediction.confidence_upper)}
                </Typography>
              </Box>
            )}
          </CardContent>

          {/* Floating info button */}
          <Tooltip title="About this prediction">
            <IconButton
              sx={{
                position: 'absolute',
                top: 16,
                right: 16,
                bgcolor: 'white',
                boxShadow: 2,
                '&:hover': { bgcolor: 'grey.50' }
              }}
              size="small"
            >
              <InfoOutlined />
            </IconButton>
          </Tooltip>
        </Card>
      </motion.div>
    );
  };

  // Render clinical recommendations
  const renderRecommendations = () => (
    <Card elevation={2}>
      <CardHeader
        title="Clinical Recommendations"
        avatar={<LocalHospitalOutlined color="primary" />}
        action={
          <Chip 
            label={`${prediction.recommendations?.length || 0} recommendations`}
            size="small"
            color="primary"
            variant="outlined"
          />
        }
      />
      <CardContent>
        {prediction.recommendations && prediction.recommendations.length > 0 ? (
          <List>
            {prediction.recommendations.map((recommendation, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleOutlined color="primary" />
                  </ListItemIcon>
                  <ListItemText
                    primary={recommendation}
                    primaryTypographyProps={{ variant: 'body1', fontWeight: 500 }}
                  />
                </ListItem>
                {index < prediction.recommendations.length - 1 && <Divider />}
              </motion.div>
            ))}
          </List>
        ) : (
          <Alert severity="info">
            No specific recommendations available. Consult with healthcare provider for personalized guidance.
          </Alert>
        )}
      </CardContent>
    </Card>
  );

  // Render feature importance
  const renderFeatureImportance = () => {
    if (!prediction.feature_importance || prediction.feature_importance.length === 0) {
      return null;
    }

    return (
      <Card elevation={2}>
        <CardHeader
          title="Key Contributing Factors"
          avatar={<AssessmentOutlined color="primary" />}
          subheader="Factors that most influenced this prediction"
        />
        <CardContent>
          <Box>
            {prediction.feature_importance.slice(0, 8).map((feature, index) => (
              <Box key={feature.feature_name} sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" fontWeight={500}>
                    {feature.description || feature.feature_name.replace(/_/g, ' ')}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {Math.round(feature.importance_score * 100)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={feature.importance_score * 100}
                  sx={{ 
                    height: 8, 
                    borderRadius: 4,
                    bgcolor: 'grey.200',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 4,
                      background: `linear-gradient(90deg, ${riskCategoryConfig.textColor}66, ${riskCategoryConfig.textColor})`
                    }
                  }}
                />
              </Box>
            ))}
          </Box>

          {prediction.feature_importance.length > 8 && (
            <Button
              size="small"
              onClick={() => handleSectionToggle('all_features')}
              sx={{ mt: 2 }}
            >
              {expandedSections.includes('all_features') ? 'Show Less' : `Show All ${prediction.feature_importance.length} Features`}
            </Button>
          )}

          <Collapse in={expandedSections.includes('all_features')}>
            <Box sx={{ mt: 2 }}>
              <DataVisualization
                type="horizontal_bar"
                data={prediction.feature_importance.map(f => ({
                  name: f.description || f.feature_name,
                  value: f.importance_score * 100,
                  category: f.category || 'other'
                }))}
                title="Complete Feature Importance"
                height={400}
              />
            </Box>
          </Collapse>
        </CardContent>
      </Card>
    );
  };

  // Render prediction metadata
  const renderPredictionMetadata = () => (
    <Card elevation={1} sx={{ bgcolor: 'background.default' }}>
      <CardHeader
        title="Prediction Details"
        avatar={<InfoOutlined color="primary" />}
        sx={{ pb: 1 }}
      />
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Model Information
              </Typography>
              <Typography variant="body1" fontWeight={500}>
                {prediction.model_version} ({prediction.model_type})
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Processing Time
              </Typography>
              <Typography variant="body1" fontWeight={500}>
                {Math.round(prediction.prediction_time_ms)}ms
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Prediction ID
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                {prediction.prediction_id}
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12} sm={6}>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Generated At
              </Typography>
              <Typography variant="body1">
                {format(new Date(prediction.timestamp), 'MMM dd, yyyy HH:mm:ss')}
              </Typography>
            </Box>
          </Grid>

          {prediction.confidence_score && (
            <Grid item xs={12}>
              <Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Prediction Confidence
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <CircularProgress
                    variant="determinate"
                    value={prediction.confidence_score * 100}
                    size={40}
                    thickness={6}
                    sx={{
                      color: prediction.confidence_score > 0.8 ? 'success.main' : 
                             prediction.confidence_score > 0.6 ? 'warning.main' : 'error.main'
                    }}
                  />
                  <Typography variant="h6" fontWeight={600}>
                    {Math.round(prediction.confidence_score * 100)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Model confidence in this prediction
                  </Typography>
                </Box>
              </Box>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );

  // Render risk factors identified
  const renderRiskFactors = () => {
    if (!prediction.risk_factors || prediction.risk_factors.length === 0) {
      return null;
    }

    return (
      <Card elevation={2}>
        <CardHeader
          title="Identified Risk Factors"
          avatar={<WarningAmberOutlined color="warning" />}
          subheader={`${prediction.risk_factors.length} risk factors contributing to this assessment`}
        />
        <CardContent>
          <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
            {prediction.risk_factors.map((factor, index) => (
              <motion.div
                key={factor}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                <Chip
                  label={factor}
                  color="warning"
                  variant="outlined"
                  size="medium"
                  sx={{ fontWeight: 500 }}
                />
              </motion.div>
            ))}
          </Stack>
        </CardContent>
      </Card>
    );
  };

  // Render confidence interval visualization
  const renderConfidenceInterval = () => {
    if (!prediction.confidence_lower || !prediction.confidence_upper) {
      return null;
    }

    const confidenceData = [
      {
        name: 'Risk Range',
        lower: prediction.confidence_lower,
        upper: prediction.confidence_upper,
        actual: prediction.risk_score,
      }
    ];

    return (
      <Card elevation={2}>
        <CardHeader
          title="Confidence Interval"
          avatar={<AssessmentOutlined color="primary" />}
          subheader={`${Math.round((prediction.confidence_level || 0.95) * 100)}% confidence interval for risk prediction`}
        />
        <CardContent>
          <DataVisualization
            type="confidence_interval"
            data={confidenceData}
            height={200}
            config={{
              showTooltip: true,
              responsive: true,
            }}
          />
          
          <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Interpretation
            </Typography>
            <Typography variant="body1">
              We are {Math.round((prediction.confidence_level || 0.95) * 100)}% confident that the true risk score 
              falls between <strong>{formatRiskScore(prediction.confidence_lower)}</strong> and{' '}
              <strong>{formatRiskScore(prediction.confidence_upper)}</strong>.
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  };

  // Render next steps and timeline
  const renderNextSteps = () => (
    <Card elevation={2}>
      <CardHeader
        title="Recommended Next Steps"
        avatar={<ScheduleOutlined color="primary" />}
        subheader="Clinical action timeline based on risk assessment"
      />
      <CardContent>
        <List>
          {prediction.risk_category === 'high' && (
            <>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="h6" color="error.main" sx={{ 
                    minWidth: 40, 
                    textAlign: 'center',
                    fontWeight: 700
                  }}>
                    Now
                  </Typography>
                </ListItemIcon>
                <ListItemText
                  primary="Immediate cardiovascular evaluation"
                  secondary="Schedule urgent appointment with cardiologist or primary care provider"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="body1" color="warning.main" sx={{ 
                    minWidth: 40, 
                    textAlign: 'center',
                    fontWeight: 600
                  }}>
                    1-2 wks
                  </Typography>
                </ListItemIcon>
                <ListItemText
                  primary="Comprehensive cardiac workup"
                  secondary="ECG, echocardiogram, stress testing as appropriate"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="body1" color="info.main" sx={{ 
                    minWidth: 40, 
                    textAlign: 'center',
                    fontWeight: 600
                  }}>
                    1 mo
                  </Typography>
                </ListItemIcon>
                <ListItemText
                  primary="Follow-up assessment"
                  secondary="Review intervention effectiveness and adjust treatment plan"
                />
              </ListItem>
            </>
          )}

          {prediction.risk_category === 'medium' && (
            <>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="h6" color="warning.main" sx={{ 
                    minWidth: 40, 
                    textAlign: 'center',
                    fontWeight: 700
                  }}>
                    2-4 wks
                  </Typography>
                </ListItemIcon>
                <ListItemText
                  primary="Schedule preventive care visit"
                  secondary="Discuss risk factors and prevention strategies"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="body1" color="info.main" sx={{ 
                    minWidth: 40, 
                    textAlign: 'center',
                    fontWeight: 600
                  }}>
                    3 mo
                  </Typography>
                </ListItemIcon>
                <ListItemText
                  primary="Risk factor reassessment"
                  secondary="Monitor progress and adjust interventions"
                />
              </ListItem>
            </>
          )}

          {prediction.risk_category === 'low' && (
            <>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="h6" color="success.main" sx={{ 
                    minWidth: 40, 
                    textAlign: 'center',
                    fontWeight: 700
                  }}>
                    6-12 mo
                  </Typography>
                </ListItemIcon>
                <ListItemText
                  primary="Routine follow-up"
                  secondary="Continue current preventive care and monitoring"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="body1" color="primary.main" sx={{ 
                    minWidth: 40, 
                    textAlign: 'center',
                    fontWeight: 600
                  }}>
                    Annual
                  </Typography>
                </ListItemIcon>
                <ListItemText
                  primary="Annual risk reassessment"
                  secondary="Routine cardiovascular risk evaluation"
                />
              </ListItem>
            </>
          )}
        </List>

        <Alert severity="info" sx={{ mt: 2 }}>
          <AlertTitle>Important Note</AlertTitle>
          These recommendations are based on the prediction model and should be integrated with clinical judgment. 
          Always consult with healthcare providers for personalized medical decisions.
        </Alert>
      </CardContent>
    </Card>
  );

  // Action buttons
  const renderActionButtons = () => {
    if (!showActions) return null;

    return (
      <Paper elevation={1} sx={{ p: 2, bgcolor: 'background.default' }}>
        <Stack direction="row" spacing={2} justifyContent="center" flexWrap="wrap">
          <Button
            variant="contained"
            startIcon={<RefreshOutlined />}
            onClick={onNewPrediction}
            size="large"
          >
            New Prediction
          </Button>

          <Button
            variant="outlined"
            startIcon={<ShareOutlined />}
            onClick={() => setShowShareDialog(true)}
          >
            Share Results
          </Button>

          <Button
            variant="outlined"
            startIcon={<DownloadOutlined />}
            onClick={() => setShowExportDialog(true)}
          >
            Export Report
          </Button>

          <Button
            variant="outlined"
            startIcon={<PrintOutlined />}
            onClick={() => handleExport('print')}
          >
            Print
          </Button>
        </Stack>
      </Paper>
    );
  };

  return (
    <Box className={className} sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
          Cardiovascular Risk Assessment Results
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          AI-powered prediction based on clinical data and evidence-based algorithms
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Main risk score - full width on mobile, half on desktop */}
        <Grid item xs={12} lg={6}>
          {renderRiskScoreCard()}
        </Grid>

        {/* Metadata card */}
        <Grid item xs={12} lg={6}>
          {renderPredictionMetadata()}
        </Grid>

        {/* Risk factors */}
        {prediction.risk_factors && prediction.risk_factors.length > 0 && (
          <Grid item xs={12}>
            {renderRiskFactors()}
          </Grid>
        )}

        {/* Feature importance */}
        {prediction.feature_importance && (
          <Grid item xs={12} lg={8}>
            {renderFeatureImportance()}
          </Grid>
        )}

        {/* Confidence interval */}
        {prediction.confidence_lower !== undefined && (
          <Grid item xs={12} lg={4}>
            {renderConfidenceInterval()}
          </Grid>
        )}

        {/* Recommendations */}
        <Grid item xs={12} lg={8}>
          {renderRecommendations()}
        </Grid>

        {/* Next steps */}
        <Grid item xs={12} lg={4}>
          {renderNextSteps()}
        </Grid>

        {/* Action buttons */}
        {showActions && (
          <Grid item xs={12}>
            {renderActionButtons()}
          </Grid>
        )}
      </Grid>

      {/* Export Dialog */}
      <Dialog open={showExportDialog} onClose={() => setShowExportDialog(false)}>
        <DialogTitle>Export Prediction Report</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Choose the format for exporting this prediction report:
          </Typography>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <Button
              variant="outlined"
              startIcon={<DownloadOutlined />}
              onClick={() => handleExport('pdf')}
              fullWidth
            >
              Export as PDF Report
            </Button>
            <Button
              variant="outlined"
              startIcon={<DownloadOutlined />}
              onClick={() => handleExport('json')}
              fullWidth
            >
              Export as JSON Data
            </Button>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowExportDialog(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>

      {/* Share Dialog */}
      <Dialog open={showShareDialog} onClose={() => setShowShareDialog(false)}>
        <DialogTitle>Share Prediction Results</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Share this prediction result with colleagues or copy to clipboard:
          </Typography>
          <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              Risk Score: {formatRiskScore(prediction.risk_score)} ({prediction.risk_category.toUpperCase()})
              <br />
              Model: {prediction.model_version}
              <br />
              Generated: {format(new Date(prediction.timestamp), 'MMM dd, yyyy HH:mm')}
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowShareDialog(false)}>Cancel</Button>
          <Button onClick={handleShare} variant="contained">
            Share
          </Button>
        </DialogActions>
      </Dialog>

      {/* Accessibility announcements */}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {prediction && (
          `Cardiovascular risk assessment completed. Risk score: ${formatRiskScore(prediction.risk_score)}, 
           Category: ${prediction.risk_category}. 
           ${prediction.recommendations?.length || 0} clinical recommendations available.`
        )}
      </div>
    </Box>
  );
};

export default ResultsDashboard;