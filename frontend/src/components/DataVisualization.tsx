import React, { useMemo, useRef, useEffect, useState } from 'react';
import { Box, Typography, Card, CardContent, CardHeader, IconButton, Tooltip, useTheme } from '@mui/material';
import { FullscreenOutlined, DownloadOutlined, RefreshOutlined } from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
  TimeScale,
} from 'chart.js';
import { Line, Bar, Pie, Doughnut, Scatter } from 'react-chartjs-2';
import { motion } from 'framer-motion';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  ChartTooltip,
  Legend,
  Filler,
  TimeScale
);

// Types
interface DataVisualizationProps {
  type: 'line' | 'bar' | 'horizontal_bar' | 'pie' | 'doughnut' | 'scatter' | 'area' | 'confidence_interval';
  data: any[];
  title?: string;
  subtitle?: string;
  height?: number;
  width?: number;
  config?: {
    showLegend?: boolean;
    showTooltip?: boolean;
    responsive?: boolean;
    xAxisLabel?: string;
    yAxisLabel?: string;
    color?: string;
    colors?: string[];
    animation?: boolean;
    gridLines?: boolean;
    borderWidth?: number;
  };
  className?: string;
  onDataPointClick?: (dataPoint: any, index: number) => void;
  loading?: boolean;
  error?: string;
}

const DataVisualization: React.FC<DataVisualizationProps> = ({
  type,
  data,
  title,
  subtitle,
  height = 300,
  width,
  config = {},
  className,
  onDataPointClick,
  loading = false,
  error,
}) => {
  const theme = useTheme();
  const chartRef = useRef<any>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Default configuration
  const defaultConfig = {
    showLegend: true,
    showTooltip: true,
    responsive: true,
    animation: true,
    gridLines: true,
    borderWidth: 2,
    ...config,
  };

  // Color palette based on theme
  const colorPalette = useMemo(() => {
    const isDark = theme.palette.mode === 'dark';
    
    return {
      primary: theme.palette.primary.main,
      secondary: theme.palette.secondary.main,
      success: theme.palette.success.main,
      warning: theme.palette.warning.main,
      error: theme.palette.error.main,
      info: theme.palette.info.main,
      
      // Risk category colors
      riskColors: {
        low: theme.palette.success.main,
        medium: theme.palette.warning.main,
        high: theme.palette.error.main,
      },
      
      // Chart-specific colors
      chartColors: [
        theme.palette.primary.main,
        theme.palette.secondary.main,
        theme.palette.success.main,
        theme.palette.warning.main,
        theme.palette.error.main,
        theme.palette.info.main,
        '#9c27b0', // Purple
        '#ff5722', // Deep Orange
        '#607d8b', // Blue Grey
        '#795548', // Brown
      ],
      
      // Background colors with opacity
      backgroundColors: [
        `${theme.palette.primary.main}20`,
        `${theme.palette.secondary.main}20`,
        `${theme.palette.success.main}20`,
        `${theme.palette.warning.main}20`,
        `${theme.palette.error.main}20`,
        `${theme.palette.info.main}20`,
      ],
    };
  }, [theme]);

  // Process data based on chart type
  const processedData = useMemo(() => {
    if (!data || data.length === 0) {
      return null;
    }

    switch (type) {
      case 'line':
      case 'area':
        return {
          labels: data.map(item => item.name || item.x || item.label),
          datasets: [{
            label: title || 'Data',
            data: data.map(item => item.value || item.y),
            borderColor: config.color || colorPalette.primary,
            backgroundColor: type === 'area' 
              ? `${config.color || colorPalette.primary}20`
              : config.color || colorPalette.primary,
            borderWidth: defaultConfig.borderWidth,
            fill: type === 'area',
            tension: 0.4,
            pointBackgroundColor: config.color || colorPalette.primary,
            pointBorderColor: theme.palette.background.paper,
            pointBorderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8,
          }],
        };

      case 'bar':
        return {
          labels: data.map(item => item.name || item.label),
          datasets: [{
            label: title || 'Data',
            data: data.map(item => item.value || item.y),
            backgroundColor: config.colors || data.map((_, index) => 
              colorPalette.chartColors[index % colorPalette.chartColors.length]
            ),
            borderColor: config.colors || data.map((_, index) => 
              colorPalette.chartColors[index % colorPalette.chartColors.length]
            ),
            borderWidth: 1,
            borderRadius: 8,
          }],
        };

      case 'horizontal_bar':
        return {
          labels: data.map(item => item.name || item.label),
          datasets: [{
            label: title || 'Importance',
            data: data.map(item => item.value || item.y),
            backgroundColor: data.map(item => {
              if (item.category === 'demographics') return colorPalette.info;
              if (item.category === 'medical_history') return colorPalette.error;
              if (item.category === 'clinical_measures') return colorPalette.warning;
              if (item.category === 'utilization') return colorPalette.secondary;
              return colorPalette.primary;
            }),
            borderWidth: 0,
            borderRadius: 4,
          }],
        };

      case 'pie':
      case 'doughnut':
        return {
          labels: data.map(item => item.name || item.label),
          datasets: [{
            data: data.map(item => item.value || item.y),
            backgroundColor: data.map(item => {
              if (item.category === 'low') return colorPalette.riskColors.low;
              if (item.category === 'medium') return colorPalette.riskColors.medium;
              if (item.category === 'high') return colorPalette.riskColors.high;
              return colorPalette.chartColors[data.indexOf(item) % colorPalette.chartColors.length];
            }),
            borderColor: theme.palette.background.paper,
            borderWidth: 2,
            hoverOffset: 8,
          }],
        };

      case 'scatter':
        return {
          datasets: [{
            label: title || 'Data Points',
            data: data.map(item => ({
              x: item.x,
              y: item.y,
            })),
            backgroundColor: config.color || colorPalette.primary,
            borderColor: config.color || colorPalette.primary,
            pointRadius: 6,
            pointHoverRadius: 8,
          }],
        };

      case 'confidence_interval':
        const confidenceData = data[0];
        return {
          labels: ['Risk Score'],
          datasets: [
            {
              label: 'Confidence Interval',
              data: [confidenceData.lower, confidenceData.upper],
              backgroundColor: `${colorPalette.primary}40`,
              borderColor: colorPalette.primary,
              type: 'bar' as const,
            },
            {
              label: 'Predicted Risk',
              data: [confidenceData.actual],
              backgroundColor: colorPalette.error,
              borderColor: colorPalette.error,
              type: 'scatter' as const,
              pointRadius: 8,
              pointHoverRadius: 10,
            },
          ],
        };

      default:
        return null;
    }
  }, [data, type, title, config, colorPalette, theme, defaultConfig.borderWidth]);

  // Chart options
  const chartOptions = useMemo(() => {
    const baseOptions = {
      responsive: defaultConfig.responsive,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index' as const,
      },
      animation: defaultConfig.animation ? {
        duration: 1000,
        easing: 'easeOutQuart' as const,
      } : false,
      plugins: {
        legend: {
          display: defaultConfig.showLegend,
          position: 'top' as const,
          labels: {
            color: theme.palette.text.primary,
            usePointStyle: true,
            padding: 20,
            font: {
              size: 12,
              family: theme.typography.fontFamily,
            },
          },
        },
        tooltip: {
          enabled: defaultConfig.showTooltip,
          backgroundColor: theme.palette.background.paper,
          titleColor: theme.palette.text.primary,
          bodyColor: theme.palette.text.primary,
          borderColor: theme.palette.divider,
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          usePointStyle: true,
          callbacks: {
            label: (context: any) => {
              const label = context.dataset.label || '';
              const value = context.parsed.y || context.parsed;
              
              if (type === 'confidence_interval') {
                return `${label}: ${(value * 100).toFixed(1)}%`;
              }
              
              if (type === 'horizontal_bar') {
                return `${label}: ${value.toFixed(1)}%`;
              }
              
              return `${label}: ${value}`;
            },
          },
        },
        title: {
          display: !!title,
          text: title,
          color: theme.palette.text.primary,
          font: {
            size: 16,
            weight: '600' as const,
            family: theme.typography.fontFamily,
          },
          padding: {
            bottom: 20,
          },
        },
        subtitle: {
          display: !!subtitle,
          text: subtitle,
          color: theme.palette.text.secondary,
          font: {
            size: 12,
            family: theme.typography.fontFamily,
          },
        },
      },
      scales: type.includes('bar') || type === 'line' || type === 'area' || type === 'scatter' ? {
        x: {
          display: true,
          title: {
            display: !!defaultConfig.xAxisLabel,
            text: defaultConfig.xAxisLabel,
            color: theme.palette.text.primary,
            font: {
              size: 12,
              weight: '500' as const,
            },
          },
          grid: {
            display: defaultConfig.gridLines,
            color: theme.palette.divider,
          },
          ticks: {
            color: theme.palette.text.secondary,
            font: {
              size: 11,
            },
          },
        },
        y: {
          display: true,
          title: {
            display: !!defaultConfig.yAxisLabel,
            text: defaultConfig.yAxisLabel,
            color: theme.palette.text.primary,
            font: {
              size: 12,
              weight: '500' as const,
            },
          },
          grid: {
            display: defaultConfig.gridLines,
            color: theme.palette.divider,
          },
          ticks: {
            color: theme.palette.text.secondary,
            font: {
              size: 11,
            },
          },
        },
      } : undefined,
      onClick: onDataPointClick ? (event: any, elements: any[]) => {
        if (elements.length > 0) {
          const element = elements[0];
          const dataIndex = element.index;
          onDataPointClick(data[dataIndex], dataIndex);
        }
      } : undefined,
    };

    // Special handling for horizontal bar charts
    if (type === 'horizontal_bar') {
      return {
        ...baseOptions,
        indexAxis: 'y' as const,
        scales: {
          x: {
            beginAtZero: true,
            title: {
              display: !!defaultConfig.xAxisLabel,
              text: defaultConfig.xAxisLabel || 'Importance (%)',
              color: theme.palette.text.primary,
            },
            grid: {
              display: defaultConfig.gridLines,
              color: theme.palette.divider,
            },
            ticks: {
              color: theme.palette.text.secondary,
              callback: (value: any) => `${value}%`,
            },
          },
          y: {
            title: {
              display: !!defaultConfig.yAxisLabel,
              text: defaultConfig.yAxisLabel,
              color: theme.palette.text.primary,
            },
            grid: {
              display: false,
            },
            ticks: {
              color: theme.palette.text.secondary,
              font: {
                size: 10,
              },
            },
          },
        },
      };
    }

    return baseOptions;
  }, [
    defaultConfig, 
    theme, 
    title, 
    subtitle, 
    type, 
    onDataPointClick, 
    data
  ]);

  // Download chart as image
  const handleDownload = useCallback(() => {
    if (chartRef.current) {
      const canvas = chartRef.current.canvas;
      const link = document.createElement('a');
      link.download = `${title || 'chart'}_${new Date().toISOString().split('T')[0]}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  }, [title]);

  // Toggle fullscreen
  const handleFullscreen = useCallback(() => {
    setIsFullscreen(!isFullscreen);
  }, [isFullscreen]);

  // Render chart component based on type
  const renderChart = () => {
    if (!processedData) {
      return (
        <Box 
          sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: height,
            color: 'text.secondary'
          }}
        >
          <Typography>No data available</Typography>
        </Box>
      );
    }

    const chartProps = {
      ref: chartRef,
      data: processedData,
      options: chartOptions,
      height: height,
      width: width,
    };

    switch (type) {
      case 'line':
      case 'area':
        return <Line {...chartProps} />;
      
      case 'bar':
      case 'horizontal_bar':
        return <Bar {...chartProps} />;
      
      case 'pie':
        return <Pie {...chartProps} />;
      
      case 'doughnut':
        return <Doughnut {...chartProps} />;
      
      case 'scatter':
        return <Scatter {...chartProps} />;
      
      case 'confidence_interval':
        return <Bar {...chartProps} />;
      
      default:
        return (
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: height,
            color: 'text.secondary'
          }}>
            <Typography>Unsupported chart type: {type}</Typography>
          </Box>
        );
    }
  };

  // Loading state
  if (loading) {
    return (
      <Card className={className} elevation={2}>
        {(title || subtitle) && (
          <CardHeader
            title={title}
            subheader={subtitle}
          />
        )}
        <CardContent>
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: height,
            flexDirection: 'column',
            gap: 2
          }}>
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            >
              <RefreshOutlined sx={{ fontSize: 48, color: 'primary.main' }} />
            </motion.div>
            <Typography color="text.secondary">Loading chart data...</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error) {
    return (
      <Card className={className} elevation={2} sx={{ border: '1px solid', borderColor: 'error.main' }}>
        {(title || subtitle) && (
          <CardHeader
            title={title}
            subheader={subtitle}
          />
        )}
        <CardContent>
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: height,
            flexDirection: 'column',
            gap: 2,
            color: 'error.main'
          }}>
            <Typography variant="h6" color="error">Chart Error</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
              {error}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={className}
    >
      <Card 
        elevation={2} 
        sx={{ 
          position: isFullscreen ? 'fixed' : 'relative',
          top: isFullscreen ? 0 : 'auto',
          left: isFullscreen ? 0 : 'auto',
          width: isFullscreen ? '100vw' : 'auto',
          height: isFullscreen ? '100vh' : 'auto',
          zIndex: isFullscreen ? 9999 : 'auto',
          borderRadius: isFullscreen ? 0 : undefined,
        }}
      >
        {/* Card header with actions */}
        {(title || subtitle || showActions) && (
          <CardHeader
            title={title}
            subheader={subtitle}
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Tooltip title="Download as PNG">
                  <IconButton onClick={handleDownload} size="small">
                    <DownloadOutlined />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title={isFullscreen ? "Exit fullscreen" : "View fullscreen"}>
                  <IconButton onClick={handleFullscreen} size="small">
                    <FullscreenOutlined />
                  </IconButton>
                </Tooltip>
              </Box>
            }
            sx={{
              borderBottom: '1px solid',
              borderColor: 'divider',
              '& .MuiCardHeader-title': {
                fontSize: '1.1rem',
                fontWeight: 600,
              },
              '& .MuiCardHeader-subheader': {
                fontSize: '0.875rem',
              },
            }}
          />
        )}

        {/* Chart content */}
        <CardContent sx={{ 
          position: 'relative',
          height: isFullscreen ? 'calc(100vh - 100px)' : height + 40,
          p: 3,
        }}>
          <Box sx={{ 
            height: isFullscreen ? '100%' : height,
            width: '100%',
            position: 'relative'
          }}>
            {renderChart()}
          </Box>

          {/* Data summary for accessibility */}
          <Box className="sr-only" aria-live="polite">
            <Typography>
              Chart showing {type} visualization with {data?.length || 0} data points.
              {title && ` Title: ${title}.`}
              {data && data.length > 0 && (
                ` Highest value: ${Math.max(...data.map(d => d.value || d.y || 0)).toFixed(2)}.
                  Lowest value: ${Math.min(...data.map(d => d.value || d.y || 0)).toFixed(2)}.`
              )}
            </Typography>
          </Box>
        </CardContent>

        {/* Data table for accessibility (hidden by default) */}
        {data && data.length > 0 && (
          <details style={{ margin: '16px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 600, marginBottom: '8px' }}>
              View Data Table
            </summary>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small" aria-label="Chart data table">
                <TableHead>
                  <TableRow>
                    <TableCell>Label</TableCell>
                    <TableCell align="right">Value</TableCell>
                    {data[0]?.category && <TableCell>Category</TableCell>}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.map((item, index) => (
                    <TableRow key={index}>
                      <TableCell component="th" scope="row">
                        {item.name || item.label || `Item ${index + 1}`}
                      </TableCell>
                      <TableCell align="right">
                        {typeof (item.value || item.y) === 'number' 
                          ? (item.value || item.y).toFixed(2)
                          : item.value || item.y
                        }
                      </TableCell>
                      {item.category && (
                        <TableCell>{item.category}</TableCell>
                      )}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </details>
        )}
      </Card>

      {/* Fullscreen overlay backdrop */}
      {isFullscreen && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            zIndex: 9998,
          }}
          onClick={handleFullscreen}
        />
      )}
    </motion.div>
  );
};

// Specialized chart components for common use cases
export const RiskDistributionChart: React.FC<{
  lowCount: number;
  mediumCount: number;
  highCount: number;
  title?: string;
}> = ({ lowCount, mediumCount, highCount, title = "Risk Distribution" }) => {
  const data = [
    { name: 'Low Risk', value: lowCount, category: 'low' },
    { name: 'Medium Risk', value: mediumCount, category: 'medium' },
    { name: 'High Risk', value: highCount, category: 'high' },
  ];

  return (
    <DataVisualization
      type="doughnut"
      data={data}
      title={title}
      config={{
        showLegend: true,
        showTooltip: true,
      }}
      height={250}
    />
  );
};

export const FeatureImportanceChart: React.FC<{
  features: FeatureImportance[];
  title?: string;
  maxFeatures?: number;
}> = ({ features, title = "Feature Importance", maxFeatures = 10 }) => {
  const data = features
    .slice(0, maxFeatures)
    .map(feature => ({
      name: feature.description || feature.feature_name.replace(/_/g, ' '),
      value: feature.importance_score * 100,
      category: feature.category || 'other'
    }));

  return (
    <DataVisualization
      type="horizontal_bar"
      data={data}
      title={title}
      config={{
        showLegend: false,
        xAxisLabel: 'Importance (%)',
        yAxisLabel: 'Features',
      }}
      height={Math.max(250, data.length * 40)}
    />
  );
};

export const RiskTrendChart: React.FC<{
  trendData: Array<{ date: string; riskScore: number }>;
  title?: string;
}> = ({ trendData, title = "Risk Score Trend" }) => {
  const data = trendData.map(item => ({
    name: item.date,
    value: item.riskScore,
    x: item.date,
    y: item.riskScore,
  }));

  return (
    <DataVisualization
      type="line"
      data={data}
      title={title}
      config={{
        showLegend: false,
        xAxisLabel: 'Date',
        yAxisLabel: 'Risk Score',
        animation: true,
      }}
      height={300}
    />
  );
};

export const ConfidenceIntervalChart: React.FC<{
  prediction: PredictionResponse;
  title?: string;
}> = ({ prediction, title = "Prediction Confidence" }) => {
  if (!prediction.confidence_lower || !prediction.confidence_upper) {
    return null;
  }

  const data = [{
    name: 'Risk Score',
    lower: prediction.confidence_lower,
    upper: prediction.confidence_upper,
    actual: prediction.risk_score,
  }];

  return (
    <DataVisualization
      type="confidence_interval"
      data={data}
      title={title}
      config={{
        showLegend: true,
        yAxisLabel: 'Risk Score',
      }}
      height={250}
    />
  );
};

export default DataVisualization;