import React from 'react';
import { 
  Box, 
  CircularProgress, 
  LinearProgress, 
  Typography, 
  Skeleton,
  useTheme,
  keyframes,
  styled
} from '@mui/material';
import { motion } from 'framer-motion';

// Types
interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  variant?: 'circular' | 'linear' | 'dots' | 'pulse' | 'skeleton';
  message?: string;
  progress?: number;
  color?: 'primary' | 'secondary' | 'inherit';
  overlay?: boolean;
  fullScreen?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

// Styled components for custom animations
const BouncingDots = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: 4,
  '& > div': {
    width: 8,
    height: 8,
    borderRadius: '50%',
    backgroundColor: theme.palette.primary.main,
    animation: `${keyframes`
      0%, 80%, 100% {
        transform: scale(0);
        opacity: 0.5;
      }
      40% {
        transform: scale(1);
        opacity: 1;
      }
    `} 1.4s infinite ease-in-out both`,
    
    '&:nth-of-type(1)': { animationDelay: '-0.32s' },
    '&:nth-of-type(2)': { animationDelay: '-0.16s' },
    '&:nth-of-type(3)': { animationDelay: '0s' },
  },
}));

const PulseContainer = styled(Box)(({ theme }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  animation: `${keyframes`
    0% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.1);
      opacity: 0.7;
    }
    100% {
      transform: scale(1);
      opacity: 1;
    }
  `} 2s ease-in-out infinite`,
}));

const HeartbeatIcon = styled(Box)(({ theme }) => ({
  width: 24,
  height: 24,
  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='${encodeURIComponent(theme.palette.primary.main)}'%3E%3Cpath d='M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z'/%3E%3C/svg%3E")`,
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'center',
  backgroundSize: 'contain',
  animation: `${keyframes`
    0% {
      transform: scale(1);
    }
    14% {
      transform: scale(1.3);
    }
    28% {
      transform: scale(1);
    }
    42% {
      transform: scale(1.3);
    }
    70% {
      transform: scale(1);
    }
  `} 1.5s ease-in-out infinite`,
}));

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'medium',
  variant = 'circular',
  message,
  progress,
  color = 'primary',
  overlay = false,
  fullScreen = false,
  className,
  style,
}) => {
  const theme = useTheme();

  // Size configurations
  const sizeConfig = {
    small: {
      circular: 20,
      linear: { height: 4, width: 100 },
      dots: 6,
      fontSize: '0.75rem',
    },
    medium: {
      circular: 40,
      linear: { height: 6, width: 200 },
      dots: 8,
      fontSize: '0.875rem',
    },
    large: {
      circular: 60,
      linear: { height: 8, width: 300 },
      dots: 12,
      fontSize: '1rem',
    },
  };

  const currentSize = sizeConfig[size];

  // Render different loading variants
  const renderSpinner = () => {
    switch (variant) {
      case 'circular':
        return (
          <CircularProgress
            size={currentSize.circular}
            color={color}
            variant={progress !== undefined ? 'determinate' : 'indeterminate'}
            value={progress}
            thickness={4}
            sx={{
              animation: progress === undefined ? undefined : 'none',
            }}
          />
        );

      case 'linear':
        return (
          <LinearProgress
            color={color}
            variant={progress !== undefined ? 'determinate' : 'indeterminate'}
            value={progress}
            sx={{
              width: currentSize.linear.width,
              height: currentSize.linear.height,
              borderRadius: currentSize.linear.height / 2,
              backgroundColor: `${theme.palette[color].main}20`,
            }}
          />
        );

      case 'dots':
        return (
          <BouncingDots>
            <div />
            <div />
            <div />
          </BouncingDots>
        );

      case 'pulse':
        return (
          <PulseContainer>
            <HeartbeatIcon sx={{ 
              width: currentSize.circular, 
              height: currentSize.circular 
            }} />
          </PulseContainer>
        );

      case 'skeleton':
        return (
          <Box sx={{ width: currentSize.linear.width }}>
            <Skeleton variant="text" height={currentSize.linear.height * 4} />
            <Skeleton variant="rectangular" height={currentSize.linear.height * 8} sx={{ mt: 1 }} />
            <Skeleton variant="text" height={currentSize.linear.height * 3} sx={{ mt: 1 }} />
          </Box>
        );

      default:
        return (
          <CircularProgress
            size={currentSize.circular}
            color={color}
          />
        );
    }
  };

  // Content container
  const content = (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.3 }}
      style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}
    >
      {renderSpinner()}
      
      {message && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{
            fontSize: currentSize.fontSize,
            textAlign: 'center',
            maxWidth: 200,
            lineHeight: 1.4,
          }}
        >
          {message}
        </Typography>
      )}

      {progress !== undefined && variant !== 'circular' && variant !== 'linear' && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontSize: '0.75rem' }}
        >
          {Math.round(progress)}%
        </Typography>
      )}
    </motion.div>
  );

  // Wrapper styling based on props
  const wrapperSx = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    ...(overlay && {
      position: 'absolute' as const,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(255, 255, 255, 0.8)',
      zIndex: 1000,
    }),
    ...(fullScreen && {
      position: 'fixed' as const,
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      backgroundColor: theme.palette.mode === 'dark' 
        ? 'rgba(0, 0, 0, 0.9)' 
        : 'rgba(255, 255, 255, 0.9)',
      zIndex: 9999,
    }),
    ...style,
  };

  return (
    <Box className={className} sx={wrapperSx}>
      {content}
    </Box>
  );
};

// Specialized loading components for common use cases
export const PageLoadingSpinner: React.FC<{ message?: string }> = ({ 
  message = 'Loading page...' 
}) => (
  <LoadingSpinner
    size="large"
    variant="pulse"
    message={message}
    fullScreen
  />
);

export const ComponentLoadingSpinner: React.FC<{ 
  message?: string;
  height?: number;
}> = ({ 
  message = 'Loading...', 
  height = 200 
}) => (
  <Box sx={{ 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: height,
    width: '100%'
  }}>
    <LoadingSpinner
      size="medium"
      variant="circular"
      message={message}
    />
  </Box>
);

export const ButtonLoadingSpinner: React.FC = () => (
  <LoadingSpinner
    size="small"
    variant="circular"
    color="inherit"
  />
);

export const TableLoadingSpinner: React.FC<{ rows?: number }> = ({ rows = 5 }) => (
  <Box sx={{ width: '100%' }}>
    {Array.from({ length: rows }).map((_, index) => (
      <Skeleton
        key={index}
        variant="rectangular"
        height={52}
        sx={{ 
          mb: 1,
          borderRadius: 1,
          '&:last-child': { mb: 0 }
        }}
      />
    ))}
  </Box>
);

export const CardLoadingSpinner: React.FC<{ 
  showHeader?: boolean;
  showContent?: boolean;
  height?: number;
}> = ({ 
  showHeader = true, 
  showContent = true, 
  height = 200 
}) => (
  <Box sx={{ p: 2 }}>
    {showHeader && (
      <Box sx={{ mb: 2 }}>
        <Skeleton variant="text" height={32} width="60%" />
        <Skeleton variant="text" height={20} width="40%" />
      </Box>
    )}
    
    {showContent && (
      <Skeleton variant="rectangular" height={height} sx={{ borderRadius: 2 }} />
    )}
  </Box>
);

export const FormLoadingSpinner: React.FC<{ fields?: number }> = ({ fields = 6 }) => (
  <Box sx={{ width: '100%' }}>
    {Array.from({ length: fields }).map((_, index) => (
      <Box key={index} sx={{ mb: 3 }}>
        <Skeleton variant="text" height={24} width="30%" sx={{ mb: 1 }} />
        <Skeleton variant="rectangular" height={56} sx={{ borderRadius: 1 }} />
      </Box>
    ))}
  </Box>
);

export const ChartLoadingSpinner: React.FC<{ 
  height?: number;
  showTitle?: boolean;
}> = ({ 
  height = 300, 
  showTitle = true 
}) => (
  <Box sx={{ width: '100%' }}>
    {showTitle && (
      <Skeleton variant="text" height={32} width="40%" sx={{ mb: 2 }} />
    )}
    <Skeleton variant="rectangular" height={height} sx={{ borderRadius: 2 }} />
  </Box>
);

// Progress indicator for file uploads
export const UploadProgressSpinner: React.FC<{
  progress: number;
  fileName?: string;
  status?: 'uploading' | 'processing' | 'completed' | 'error';
}> = ({ progress, fileName, status = 'uploading' }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'uploading': return 'primary';
      case 'processing': return 'warning';
      case 'completed': return 'success';
      case 'error': return 'error';
      default: return 'primary';
    }
  };

  const getStatusMessage = () => {
    switch (status) {
      case 'uploading': return 'Uploading file...';
      case 'processing': return 'Processing data...';
      case 'completed': return 'Upload completed';
      case 'error': return 'Upload failed';
      default: return 'Processing...';
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      gap: 2,
      p: 3
    }}>
      <Box sx={{ position: 'relative', display: 'inline-flex' }}>
        <CircularProgress
          variant="determinate"
          value={progress}
          size={60}
          thickness={4}
          color={getStatusColor()}
          sx={{
            '& .MuiCircularProgress-circle': {
              strokeLinecap: 'round',
            },
          }}
        />
        <Box
          sx={{
            top: 0,
            left: 0,
            bottom: 0,
            right: 0,
            position: 'absolute',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Typography
            variant="caption"
            component="div"
            color="text.secondary"
            sx={{ fontWeight: 600 }}
          >
            {Math.round(progress)}%
          </Typography>
        </Box>
      </Box>

      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="body2" color="text.primary" fontWeight={500}>
          {getStatusMessage()}
        </Typography>
        
        {fileName && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
            {fileName}
          </Typography>
        )}
      </Box>

      <LinearProgress
        variant="determinate"
        value={progress}
        color={getStatusColor()}
        sx={{
          width: 200,
          height: 6,
          borderRadius: 3,
          backgroundColor: `${getStatusColor()}.light`,
        }}
      />
    </Box>
  );
};

// Prediction loading with heartbeat animation
export const PredictionLoadingSpinner: React.FC<{
  message?: string;
  stage?: 'validating' | 'processing' | 'generating' | 'finalizing';
}> = ({ 
  message = 'Generating prediction...', 
  stage = 'processing' 
}) => {
  const stages = {
    validating: { progress: 25, message: 'Validating patient data...' },
    processing: { progress: 50, message: 'Processing features...' },
    generating: { progress: 75, message: 'Generating prediction...' },
    finalizing: { progress: 100, message: 'Finalizing results...' },
  };

  const currentStage = stages[stage];

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      gap: 3,
      p: 4
    }}>
      {/* Heartbeat animation */}
      <motion.div
        animate={{
          scale: [1, 1.2, 1, 1.2, 1],
        }}
        transition={{
          duration: 1.5,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      >
        <HeartbeatIcon sx={{ width: 48, height: 48 }} />
      </motion.div>

      {/* Progress indicator */}
      <Box sx={{ width: 240, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom fontWeight={600}>
          Cardiovascular Risk Assessment
        </Typography>
        
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {currentStage.message}
        </Typography>

        <LinearProgress
          variant="determinate"
          value={currentStage.progress}
          color="primary"
          sx={{
            mt: 2,
            height: 8,
            borderRadius: 4,
            backgroundColor: 'primary.light',
            '& .MuiLinearProgress-bar': {
              borderRadius: 4,
            },
          }}
        />

        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
          {currentStage.progress}% complete
        </Typography>
      </Box>

      {/* Custom message */}
      {message && message !== currentStage.message && (
        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          {message}
        </Typography>
      )}
    </Box>
  );
};

// Data loading skeleton for tables and lists
export const DataLoadingSkeleton: React.FC<{
  type?: 'table' | 'list' | 'grid' | 'chart';
  items?: number;
  height?: number;
}> = ({ 
  type = 'list', 
  items = 5, 
  height = 60 
}) => {
  const renderSkeleton = () => {
    switch (type) {
      case 'table':
        return (
          <Box>
            {/* Table header */}
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              {Array.from({ length: 4 }).map((_, index) => (
                <Skeleton key={index} variant="text" height={32} width="20%" />
              ))}
            </Box>
            
            {/* Table rows */}
            {Array.from({ length: items }).map((_, index) => (
              <Box key={index} sx={{ display: 'flex', gap: 2, mb: 1 }}>
                {Array.from({ length: 4 }).map((_, cellIndex) => (
                  <Skeleton key={cellIndex} variant="text" height={height} width="20%" />
                ))}
              </Box>
            ))}
          </Box>
        );

      case 'list':
        return (
          <Box>
            {Array.from({ length: items }).map((_, index) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <Skeleton variant="circular" width={40} height={40} />
                <Box sx={{ flexGrow: 1 }}>
                  <Skeleton variant="text" height={24} width="70%" />
                  <Skeleton variant="text" height={20} width="50%" />
                </Box>
              </Box>
            ))}
          </Box>
        );

      case 'grid':
        return (
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: 2 }}>
            {Array.from({ length: items }).map((_, index) => (
              <Box key={index}>
                <Skeleton variant="rectangular" height={height} sx={{ borderRadius: 2, mb: 1 }} />
                <Skeleton variant="text" height={24} width="80%" />
                <Skeleton variant="text" height={20} width="60%" />
              </Box>
            ))}
          </Box>
        );

      case 'chart':
        return (
          <Box>
            <Skeleton variant="text" height={32} width="40%" sx={{ mb: 2 }} />
            <Skeleton variant="rectangular" height={height} sx={{ borderRadius: 2 }} />
          </Box>
        );

      default:
        return <Skeleton variant="rectangular" height={height} />;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      {renderSkeleton()}
    </motion.div>
  );
};

// Error state with retry functionality
export const ErrorLoadingState: React.FC<{
  error: string;
  onRetry?: () => void;
  retryLabel?: string;
}> = ({ 
  error, 
  onRetry, 
  retryLabel = 'Try Again' 
}) => (
  <Box sx={{ 
    display: 'flex', 
    flexDirection: 'column', 
    alignItems: 'center', 
    gap: 2,
    p: 4,
    textAlign: 'center'
  }}>
    <Typography variant="h6" color="error" gutterBottom>
      Loading Failed
    </Typography>
    
    <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 400 }}>
      {error}
    </Typography>

    {onRetry && (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Box sx={{ mt: 2 }}>
          <button
            onClick={onRetry}
            style={{
              backgroundColor: theme.palette.primary.main,
              color: theme.palette.primary.contrastText,
              border: 'none',
              padding: '8px 16px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: 600,
            }}
          >
            {retryLabel}
          </button>
        </Box>
      </motion.div>
    )}
  </Box>
);

// Export all loading components
export {
  BouncingDots,
  PulseContainer,
  HeartbeatIcon,
};

export default LoadingSpinner;