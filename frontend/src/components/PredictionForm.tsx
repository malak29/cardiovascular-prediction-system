import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Grid,
  TextField,
  FormControl,
  FormLabel,
  FormGroup,
  FormControlLabel,
  FormHelperText,
  Select,
  MenuItem,
  Checkbox,
  Switch,
  Button,
  Typography,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Alert,
  Collapse,
  Chip,
  Tooltip,
  IconButton,
  LinearProgress,
  Divider,
  InputAdornment,
} from '@mui/material';
import {
  PersonOutline,
  LocalHospitalOutlined,
  MonitorHeartOutlined,
  ScienceOutlined,
  InfoOutlined,
  HelpOutlineOutlined,
  RestartAltOutlined,
  PredictionsOutlined,
  WarningAmberOutlined,
} from '@mui/icons-material';
import { useForm, Controller } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import toast from 'react-hot-toast';

// Types and hooks
import { PatientData, PatientFormData, PredictionResponse, GenderType, RaceEthnicityType, SmokingStatusType } from '../types';
import { usePrediction } from '../hooks/usePrediction';
import { useAppSettings } from '../hooks/useAppSettings';

// Components
import LoadingSpinner from './common/LoadingSpinner';
import ResultsDashboard from './ResultsDashboard';

// Validation schema
const patientDataSchema = yup.object({
  // Demographics (Required)
  age: yup
    .number()
    .required('Age is required')
    .min(18, 'Age must be at least 18')
    .max(120, 'Age must be less than 120'),
  gender: yup
    .string()
    .required('Gender is required')
    .oneOf(['male', 'female', 'other', 'unknown'], 'Invalid gender selection'),
  
  // Geographic (Optional)
  state_code: yup
    .string()
    .length(2, 'State code must be 2 characters')
    .matches(/^[A-Z]{2}$/, 'State code must be uppercase letters'),
  zip_code: yup
    .string()
    .matches(/^\d{5}(-\d{4})?$/, 'Invalid ZIP code format'),
  
  // Vital Signs (Optional but recommended)
  systolic_bp: yup
    .number()
    .min(50, 'Systolic BP must be at least 50 mmHg')
    .max(300, 'Systolic BP must be less than 300 mmHg'),
  diastolic_bp: yup
    .number()
    .min(30, 'Diastolic BP must be at least 30 mmHg')
    .max(200, 'Diastolic BP must be less than 200 mmHg'),
  
  // Lab Values (Optional)
  total_cholesterol: yup
    .number()
    .min(50, 'Total cholesterol must be at least 50 mg/dL')
    .max(500, 'Total cholesterol must be less than 500 mg/dL'),
  hdl_cholesterol: yup
    .number()
    .min(10, 'HDL cholesterol must be at least 10 mg/dL')
    .max(150, 'HDL cholesterol must be less than 150 mg/dL'),
  blood_glucose: yup
    .number()
    .min(30, 'Blood glucose must be at least 30 mg/dL')
    .max(600, 'Blood glucose must be less than 600 mg/dL'),
  hba1c: yup
    .number()
    .min(3, 'HbA1c must be at least 3%')
    .max(20, 'HbA1c must be less than 20%'),
  
  // BMI (Optional)
  bmi: yup
    .number()
    .min(10, 'BMI must be at least 10')
    .max(100, 'BMI must be less than 100'),
  
  // Hospitalizations
  cardiovascular_hospitalizations_last_year: yup
    .number()
    .min(0, 'Cannot be negative')
    .max(50, 'Seems unusually high'),
  total_hospitalizations_last_year: yup
    .number()
    .min(0, 'Cannot be negative')
    .max(50, 'Seems unusually high')
    .test(
      'hospitalization-consistency',
      'Total hospitalizations must be >= cardiovascular hospitalizations',
      function(value) {
        const cvHospitalizations = this.parent.cardiovascular_hospitalizations_last_year || 0;
        return !value || value >= cvHospitalizations;
      }
    ),
});

// Form steps configuration
const formSteps = [
  {
    label: 'Demographics',
    description: 'Basic patient information',
    icon: PersonOutline,
    fields: ['age', 'gender', 'race_ethnicity', 'state_code', 'zip_code'],
  },
  {
    label: 'Medical History',
    description: 'Previous conditions and diagnoses',
    icon: LocalHospitalOutlined,
    fields: [
      'has_hypertension', 'has_diabetes', 'has_heart_disease',
      'has_stroke_history', 'has_heart_attack_history', 'has_atrial_fibrillation',
      'has_heart_failure', 'has_kidney_disease', 'has_copd'
    ],
  },
  {
    label: 'Vital Signs',
    description: 'Current measurements and readings',
    icon: MonitorHeartOutlined,
    fields: ['systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi', 'smoking_status'],
  },
  {
    label: 'Laboratory Results',
    description: 'Blood tests and lab values',
    icon: ScienceOutlined,
    fields: ['total_cholesterol', 'hdl_cholesterol', 'blood_glucose', 'hba1c', 'creatinine'],
  },
];

interface PredictionFormProps {
  onPredictionComplete?: (prediction: PredictionResponse) => void;
  initialData?: Partial<PatientFormData>;
  className?: string;
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onPredictionComplete,
  initialData,
  className,
}) => {
  // State management
  const [activeStep, setActiveStep] = useState(0);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [formProgress, setFormProgress] = useState(0);

  // Hooks
  const { makePrediction, loading, error } = usePrediction();
  const { settings } = useAppSettings();

  // Form management
  const {
    control,
    handleSubmit,
    watch,
    setValue,
    getValues,
    trigger,
    formState: { errors, isValid, dirtyFields },
    reset,
  } = useForm<PatientFormData>({
    resolver: yupResolver(patientDataSchema),
    defaultValues: {
      // Demographics
      age: initialData?.age || '',
      gender: initialData?.gender || '',
      race_ethnicity: initialData?.race_ethnicity || '',
      state_code: initialData?.state_code || '',
      zip_code: initialData?.zip_code || '',
      
      // Medical History (default to false)
      has_hypertension: initialData?.has_hypertension || false,
      has_diabetes: initialData?.has_diabetes || false,
      has_heart_disease: initialData?.has_heart_disease || false,
      has_stroke_history: initialData?.has_stroke_history || false,
      has_heart_attack_history: initialData?.has_heart_attack_history || false,
      has_atrial_fibrillation: initialData?.has_atrial_fibrillation || false,
      has_heart_failure: initialData?.has_heart_failure || false,
      has_kidney_disease: initialData?.has_kidney_disease || false,
      has_copd: initialData?.has_copd || false,
      has_depression: initialData?.has_depression || false,
      
      // Insurance (default to standard Medicare)
      medicare_part_a: initialData?.medicare_part_a ?? true,
      medicare_part_b: initialData?.medicare_part_b ?? true,
      medicare_part_c: initialData?.medicare_part_c || false,
      medicare_part_d: initialData?.medicare_part_d || false,
      dual_eligible: initialData?.dual_eligible || false,
      
      // Utilization
      cardiovascular_hospitalizations_last_year: initialData?.cardiovascular_hospitalizations_last_year || 0,
      total_hospitalizations_last_year: initialData?.total_hospitalizations_last_year || 0,
      
      // Prediction options
      includeConfidence: initialData?.includeConfidence ?? settings?.predictions?.includeConfidenceByDefault ?? true,
      includeFeatures: initialData?.includeFeatures ?? settings?.predictions?.includeFeaturesByDefault ?? true,
      modelVersion: initialData?.modelVersion || settings?.predictions?.defaultModelVersion,
    },
    mode: 'onChange',
  });

  // Watch all form values for progress calculation
  const watchedValues = watch();

  // Calculate form completion progress
  useEffect(() => {
    const totalFields = Object.keys(patientDataSchema.fields).length;
    const completedFields = Object.keys(dirtyFields).length;
    const newProgress = Math.round((completedFields / totalFields) * 100);
    setFormProgress(newProgress);
  }, [dirtyFields]);

  // Form submission handler
  const onSubmit = useCallback(async (data: PatientFormData) => {
    try {
      console.log('Submitting prediction request:', data);

      // Prepare patient data (remove form-specific fields)
      const { includeConfidence, includeFeatures, modelVersion, ...patientData } = data;

      // Make prediction
      const result = await makePrediction({
        patient_data: patientData as PatientData,
        model_version: modelVersion,
        include_confidence: includeConfidence || false,
        include_features: includeFeatures || false,
      });

      setPredictionResult(result);
      
      // Callback for parent component
      if (onPredictionComplete) {
        onPredictionComplete(result);
      }

      // Show success notification
      toast.success('Prediction completed successfully!', {
        duration: 4000,
        icon: '🎯',
      });

    } catch (err: any) {
      console.error('Prediction failed:', err);
      
      // Show error notification
      toast.error(err.message || 'Prediction failed. Please try again.', {
        duration: 6000,
        icon: '❌',
      });
    }
  }, [makePrediction, onPredictionComplete]);

  // Step navigation
  const handleNext = useCallback(async () => {
    const currentStepFields = formSteps[activeStep]?.fields || [];
    const isStepValid = await trigger(currentStepFields as any);

    if (isStepValid) {
      setActiveStep((prev) => Math.min(prev + 1, formSteps.length - 1));
    } else {
      toast.error('Please correct the errors in this section before continuing.');
    }
  }, [activeStep, trigger]);

  const handleBack = useCallback(() => {
    setActiveStep((prev) => Math.max(prev - 1, 0));
  }, []);

  const handleReset = useCallback(() => {
    reset();
    setActiveStep(0);
    setPredictionResult(null);
    setFormProgress(0);
    toast.success('Form reset successfully');
  }, [reset]);

  // Calculate risk factors for current data
  const calculateCurrentRiskFactors = useCallback(() => {
    const values = getValues();
    const riskFactors = [];

    if (values.age && values.age >= 75) riskFactors.push('Advanced age');
    if (values.has_hypertension) riskFactors.push('Hypertension');
    if (values.has_diabetes) riskFactors.push('Diabetes');
    if (values.has_heart_disease) riskFactors.push('Heart disease');
    if (values.smoking_status === 'current') riskFactors.push('Current smoking');
    if (values.bmi && values.bmi >= 30) riskFactors.push('Obesity');
    if (values.systolic_bp && values.systolic_bp >= 140) riskFactors.push('Elevated blood pressure');

    return riskFactors;
  }, [getValues]);

  // Render form step content
  const renderStepContent = (stepIndex: number) => {
    const step = formSteps[stepIndex];
    const StepIcon = step.icon;

    return (
      <motion.div
        key={stepIndex}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.3 }}
      >
        <Card elevation={2} sx={{ mb: 3 }}>
          <CardHeader
            avatar={<StepIcon color="primary" />}
            title={step.label}
            subheader={step.description}
            sx={{ pb: 1 }}
          />
          <CardContent>
            {stepIndex === 0 && renderDemographicsStep()}
            {stepIndex === 1 && renderMedicalHistoryStep()}
            {stepIndex === 2 && renderVitalSignsStep()}
            {stepIndex === 3 && renderLabResultsStep()}
          </CardContent>
        </Card>
      </motion.div>
    );
  };

  // Demographics step
  const renderDemographicsStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} sm={6}>
        <Controller
          name="age"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="Age"
              type="number"
              fullWidth
              required
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'Patient age in years'}
              InputProps={{
                endAdornment: <InputAdornment position="end">years</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="gender"
          control={control}
          render={({ field, fieldState }) => (
            <FormControl fullWidth required error={!!fieldState.error}>
              <FormLabel>Gender</FormLabel>
              <Select {...field} displayEmpty>
                <MenuItem value="">Select gender</MenuItem>
                <MenuItem value="male">Male</MenuItem>
                <MenuItem value="female">Female</MenuItem>
                <MenuItem value="other">Other</MenuItem>
                <MenuItem value="unknown">Unknown</MenuItem>
              </Select>
              {fieldState.error && (
                <FormHelperText>{fieldState.error.message}</FormHelperText>
              )}
            </FormControl>
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="race_ethnicity"
          control={control}
          render={({ field }) => (
            <FormControl fullWidth>
              <FormLabel>Race/Ethnicity (Optional)</FormLabel>
              <Select {...field} displayEmpty>
                <MenuItem value="">Select race/ethnicity</MenuItem>
                <MenuItem value="white_non_hispanic">White (Non-Hispanic)</MenuItem>
                <MenuItem value="black_non_hispanic">Black (Non-Hispanic)</MenuItem>
                <MenuItem value="hispanic">Hispanic</MenuItem>
                <MenuItem value="asian_non_hispanic">Asian (Non-Hispanic)</MenuItem>
                <MenuItem value="american_indian_alaska_native">American Indian/Alaska Native</MenuItem>
                <MenuItem value="native_hawaiian_pacific_islander">Native Hawaiian/Pacific Islander</MenuItem>
                <MenuItem value="multiracial">Multiracial</MenuItem>
                <MenuItem value="other">Other</MenuItem>
                <MenuItem value="unknown">Unknown</MenuItem>
              </Select>
            </FormControl>
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="state_code"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="State"
              placeholder="MA"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'US state abbreviation (e.g., MA, CA, TX)'}
              inputProps={{ maxLength: 2, style: { textTransform: 'uppercase' } }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12}>
        <Controller
          name="zip_code"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="ZIP Code (Optional)"
              placeholder="02101"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'US ZIP code (5 or 9 digits)'}
            />
          )}
        />
      </Grid>
    </Grid>
  );

  // Medical history step
  const renderMedicalHistoryStep = () => (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <LocalHospitalOutlined color="primary" />
        Cardiovascular Conditions
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { name: 'has_hypertension', label: 'Hypertension (High Blood Pressure)' },
          { name: 'has_diabetes', label: 'Diabetes Mellitus' },
          { name: 'has_heart_disease', label: 'Heart Disease' },
          { name: 'has_stroke_history', label: 'History of Stroke' },
          { name: 'has_heart_attack_history', label: 'History of Heart Attack' },
          { name: 'has_atrial_fibrillation', label: 'Atrial Fibrillation' },
          { name: 'has_heart_failure', label: 'Heart Failure' },
        ].map((condition) => (
          <Grid item xs={12} sm={6} key={condition.name}>
            <Controller
              name={condition.name as keyof PatientFormData}
              control={control}
              render={({ field }) => (
                <FormControlLabel
                  control={
                    <Checkbox
                      {...field}
                      checked={!!field.value}
                      color="primary"
                    />
                  }
                  label={condition.label}
                />
              )}
            />
          </Grid>
        ))}
      </Grid>

      <Divider sx={{ my: 3 }} />

      <Typography variant="h6" gutterBottom>
        Other Conditions
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {[
          { name: 'has_kidney_disease', label: 'Kidney Disease' },
          { name: 'has_copd', label: 'COPD (Chronic Obstructive Pulmonary Disease)' },
          { name: 'has_depression', label: 'Depression' },
        ].map((condition) => (
          <Grid item xs={12} sm={6} key={condition.name}>
            <Controller
              name={condition.name as keyof PatientFormData}
              control={control}
              render={({ field }) => (
                <FormControlLabel
                  control={
                    <Checkbox
                      {...field}
                      checked={!!field.value}
                      color="primary"
                    />
                  }
                  label={condition.label}
                />
              )}
            />
          </Grid>
        ))}
      </Grid>

      <Divider sx={{ my: 3 }} />

      <Typography variant="h6" gutterBottom>
        Recent Hospitalizations
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6}>
          <Controller
            name="cardiovascular_hospitalizations_last_year"
            control={control}
            render={({ field, fieldState }) => (
              <TextField
                {...field}
                label="CV Hospitalizations (Last Year)"
                type="number"
                fullWidth
                error={!!fieldState.error}
                helperText={fieldState.error?.message || 'Number of cardiovascular-related hospitalizations'}
                InputProps={{
                  inputProps: { min: 0, max: 50 },
                  endAdornment: <InputAdornment position="end">visits</InputAdornment>,
                }}
              />
            )}
          />
        </Grid>

        <Grid item xs={12} sm={6}>
          <Controller
            name="total_hospitalizations_last_year"
            control={control}
            render={({ field, fieldState }) => (
              <TextField
                {...field}
                label="Total Hospitalizations (Last Year)"
                type="number"
                fullWidth
                error={!!fieldState.error}
                helperText={fieldState.error?.message || 'Total number of hospitalizations'}
                InputProps={{
                  inputProps: { min: 0, max: 50 },
                  endAdornment: <InputAdornment position="end">visits</InputAdornment>,
                }}
              />
            )}
          />
        </Grid>
      </Grid>
    </Box>
  );

  // Vital signs step
  const renderVitalSignsStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <MonitorHeartOutlined color="primary" />
          Blood Pressure
        </Typography>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="systolic_bp"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="Systolic Blood Pressure"
              type="number"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'Upper number in blood pressure reading'}
              InputProps={{
                endAdornment: <InputAdornment position="end">mmHg</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="diastolic_bp"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="Diastolic Blood Pressure"
              type="number"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'Lower number in blood pressure reading'}
              InputProps={{
                endAdornment: <InputAdornment position="end">mmHg</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="heart_rate"
          control={control}
          render={({ field }) => (
            <TextField
              {...field}
              label="Heart Rate (Optional)"
              type="number"
              fullWidth
              helperText="Resting heart rate"
              InputProps={{
                endAdornment: <InputAdornment position="end">bpm</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="bmi"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="BMI (Body Mass Index)"
              type="number"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'Body Mass Index (kg/m²)'}
              InputProps={{
                endAdornment: <InputAdornment position="end">kg/m²</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12}>
        <Controller
          name="smoking_status"
          control={control}
          render={({ field }) => (
            <FormControl fullWidth>
              <FormLabel>Smoking Status</FormLabel>
              <Select {...field} displayEmpty>
                <MenuItem value="">Select smoking status</MenuItem>
                <MenuItem value="never">Never smoked</MenuItem>
                <MenuItem value="former">Former smoker</MenuItem>
                <MenuItem value="current">Current smoker</MenuItem>
                <MenuItem value="unknown">Unknown</MenuItem>
              </Select>
            </FormControl>
          )}
        />
      </Grid>
    </Grid>
  );

  // Laboratory results step
  const renderLabResultsStep = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ScienceOutlined color="primary" />
          Cholesterol Panel
        </Typography>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="total_cholesterol"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="Total Cholesterol"
              type="number"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'Total cholesterol level'}
              InputProps={{
                endAdornment: <InputAdornment position="end">mg/dL</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="hdl_cholesterol"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="HDL Cholesterol"
              type="number"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'High-density lipoprotein (good cholesterol)'}
              InputProps={{
                endAdornment: <InputAdornment position="end">mg/dL</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
          Blood Sugar
        </Typography>
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="blood_glucose"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="Blood Glucose"
              type="number"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'Fasting blood glucose level'}
              InputProps={{
                endAdornment: <InputAdornment position="end">mg/dL</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12} sm={6}>
        <Controller
          name="hba1c"
          control={control}
          render={({ field, fieldState }) => (
            <TextField
              {...field}
              label="HbA1c"
              type="number"
              fullWidth
              error={!!fieldState.error}
              helperText={fieldState.error?.message || 'Hemoglobin A1c percentage'}
              InputProps={{
                endAdornment: <InputAdornment position="end">%</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>

      <Grid item xs={12}>
        <Controller
          name="creatinine"
          control={control}
          render={({ field }) => (
            <TextField
              {...field}
              label="Serum Creatinine (Optional)"
              type="number"
              fullWidth
              helperText="Kidney function indicator"
              InputProps={{
                endAdornment: <InputAdornment position="end">mg/dL</InputAdornment>,
              }}
            />
          )}
        />
      </Grid>
    </Grid>
  );

  // Advanced options
  const renderAdvancedOptions = () => (
    <Card elevation={1} sx={{ mt: 3, bgcolor: 'background.default' }}>
      <CardHeader
        title="Prediction Options"
        action={
          <Switch
            checked={showAdvancedOptions}
            onChange={(e) => setShowAdvancedOptions(e.target.checked)}
            color="primary"
          />
        }
        sx={{ pb: 1 }}
      />
      
      <Collapse in={showAdvancedOptions}>
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <Controller
                name="includeConfidence"
                control={control}
                render={({ field }) => (
                  <FormControlLabel
                    control={
                      <Checkbox
                        {...field}
                        checked={!!field.value}
                        color="primary"
                      />
                    }
                    label="Include Confidence Intervals"
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} sm={4}>
              <Controller
                name="includeFeatures"
                control={control}
                render={({ field }) => (
                  <FormControlLabel
                    control={
                      <Checkbox
                        {...field}
                        checked={!!field.value}
                        color="primary"
                      />
                    }
                    label="Include Feature Importance"
                  />
                )}
              />
            </Grid>

            <Grid item xs={12} sm={4}>
              <Controller
                name="modelVersion"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth size="small">
                    <FormLabel>Model Version</FormLabel>
                    <Select {...field} displayEmpty>
                      <MenuItem value="">Use Default Model</MenuItem>
                      <MenuItem value="ridge_v1.2.0">Ridge v1.2.0 (Recommended)</MenuItem>
                      <MenuItem value="lasso_v1.1.0">Lasso v1.1.0</MenuItem>
                      <MenuItem value="random_forest_v1.0.0">Random Forest v1.0.0</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
          </Grid>
        </CardContent>
      </Collapse>
    </Card>
  );

  // Current risk factors display
  const currentRiskFactors = calculateCurrentRiskFactors();

  if (predictionResult) {
    return (
      <Box className={className}>
        <ResultsDashboard
          prediction={predictionResult}
          onNewPrediction={() => {
            setPredictionResult(null);
            setActiveStep(0);
          }}
        />
      </Box>
    );
  }

  return (
    <Box className={className} sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 700 }}>
          Cardiovascular Risk Assessment
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          Evidence-based prediction for Medicare beneficiaries
        </Typography>
        
        {/* Progress indicator */}
        <Box sx={{ mt: 3, mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Form Progress: {formProgress}%
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={formProgress} 
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
      </Box>

      {/* Error display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => {}}>
              <Typography variant="h6">Prediction Error</Typography>
              {error}
            </Alert>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Current risk factors preview */}
      {currentRiskFactors.length > 0 && (
        <Card elevation={1} sx={{ mb: 3, bgcolor: 'warning.light', color: 'warning.contrastText' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <WarningAmberOutlined />
              Identified Risk Factors
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {currentRiskFactors.map((factor, index) => (
                <Chip
                  key={index}
                  label={factor}
                  size="small"
                  color="warning"
                  variant="outlined"
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Form Steps */}
      <form onSubmit={handleSubmit(onSubmit)}>
        <Stepper activeStep={activeStep} orientation="vertical">
          {formSteps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel>
                <Typography variant="h6">{step.label}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {step.description}
                </Typography>
              </StepLabel>
              <StepContent>
                {renderStepContent(index)}
                
                {/* Step navigation */}
                <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                  <Button
                    disabled={index === 0}
                    onClick={handleBack}
                    variant="outlined"
                  >
                    Back
                  </Button>
                  
                  {index === formSteps.length - 1 ? (
                    <Button
                      type="submit"
                      variant="contained"
                      disabled={loading || !isValid}
                      startIcon={loading ? <LoadingSpinner size="small" /> : <PredictionsOutlined />}
                      sx={{ minWidth: 200 }}
                    >
                      {loading ? 'Generating Prediction...' : 'Generate Prediction'}
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      onClick={handleNext}
                      disabled={loading}
                    >
                      Continue
                    </Button>
                  )}
                </Box>
              </StepContent>
            </Step>
          ))}
        </Stepper>

        {/* Advanced options */}
        {renderAdvancedOptions()}

        {/* Form actions */}
        <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            onClick={handleReset}
            variant="outlined"
            startIcon={<RestartAltOutlined />}
            disabled={loading}
          >
            Reset Form
          </Button>
          
          <Tooltip title="Get help with form completion">
            <IconButton color="primary" size="large">
              <HelpOutlineOutlined />
            </IconButton>
          </Tooltip>
        </Box>
      </form>

      {/* Form validation summary */}
      {Object.keys(errors).length > 0 && (
        <Card elevation={1} sx={{ mt: 3, bgcolor: 'error.light' }}>
          <CardContent>
            <Typography variant="h6" color="error" gutterBottom>
              Please correct the following errors:
            </Typography>
            <Box component="ul" sx={{ m: 0, pl: 3 }}>
              {Object.entries(errors).map(([field, error]) => (
                <Typography component="li" key={field} color="error" variant="body2">
                  {error.message}
                </Typography>
              ))}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default PredictionForm;