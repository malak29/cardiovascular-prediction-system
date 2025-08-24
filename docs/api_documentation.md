# Cardiovascular Disease Prediction API Documentation

## Overview

The Cardiovascular Disease Prediction API provides machine learning-powered risk assessment capabilities for healthcare professionals and researchers. This RESTful API offers endpoints for individual predictions, batch processing, data management, and system health monitoring.

**Base URL**: `https://cvd-prediction.example.com/api/v1`  
**Current Version**: `v1.0.0`  
**Documentation Version**: `2024-01-15`

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Health Endpoints](#health-endpoints)
  - [Prediction Endpoints](#prediction-endpoints)
  - [Data Management](#data-management)
  - [Model Information](#model-information)
- [Data Models](#data-models)
- [Examples](#examples)
- [SDKs and Libraries](#sdks-and-libraries)

## Authentication

### API Key Authentication

All API requests require authentication using an API key. Include your API key in the request header:

```http
X-API-Key: your-api-key-here
```

### OAuth 2.0 (Enterprise)

For enterprise customers, OAuth 2.0 authentication is available:

```http
Authorization: Bearer your-oauth-token
```

### Request Signing (High Security)

For high-security environments, request signing is supported:

```http
Authorization: CVD-HMAC-SHA256 Credential=access_key_id,SignedHeaders=host;x-cvd-date,Signature=calculated_signature
X-CVD-Date: 20240115T120000Z
```

## Rate Limiting

API requests are rate-limited to ensure fair usage and system stability:

- **Standard Tier**: 1,000 requests per hour
- **Professional Tier**: 10,000 requests per hour  
- **Enterprise Tier**: 100,000 requests per hour

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 3600
```

When rate limits are exceeded, the API returns a `429 Too Many Requests` status.

## Response Format

All API responses follow a consistent JSON format:

### Success Response

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "timestamp": "2024-01-15T12:00:00Z",
  "correlation_id": "cvd-req-123456789",
  "api_version": "v1.0.0"
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid patient data provided",
    "details": {
      "field": "Age",
      "reason": "Age must be between 18 and 100"
    }
  },
  "timestamp": "2024-01-15T12:00:00Z",
  "correlation_id": "cvd-req-123456789",
  "api_version": "v1.0.0"
}
```

## Error Handling

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid input data |
| 401 | Unauthorized - Invalid or missing authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource does not exist |
| 422 | Unprocessable Entity - Validation errors |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - System maintenance |

### Error Codes

| Error Code | Description |
|------------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `MODEL_ERROR` | ML model prediction failed |
| `DATA_ERROR` | Data processing error |
| `AUTH_ERROR` | Authentication error |
| `RATE_LIMIT_ERROR` | Rate limit exceeded |
| `SYSTEM_ERROR` | Internal system error |

## Endpoints

## Health Endpoints

### GET /health

Basic health check endpoint.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T12:00:00Z",
    "version": "v1.0.0"
  }
}
```

### GET /health/detailed

Detailed health check with system metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "components": {
      "database": {
        "status": "healthy",
        "response_time": "5ms",
        "connection_pool": {
          "active": 3,
          "idle": 7,
          "max": 20
        }
      },
      "redis": {
        "status": "healthy",
        "response_time": "2ms",
        "memory_usage": "45MB"
      },
      "ml_models": {
        "status": "healthy",
        "loaded_models": 1,
        "model_version": "v1.0.0",
        "last_prediction": "2024-01-15T11:59:30Z"
      }
    },
    "system_metrics": {
      "cpu_usage": 12.5,
      "memory_usage": 68.3,
      "disk_usage": 45.2,
      "uptime": "72h 15m"
    }
  }
}
```

## Prediction Endpoints

### POST /predictions/predict

Generate a cardiovascular disease risk prediction for a single patient.

**Request Body:**
```json
{
  "patient_data": {
    "Age": 65,
    "Sex": "Male",
    "BMI": 28.5,
    "Smoking": true,
    "HighBP": true,
    "HighChol": true,
    "Diabetes": false,
    "PhysActivity": true,
    "HvyAlcoholConsump": false,
    "PhysHlth": 5,
    "MentHlth": 2,
    "NoDocbcCost": false,
    "GenHlth": "Good",
    "DiffWalk": false,
    "Stroke": false
  },
  "model_version": "v1.0.0",
  "return_explanation": true,
  "save_to_database": false
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": 1,
    "probability": 0.75,
    "risk_level": "High",
    "confidence": 0.92,
    "model_version": "v1.0.0",
    "explanation": {
      "top_risk_factors": [
        {
          "factor": "HighBP",
          "importance": 0.25,
          "description": "High blood pressure significantly increases cardiovascular risk"
        },
        {
          "factor": "Age",
          "importance": 0.22,
          "description": "Age is a major non-modifiable risk factor"
        },
        {
          "factor": "BMI",
          "importance": 0.18,
          "description": "BMI of 28.5 indicates overweight status"
        }
      ],
      "modifiable_factors": [
        "Smoking",
        "BMI",
        "PhysActivity"
      ],
      "recommendations": [
        "Consider smoking cessation programs",
        "Maintain healthy weight through diet and exercise",
        "Regular cardiovascular exercise recommended"
      ]
    },
    "processing_time": "145ms"
  }
}
```

### POST /predictions/batch

Generate predictions for multiple patients in a single request.

**Request Body:**
```json
{
  "predictions": [
    {
      "patient_id": "patient_001",
      "patient_data": {
        "Age": 65,
        "Sex": "Male",
        // ... other patient data
      }
    },
    {
      "patient_id": "patient_002",
      "patient_data": {
        "Age": 45,
        "Sex": "Female",
        // ... other patient data
      }
    }
  ],
  "batch_id": "batch_20240115_001",
  "model_version": "v1.0.0",
  "return_explanations": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_20240115_001",
    "predictions": [
      {
        "patient_id": "patient_001",
        "prediction": 1,
        "probability": 0.75,
        "risk_level": "High"
      },
      {
        "patient_id": "patient_002", 
        "prediction": 0,
        "probability": 0.25,
        "risk_level": "Low"
      }
    ],
    "summary": {
      "total_predictions": 2,
      "high_risk_count": 1,
      "medium_risk_count": 0,
      "low_risk_count": 1,
      "average_probability": 0.50,
      "processing_time": "280ms"
    }
  }
}
```

### GET /predictions/{prediction_id}

Retrieve a previously generated prediction by ID.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "pred_123456789",
    "prediction": 1,
    "probability": 0.75,
    "risk_level": "High",
    "patient_data": {
      // Original patient data (if permitted)
    },
    "created_at": "2024-01-15T12:00:00Z",
    "model_version": "v1.0.0"
  }
}
```

### GET /predictions/history

Retrieve prediction history with filtering and pagination.

**Query Parameters:**
- `limit`: Number of results (default: 50, max: 1000)
- `offset`: Pagination offset (default: 0)
- `start_date`: Filter by start date (ISO format)
- `end_date`: Filter by end date (ISO format)
- `risk_level`: Filter by risk level (High, Medium, Low)
- `model_version`: Filter by model version

**Response:**
```json
{
  "success": true,
  "data": {
    "predictions": [
      // Array of prediction objects
    ],
    "pagination": {
      "total": 1250,
      "limit": 50,
      "offset": 0,
      "has_next": true,
      "has_previous": false
    }
  }
}
```

## Data Management

### GET /data/statistics

Retrieve dataset statistics and insights.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_records": 100000,
    "date_range": {
      "start": "2020-01-01",
      "end": "2024-01-01"
    },
    "feature_statistics": {
      "Age": {
        "mean": 58.2,
        "std": 12.5,
        "min": 18,
        "max": 100,
        "percentiles": {
          "25": 48,
          "50": 58,
          "75": 68
        }
      },
      "BMI": {
        "mean": 26.8,
        "std": 4.2,
        "min": 15.0,
        "max": 50.0
      }
    },
    "target_distribution": {
      "positive_cases": 15000,
      "negative_cases": 85000,
      "positive_rate": 0.15
    }
  }
}
```

### GET /data/validate

Validate current dataset quality.

**Response:**
```json
{
  "success": true,
  "data": {
    "is_valid": true,
    "quality_score": 0.92,
    "last_validation": "2024-01-15T08:00:00Z",
    "issues": [],
    "recommendations": [
      "Consider additional feature engineering for Age variable"
    ],
    "metrics": {
      "completeness": 0.95,
      "consistency": 0.89,
      "accuracy": 0.92,
      "timeliness": 0.88
    }
  }
}
```

### POST /data/upload

Upload new training data (Enterprise tier only).

**Request:** Multipart form data with CSV file

**Response:**
```json
{
  "success": true,
  "data": {
    "upload_id": "upload_123456789",
    "filename": "new_training_data.csv",
    "records_processed": 5000,
    "validation_status": "passed",
    "estimated_training_time": "2-3 hours"
  }
}
```

## Model Information

### GET /models/info

Get information about the current model.

**Response:**
```json
{
  "success": true,
  "data": {
    "model_name": "cardiovascular_risk_predictor",
    "version": "v1.0.0",
    "model_type": "RandomForestClassifier",
    "training_date": "2024-01-10T10:00:00Z",
    "performance_metrics": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.88,
      "f1_score": 0.85,
      "roc_auc": 0.90,
      "specificity": 0.83
    },
    "feature_count": 45,
    "training_samples": 100000,
    "last_retrained": "2024-01-10T10:00:00Z"
  }
}
```

### GET /models/features

Get information about model features and their importance.

**Response:**
```json
{
  "success": true,
  "data": {
    "total_features": 45,
    "feature_importance": [
      {
        "feature": "Age",
        "importance": 0.25,
        "description": "Patient age in years"
      },
      {
        "feature": "HighBP",
        "importance": 0.22,
        "description": "High blood pressure indicator"
      }
    ],
    "feature_categories": {
      "demographic": ["Age", "Sex"],
      "medical_history": ["HighBP", "HighChol", "Diabetes", "Stroke"],
      "lifestyle": ["Smoking", "PhysActivity", "HvyAlcoholConsump"],
      "health_status": ["BMI", "PhysHlth", "MentHlth", "GenHlth"]
    }
  }
}
```

## Data Models

### Patient Data Schema

```json
{
  "Age": {
    "type": "integer",
    "minimum": 18,
    "maximum": 100,
    "description": "Patient age in years"
  },
  "Sex": {
    "type": "string",
    "enum": ["Male", "Female"],
    "description": "Patient biological sex"
  },
  "BMI": {
    "type": "number",
    "minimum": 10.0,
    "maximum": 50.0,
    "description": "Body Mass Index"
  },
  "Smoking": {
    "type": "boolean",
    "description": "Current or former smoker"
  },
  "HighBP": {
    "type": "boolean", 
    "description": "High blood pressure diagnosis"
  },
  "HighChol": {
    "type": "boolean",
    "description": "High cholesterol diagnosis"
  },
  "Diabetes": {
    "type": "boolean",
    "description": "Diabetes diagnosis"
  },
  "PhysActivity": {
    "type": "boolean",
    "description": "Regular physical activity"
  },
  "HvyAlcoholConsump": {
    "type": "boolean",
    "description": "Heavy alcohol consumption"
  },
  "PhysHlth": {
    "type": "integer",
    "minimum": 0,
    "maximum": 30,
    "description": "Days of poor physical health in past 30 days"
  },
  "MentHlth": {
    "type": "integer",
    "minimum": 0,
    "maximum": 30,
    "description": "Days of poor mental health in past 30 days"
  },
  "NoDocbcCost": {
    "type": "boolean",
    "description": "Could not see doctor because of cost"
  },
  "GenHlth": {
    "type": "string",
    "enum": ["Excellent", "Very good", "Good", "Fair", "Poor"],
    "description": "Self-reported general health"
  },
  "DiffWalk": {
    "type": "boolean",
    "description": "Difficulty walking or climbing stairs"
  },
  "Stroke": {
    "type": "boolean",
    "description": "History of stroke"
  }
}
```

## Examples

### cURL Examples

#### Basic Prediction
```bash
curl -X POST https://cvd-prediction.example.com/api/v1/predictions/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "patient_data": {
      "Age": 65,
      "Sex": "Male",
      "BMI": 28.5,
      "Smoking": true,
      "HighBP": true,
      "HighChol": true,
      "Diabetes": false,
      "PhysActivity": true,
      "HvyAlcoholConsump": false,
      "PhysHlth": 5,
      "MentHlth": 2,
      "NoDocbcCost": false,
      "GenHlth": "Good",
      "DiffWalk": false,
      "Stroke": false
    },
    "return_explanation": true
  }'
```

#### Health Check
```bash
curl -H "X-API-Key: your-api-key" \
  https://cvd-prediction.example.com/api/v1/health/detailed
```

### Python Example

```python
import requests
import json

# Configuration
API_BASE_URL = "https://cvd-prediction.example.com/api/v1"
API_KEY = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# Patient data
patient_data = {
    "Age": 65,
    "Sex": "Male", 
    "BMI": 28.5,
    "Smoking": True,
    "HighBP": True,
    "HighChol": True,
    "Diabetes": False,
    "PhysActivity": True,
    "HvyAlcoholConsump": False,
    "PhysHlth": 5,
    "MentHlth": 2,
    "NoDocbcCost": False,
    "GenHlth": "Good",
    "DiffWalk": False,
    "Stroke": False
}

# Make prediction request
payload = {
    "patient_data": patient_data,
    "return_explanation": True
}

response = requests.post(
    f"{API_BASE_URL}/predictions/predict",
    headers=headers,
    json=payload
)

if response.status_code == 200:
    result = response.json()
    if result["success"]:
        prediction_data = result["data"]
        print(f"Risk Level: {prediction_data['risk_level']}")
        print(f"Probability: {prediction_data['probability']:.2%}")
        
        # Display top risk factors
        if "explanation" in prediction_data:
            print("\nTop Risk Factors:")
            for factor in prediction_data["explanation"]["top_risk_factors"]:
                print(f"- {factor['factor']}: {factor['importance']:.1%}")
    else:
        print(f"Error: {result['error']['message']}")
else:
    print(f"HTTP Error: {response.status_code}")
```

### JavaScript Example

```javascript
const CVD_API_BASE = 'https://cvd-prediction.example.com/api/v1';
const API_KEY = 'your-api-key';

const headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
};

const patientData = {
    Age: 65,
    Sex: 'Male',
    BMI: 28.5,
    Smoking: true,
    HighBP: true,
    HighChol: true,
    Diabetes: false,
    PhysActivity: true,
    HvyAlcoholConsump: false,
    PhysHlth: 5,
    MentHlth: 2,
    NoDocbcCost: false,
    GenHlth: 'Good',
    DiffWalk: false,
    Stroke: false
};

// Make prediction
async function predictCardiovascularRisk(patientData) {
    try {
        const response = await fetch(`${CVD_API_BASE}/predictions/predict`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                patient_data: patientData,
                return_explanation: true
            })
        });

        const result = await response.json();
        
        if (result.success) {
            const prediction = result.data;
            console.log(`Risk Level: ${prediction.risk_level}`);
            console.log(`Probability: ${(prediction.probability * 100).toFixed(1)}%`);
            
            if (prediction.explanation) {
                console.log('\nTop Risk Factors:');
                prediction.explanation.top_risk_factors.forEach(factor => {
                    console.log(`- ${factor.factor}: ${(factor.importance * 100).toFixed(1)}%`);
                });
            }
            
            return prediction;
        } else {
            console.error('API Error:', result.error.message);
            throw new Error(result.error.message);
        }
    } catch (error) {
        console.error('Request failed:', error);
        throw error;
    }
}

// Usage
predictCardiovascularRisk(patientData)
    .then(prediction => {
        // Handle successful prediction
        console.log('Prediction completed successfully');
    })
    .catch(error => {
        // Handle error
        console.error('Prediction failed:', error);
    });
```

## SDKs and Libraries

### Official SDKs

- **Python SDK**: `pip install cvd-prediction-sdk`
- **JavaScript SDK**: `npm install cvd-prediction-sdk`
- **R Package**: `install.packages("cvdprediction")`

### Third-Party Libraries

- **Go Client**: Available on GitHub
- **Java Client**: Maven Central available
- **C# NuGet Package**: Available

## Webhooks

### Prediction Webhooks

Configure webhooks to receive notifications when predictions are completed:

```json
{
  "webhook_url": "https://your-app.com/webhooks/cvd-predictions",
  "events": ["prediction.completed", "batch.completed"],
  "secret": "webhook-secret-key"
}
```

Webhook payload example:
```json
{
  "event": "prediction.completed",
  "timestamp": "2024-01-15T12:00:00Z",
  "data": {
    "prediction_id": "pred_123456789",
    "patient_id": "patient_001",
    "risk_level": "High",
    "probability": 0.75
  }
}
```

## Support and Resources

- **API Status**: https://status.cvd-prediction.com
- **Developer Portal**: https://developers.cvd-prediction.com
- **Support**: support@cvd-prediction.com
- **GitHub**: https://github.com/cvd-prediction/api-examples
- **Community Forum**: https://community.cvd-prediction.com

## Changelog

### v1.0.0 (2024-01-15)
- Initial API release
- Basic prediction endpoints
- Health monitoring
- Batch processing support

---

*This documentation is updated regularly. Last updated: January 15, 2024*