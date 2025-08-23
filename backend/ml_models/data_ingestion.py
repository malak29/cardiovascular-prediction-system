import logging
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import hashlib
import sqlite3
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import boto3
from botocore.exceptions import ClientError
import schedule
import time
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Base = declarative_base()

class DataSource(Base):
    """Database model for tracking data sources."""
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String(100), unique=True, nullable=False)
    url = Column(String(500))
    last_updated = Column(DateTime)
    data_hash = Column(String(64))
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)

class CVDDataIngestion:
    """
    Comprehensive data ingestion system for cardiovascular disease data.
    
    This class handles data collection from multiple sources including CDC,
    with automated validation, cleaning, and storage capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data ingestion system.
        
        Args:
            config: Configuration dictionary containing data sources and settings
        """
        self.config = config
        self.raw_data_path = Path(config.get('raw_data_path', 'data/raw/'))
        self.processed_data_path = Path(config.get('processed_data_path', 'data/processed/'))
        self.backup_path = Path(config.get('backup_path', 'data/backup/'))
        
        # Create directories
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        db_url = config.get('database_url', 'sqlite:///data_ingestion.db')
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # AWS S3 setup (optional)
        self.s3_client = None
        if config.get('aws_s3_bucket'):
            self.s3_client = boto3.client('s3')
            self.s3_bucket = config['aws_s3_bucket']
        
        # CDC data sources configuration
        self.cdc_sources = {
            'heart_disease_mortality': {
                'url': 'https://data.cdc.gov/api/views/6x7h-5gx2/rows.csv?accessType=DOWNLOAD',
                'description': 'Heart Disease Mortality Data Among US Adults',
                'frequency': 'monthly'
            },
            'behavioral_risk_factors': {
                'url': 'https://data.cdc.gov/api/views/acme-vg9e/rows.csv?accessType=DOWNLOAD',
                'description': 'Behavioral Risk Factor Surveillance System',
                'frequency': 'yearly'
            },
            'heart_disease_hospitalization': {
                'url': 'https://data.cdc.gov/api/views/3x8b-x4qx/rows.csv?accessType=DOWNLOAD',
                'description': 'Medicare Heart Disease Hospitalizations',
                'frequency': 'quarterly'
            },
            'chronic_disease_indicators': {
                'url': 'https://data.cdc.gov/api/views/g4ie-h725/rows.csv?accessType=DOWNLOAD',
                'description': 'Chronic Disease Indicators',
                'frequency': 'yearly'
            }
        }
        
        # Data validation rules
        self.validation_rules = {
            'required_columns': [
                'LocationDesc', 'DataValue', 'Question', 'Response'
            ],
            'numeric_columns': ['DataValue', 'LowConfidenceLimit', 'HighConfidenceLimit'],
            'categorical_columns': ['LocationAbbr', 'Topic', 'Question', 'Response'],
            'date_columns': ['YearStart', 'YearEnd'],
            'value_ranges': {
                'DataValue': (0, 100),  # Assuming percentage values
                'YearStart': (2000, 2030),
                'YearEnd': (2000, 2030)
            }
        }
    
    def fetch_cdc_data(self, source_name: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch data from CDC source.
        
        Args:
            source_name: Name of the CDC data source
            force_refresh: Force data refresh even if recently updated
            
        Returns:
            DataFrame containing the fetched data or None if failed
        """
        if source_name not in self.cdc_sources:
            logger.error(f"Unknown CDC source: {source_name}")
            return None
        
        source_info = self.cdc_sources[source_name]
        
        # Check if we need to fetch new data
        if not force_refresh and not self._should_fetch_data(source_name, source_info['frequency']):
            logger.info(f"Skipping {source_name} - data is up to date")
            return self._load_existing_data(source_name)
        
        try:
            logger.info(f"Fetching data from CDC source: {source_name}")
            
            # Set up request headers
            headers = {
                'User-Agent': 'CVD-Prediction-System/1.0',
                'Accept': 'text/csv'
            }
            
            # Make request with timeout and retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        source_info['url'], 
                        headers=headers,
                        timeout=300,  # 5 minutes timeout
                        stream=True
                    )
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {source_name}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(30)  # Wait 30 seconds before retry
            
            # Parse CSV data
            df = pd.read_csv(response.iter_lines(decode_unicode=True), low_memory=False)
            logger.info(f"Successfully fetched {len(df)} records from {source_name}")
            
            # Calculate data hash for change detection
            data_hash = hashlib.sha256(df.to_string().encode()).hexdigest()
            
            # Save raw data
            raw_file_path = self.raw_data_path / f"{source_name}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(raw_file_path, index=False)
            
            # Update tracking database
            self._update_data_source_tracking(source_name, source_info['url'], data_hash)
            
            # Upload to S3 if configured
            if self.s3_client:
                self._upload_to_s3(raw_file_path, f"raw_data/{source_name}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data from {source_name}: {str(e)}")
            return None
    
    def _should_fetch_data(self, source_name: str, frequency: str) -> bool:
        """
        Check if data should be fetched based on frequency and last update.
        
        Args:
            source_name: Name of the data source
            frequency: Update frequency (daily, weekly, monthly, quarterly, yearly)
            
        Returns:
            True if data should be fetched, False otherwise
        """
        # Check last update from database
        data_source = self.session.query(DataSource).filter_by(source_name=source_name).first()
        
        if not data_source:
            return True  # First time fetching
        
        last_updated = data_source.last_updated
        if not last_updated:
            return True
        
        # Calculate time difference based on frequency
        now = datetime.utcnow()
        time_thresholds = {
            'daily': timedelta(days=1),
            'weekly': timedelta(days=7),
            'monthly': timedelta(days=30),
            'quarterly': timedelta(days=90),
            'yearly': timedelta(days=365)
        }
        
        threshold = time_thresholds.get(frequency, timedelta(days=30))
        return now - last_updated > threshold
    
    def _load_existing_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """Load most recent existing data for a source."""
        pattern = f"{source_name}_*.csv"
        files = list(self.raw_data_path.glob(pattern))
        
        if not files:
            return None
        
        # Get most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        return pd.read_csv(latest_file)
    
    def _update_data_source_tracking(self, source_name: str, url: str, data_hash: str):
        """Update data source tracking in database."""
        data_source = self.session.query(DataSource).filter_by(source_name=source_name).first()
        
        if data_source:
            data_source.last_updated = datetime.utcnow()
            data_source.data_hash = data_hash
            data_source.url = url
        else:
            data_source = DataSource(
                source_name=source_name,
                url=url,
                last_updated=datetime.utcnow(),
                data_hash=data_hash
            )
            self.session.add(data_source)
        
        self.session.commit()
    
    def _upload_to_s3(self, file_path: Path, s3_key: str):
        """Upload file to S3."""
        try:
            self.s3_client.upload_file(
                str(file_path),
                self.s3_bucket,
                f"{s3_key}/{file_path.name}"
            )
            logger.info(f"Uploaded {file_path.name} to S3")
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame, source_name: str) -> Tuple[bool, List[str]]:
        """
        Validate fetched data against predefined rules.
        
        Args:
            df: DataFrame to validate
            source_name: Name of the data source
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append("Dataset is empty")
            return False, issues
        
        # Check for required columns (flexible matching)
        available_columns = set(df.columns.str.lower())
        required_columns = set(col.lower() for col in self.validation_rules['required_columns'])
        
        # Find partial matches for required columns
        missing_required = []
        for req_col in required_columns:
            if not any(req_col in avail_col for avail_col in available_columns):
                missing_required.append(req_col)
        
        if missing_required:
            issues.append(f"Missing or partially missing required columns: {missing_required}")
        
        # Validate numeric columns
        for col in self.validation_rules['numeric_columns']:
            if col in df.columns:
                # Check for non-numeric values
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                if numeric_col.isna().sum() > len(df) * 0.5:  # More than 50% non-numeric
                    issues.append(f"Column {col} has too many non-numeric values")
                
                # Check value ranges
                if col in self.validation_rules['value_ranges']:
                    min_val, max_val = self.validation_rules['value_ranges'][col]
                    valid_values = numeric_col.dropna()
                    if not valid_values.empty:
                        if (valid_values < min_val).any() or (valid_values > max_val).any():
                            issues.append(f"Column {col} has values outside expected range [{min_val}, {max_val}]")
        
        # Check for excessive missing data
        missing_percent = df.isnull().sum() / len(df) * 100
        high_missing_cols = missing_percent[missing_percent > 70].index.tolist()  # More than 70% missing
        if high_missing_cols:
            issues.append(f"Columns with >70% missing data: {high_missing_cols}")
        
        # Check for duplicate records
        duplicate_count = df.duplicated().sum()
        if duplicate_count > len(df) * 0.1:  # More than 10% duplicates
            issues.append(f"High number of duplicate records: {duplicate_count} ({duplicate_count/len(df)*100:.1f}%)")
        
        # Data freshness check (if applicable)
        date_columns = [col for col in df.columns if 'year' in col.lower() or 'date' in col.lower()]
        if date_columns:
            for date_col in date_columns:
                try:
                    dates = pd.to_numeric(df[date_col], errors='coerce').dropna()
                    if not dates.empty:
                        latest_year = dates.max()
                        current_year = datetime.now().year
                        if latest_year < current_year - 3:  # Data older than 3 years
                            issues.append(f"Data appears outdated - latest year in {date_col}: {latest_year}")
                except Exception:
                    pass
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_and_standardize_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Clean and standardize the raw data.
        
        Args:
            df: Raw DataFrame
            source_name: Name of the data source
            
        Returns:
            Cleaned and standardized DataFrame
        """
        logger.info(f"Cleaning and standardizing data for {source_name}")
        
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Handle common data quality issues
        
        # 1. Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all')  # Remove rows with all NaN
        df_clean = df_clean.loc[:, ~df_clean.isnull().all()]  # Remove columns with all NaN
        
        # 2. Standardize text fields
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()
            # Replace common variations
            df_clean[col] = df_clean[col].replace({
                'NAN': np.nan,
                'NULL': np.nan,
                'N/A': np.nan,
                'NONE': np.nan,
                '': np.nan
            })
        
        # 3. Handle numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Remove outliers (beyond 3 standard deviations)
            if col in df_clean.columns and not df_clean[col].empty:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                if not pd.isna(std_val) and std_val != 0:
                    outlier_mask = np.abs(df_clean[col] - mean_val) > 3 * std_val
                    df_clean.loc[outlier_mask, col] = np.nan
        
        # 4. Standardize location names
        if 'locationdesc' in df_clean.columns:
            df_clean['locationdesc'] = df_clean['locationdesc'].replace({
                'UNITED STATES': 'NATIONAL',
                'US': 'NATIONAL',
                'USA': 'NATIONAL'
            })
        
        # 5. Handle date columns
        date_columns = [col for col in df_clean.columns if 'year' in col or 'date' in col]
        for date_col in date_columns:
            try:
                df_clean[date_col] = pd.to_numeric(df_clean[date_col], errors='coerce')
            except Exception:
                pass
        
        # 6. Remove duplicates but keep the most recent
        if 'yearstart' in df_clean.columns:
            df_clean = df_clean.sort_values('yearstart', ascending=False).drop_duplicates(
                subset=[col for col in df_clean.columns if col != 'yearstart'], 
                keep='first'
            )
        else:
            df_clean = df_clean.drop_duplicates()
        
        # 7. Add metadata columns
        df_clean['data_source'] = source_name
        df_clean['ingestion_timestamp'] = datetime.utcnow()
        df_clean['data_version'] = datetime.now().strftime('%Y%m%d')
        
        logger.info(f"Cleaned data shape: {df_clean.shape} (original: {df.shape})")
        
        return df_clean
    
    def merge_data_sources(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple data sources into a unified dataset.
        
        Args:
            dataframes: Dictionary of source_name -> DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging multiple data sources...")
        
        if not dataframes:
            raise ValueError("No dataframes provided for merging")
        
        # Find common columns for merging
        common_columns = set.intersection(*[set(df.columns) for df in dataframes.values()])
        merge_columns = []
        
        # Potential merge keys
        potential_keys = ['locationdesc', 'locationabbr', 'yearstart', 'topic', 'question']
        for key in potential_keys:
            if key in common_columns:
                merge_columns.append(key)
        
        if not merge_columns:
            # If no common merge columns, concatenate vertically
            logger.warning("No common merge columns found. Concatenating datasets vertically.")
            merged_df = pd.concat(dataframes.values(), ignore_index=True, sort=False)
        else:
            logger.info(f"Merging on columns: {merge_columns}")
            
            # Start with the first dataframe
            source_names = list(dataframes.keys())
            merged_df = dataframes[source_names[0]].copy()
            
            # Progressively merge other dataframes
            for source_name in source_names[1:]:
                df = dataframes[source_name]
                
                # Perform outer join to preserve all data
                merged_df = merged_df.merge(
                    df,
                    on=merge_columns,
                    how='outer',
                    suffixes=('', f'_{source_name}')
                )
        
        # Remove duplicate columns (keeping the first occurrence)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        return merged_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to multiple formats.
        
        Args:
            df: Processed DataFrame
            filename: Base filename (without extension)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{filename}_{timestamp}"
        
        # Save as CSV
        csv_path = self.processed_data_path / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed data to {csv_path}")
        
        # Save as Parquet for better performance
        parquet_path = self.processed_data_path / f"{base_filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved processed data to {parquet_path}")
        
        # Create a "latest" symlink/copy
        latest_csv = self.processed_data_path / f"{filename}_latest.csv"
        latest_parquet = self.processed_data_path / f"{filename}_latest.parquet"
        
        df.to_csv(latest_csv, index=False)
        df.to_parquet(latest_parquet, index=False)
        
        # Upload to S3 if configured
        if self.s3_client:
            self._upload_to_s3(csv_path, "processed_data")
            self._upload_to_s3(parquet_path, "processed_data")
        
        # Backup old versions
        self._backup_old_files(filename)
    
    def _backup_old_files(self, filename: str, keep_versions: int = 5):
        """Backup old versions of processed files."""
        pattern = f"{filename}_*.csv"
        files = sorted(self.processed_data_path.glob(pattern), 
                      key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent versions
        for file_to_backup in files[keep_versions:]:
            backup_file = self.backup_path / file_to_backup.name
            file_to_backup.rename(backup_file)
            logger.info(f"Backed up {file_to_backup.name}")
    
    def run_full_ingestion_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete data ingestion pipeline.
        
        Returns:
            Summary of ingestion results
        """
        logger.info("Starting full data ingestion pipeline...")
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'sources_processed': 0,
            'sources_failed': 0,
            'total_records': 0,
            'validation_issues': [],
            'failed_sources': []
        }
        
        dataframes = {}
        
        # Fetch data from all CDC sources
        for source_name in self.cdc_sources.keys():
            try:
                logger.info(f"Processing source: {source_name}")
                
                # Fetch raw data
                df = self.fetch_cdc_data(source_name)
                
                if df is None:
                    logger.error(f"Failed to fetch data for {source_name}")
                    results['sources_failed'] += 1
                    results['failed_sources'].append(source_name)
                    continue
                
                # Validate data
                is_valid, issues = self.validate_data(df, source_name)
                if issues:
                    results['validation_issues'].extend([f"{source_name}: {issue}" for issue in issues])
                
                if not is_valid:
                    logger.warning(f"Data validation failed for {source_name}: {issues}")
                    # Continue processing despite validation issues
                
                # Clean and standardize
                df_clean = self.clean_and_standardize_data(df, source_name)
                dataframes[source_name] = df_clean
                
                results['sources_processed'] += 1
                results['total_records'] += len(df_clean)
                
                logger.info(f"Successfully processed {source_name}: {len(df_clean)} records")
                
            except Exception as e:
                logger.error(f"Error processing {source_name}: {str(e)}")
                results['sources_failed'] += 1
                results['failed_sources'].append(source_name)
        
        # Merge all sources if we have multiple
        if len(dataframes) > 1:
            try:
                merged_df = self.merge_data_sources(dataframes)
                self.save_processed_data(merged_df, 'cardiovascular_disease_data')
                results['merged_records'] = len(merged_df)
                logger.info("Successfully merged all data sources")
            except Exception as e:
                logger.error(f"Failed to merge data sources: {str(e)}")
                results['merge_failed'] = True
        elif len(dataframes) == 1:
            # Save the single source
            source_name = list(dataframes.keys())[0]
            self.save_processed_data(dataframes[source_name], 'cardiovascular_disease_data')
        
        logger.info("Data ingestion pipeline completed")
        return results
    
    def schedule_automatic_ingestion(self):
        """Set up scheduled automatic data ingestion."""
        logger.info("Setting up scheduled data ingestion...")
        
        # Schedule different frequencies for different sources
        schedule.every().day.at("02:00").do(self._scheduled_ingestion, frequency='daily')
        schedule.every().monday.at("03:00").do(self._scheduled_ingestion, frequency='weekly')
        schedule.every().month.do(self._scheduled_ingestion, frequency='monthly')
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    
    def _scheduled_ingestion(self, frequency: str):
        """Run scheduled ingestion for sources with specific frequency."""
        logger.info(f"Running scheduled ingestion for {frequency} sources")
        
        # Filter sources by frequency
        sources_to_process = [
            source for source, info in self.cdc_sources.items()
            if info['frequency'] == frequency
        ]
        
        if sources_to_process:
            # Run ingestion for specific sources
            for source in sources_to_process:
                try:
                    self.fetch_cdc_data(source)
                    logger.info(f"Scheduled ingestion completed for {source}")
                except Exception as e:
                    logger.error(f"Scheduled ingestion failed for {source}: {str(e)}")


def main():
    """Main function to run data ingestion."""
    # Configuration
    config = {
        'raw_data_path': 'data/raw/',
        'processed_data_path': 'data/processed/',
        'backup_path': 'data/backup/',
        'database_url': 'sqlite:///data_ingestion.db',
        # 'aws_s3_bucket': 'your-cvd-data-bucket',  # Uncomment and set for S3 support
    }
    
    # Initialize ingestion system
    ingestion_system = CVDDataIngestion(config)
    
    # Run full pipeline
    results = ingestion_system.run_full_ingestion_pipeline()
    
    # Print summary
    print("\n" + "="*50)
    print("DATA INGESTION SUMMARY")
    print("="*50)
    print(f"Sources Processed: {results['sources_processed']}")
    print(f"Sources Failed: {results['sources_failed']}")
    print(f"Total Records: {results['total_records']}")
    
    if results.get('merged_records'):
        print(f"Merged Records: {results['merged_records']}")
    
    if results['failed_sources']:
        print(f"Failed Sources: {', '.join(results['failed_sources'])}")
    
    if results['validation_issues']:
        print(f"\nValidation Issues:")
        for issue in results['validation_issues'][:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(results['validation_issues']) > 5:
            print(f"  ... and {len(results['validation_issues']) - 5} more issues")


if __name__ == "__main__":
    main()